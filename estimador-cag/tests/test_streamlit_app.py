"""Interfaz Streamlit: cliente HTTP contra la API real (LLM simulado) y la app con AppTest."""

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

import streamlit_app
from app.main import app
from app.prompts.loader import render_system_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.services import llm_service
from app.services.llm_service import LLMResult
from app.sessions.models import ProjectMetadata
from app.sessions.store import get_session_store
from streamlit_app import (
    ApiError,
    SessionLost,
    create_session,
    error_message,
    get_session,
    read_description,
    request_turn,
    validation_message,
)
from tests.documents import pdf_bytes

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
DESCRIPTION = "App móvil para un gimnasio: reserva de clases, pago de cuotas y avisos push."
ESTIMATION = "## Estimación: App del gimnasio\n\n### Desglose por fases\n\n| Fase | Horas |"


@pytest.fixture
def fake_llm(monkeypatch):
    """Estimaciones (arrays de mensajes) que recibe el modelo; el extractor devuelve la ficha."""
    calls = []

    async def fake_call(settings, messages):
        calls.append(messages)
        return LLMResult(
            text=ESTIMATION,
            finish_reason="completed",
            truncated=False,
            input_tokens=2000,
            output_tokens=500,
        )

    async def fake_parse(settings, messages, response_model):
        facts = ProjectMetadata(
            project_name="App del gimnasio",
            mentioned_technologies=[f"Tecnología {len(calls)}"],
            agreed_scope=f"Alcance tras el turno {len(calls)}",
        )
        return facts, LLMResult("{}", "completed", False, 900, 60)

    monkeypatch.setattr(llm_service, "_call_openai", fake_call)
    monkeypatch.setattr(llm_service, "_parse_openai", fake_parse)
    return calls


def unreachable_client() -> httpx.Client:
    def refuse(request):
        raise httpx.ConnectError("Connection refused", request=request)

    return httpx.Client(base_url="http://localhost:8000", transport=httpx.MockTransport(refuse))


# --- Cliente de la API ---


def test_client_sends_a_turn_with_its_attachments(fake_llm):
    client = TestClient(app)
    request = EstimationRequest(
        description=DESCRIPTION,
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.NARRATIVE,
    )
    spec = SimpleNamespace(
        name="spec.pdf", type="application/pdf", getvalue=lambda: pdf_bytes("Pagos con Stripe")
    )
    session_id = create_session(client)

    body = request_turn(client, session_id, request, [spec])

    assert body["text"] == ESTIMATION
    assert body["prompt_version"] == "v2"
    assert body["usage"]["output_tokens"] == 500
    system, user = fake_llm[0]
    assert "sin tablas ni listas" in system["content"]  # el formato elegido llegó a la plantilla
    assert DESCRIPTION in user["content"]
    assert "--- attachment: spec.pdf ---\nPagos con Stripe" in user["content"]
    assert get_session(client, session_id)["metadata"]["project_name"] == "App del gimnasio"


def test_client_reports_a_lost_session():
    with pytest.raises(SessionLost):
        get_session(TestClient(app), "no-existe")


def test_client_translates_api_validation_errors():
    response = TestClient(app).post("/api/v1/estimate", json={"description": "hola"})
    message = error_message(response)

    assert "demasiado corta: escribe al menos 20 caracteres" in message
    assert "Field required" in message  # los errores sin traducción se muestran tal cual


def test_local_validation_uses_the_same_messages():
    with pytest.raises(ValidationError) as exc:
        EstimationRequest(
            description="x" * 80_001,
            project_type="web_saas",
            detail_level="medium",
            output_format="narrative",
        )
    assert validation_message(exc.value.errors()) == (
        "La descripción es demasiado larga: el máximo son 80000 caracteres."
    )


def test_client_reports_unreachable_api():
    request = EstimationRequest(
        description=DESCRIPTION,
        project_type="web_saas",
        detail_level="medium",
        output_format="narrative",
    )
    with pytest.raises(ApiError, match="No se pudo conectar con la API"):
        request_turn(unreachable_client(), "sesion", request, [])
    with pytest.raises(ApiError, match="No se pudo conectar con la API"):
        create_session(unreachable_client())


def test_description_joins_text_and_attached_file():
    file = SimpleNamespace(getvalue=lambda: "  Transcripción adjunta  \n".encode())
    assert read_description(" Texto escrito ", file) == "Texto escrito\n\nTranscripción adjunta"
    assert read_description("Solo texto", None) == "Solo texto"


# --- Aplicación Streamlit (AppTest ejecuta el script como "streamlit run") ---


@pytest.fixture
def run_app(monkeypatch):
    """Ejecuta la app con un cliente HTTP dado en lugar de la API real."""

    def run(client: httpx.Client) -> AppTest:
        st.cache_resource.clear()
        st.cache_data.clear()
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
        return AppTest.from_file(APP_FILE, default_timeout=10).run()

    return run


def submit(at: AppTest) -> AppTest:
    return next(b for b in at.button if b.label == "Generar estimación").click().run()


def test_app_shows_typed_form_and_cag_context(run_app):
    at = run_app(TestClient(app))

    assert not at.exception
    assert len(at.chat_input) == 0  # ya no es un chat
    assert at.selectbox(key="project_type").value == ProjectType.WEB_SAAS
    assert at.selectbox(key="output_format").value == OutputFormat.PHASES_TABLE
    assert at.pills(key="detail_level").value == DetailLevel.MEDIUM
    # El sidebar muestra el system prompt v2 que recibiría el primer turno de la conversación
    expected = render_system_prompt(
        ProjectType.WEB_SAAS, DetailLevel.MEDIUM, OutputFormat.PHASES_TABLE, "v2"
    )
    assert [block.value for block in at.sidebar.code] == [expected]
    assert len(at.sidebar.expander) == 5  # system prompt + tarifas + 3 referencias
    # Al cargar la página se abre una sesión vacía
    assert at.session_state.session_id
    assert at.sidebar.caption[0].value.startswith("Historial: 0 de 6 turnos")


def test_app_submits_the_form_and_shows_the_estimation(run_app, fake_llm):
    at = run_app(TestClient(app))
    at.text_area(key="description").input(DESCRIPTION)
    at.selectbox(key="project_type").select(ProjectType.MOBILE_APP)
    at.selectbox(key="output_format").select(OutputFormat.NARRATIVE)
    at.pills(key="detail_level").set_value(DetailLevel.DETAILED)
    submit(at)

    assert not at.exception
    assert len(fake_llm) == 1
    system, user = fake_llm[0]
    assert DESCRIPTION in user["content"]
    assert "como app móvil" in system["content"]
    assert at.main.text[0].value == DESCRIPTION  # el mensaje enviado queda en la conversación
    assert at.main.markdown[0].value == ESTIMATION
    assert at.text_area(key="description").value == ""  # listo para el siguiente turno
    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Tokens entrada"] == "2.000"
    assert metrics["Prompt"] == "v2"
    # El contexto del sidebar pasa a ser el del próximo turno: mismos parámetros, con la ficha
    next_system = at.sidebar.code[0].value
    assert "como app móvil" in next_system
    assert "- Nombre del proyecto: App del gimnasio" in next_system


def test_app_validates_before_calling_the_api(run_app, fake_llm):
    at = run_app(TestClient(app))
    at.text_area(key="description").input("muy corta")
    submit(at)

    assert not at.exception
    assert "demasiado corta" in at.error[0].value
    assert fake_llm == []


def test_app_reports_unreachable_api(run_app):
    at = run_app(unreachable_client())
    at.text_area(key="description").input(DESCRIPTION)
    submit(at)

    assert not at.exception
    assert "No se pudo conectar con la API" in at.sidebar.warning[0].value
    assert "No se pudo conectar con la API" in at.error[0].value


def test_app_fills_the_form_with_the_sample_transcription(run_app, fake_llm):
    at = run_app(TestClient(app))
    next(b for b in at.button if "ejemplo" in b.label).click().run()
    submit(at)

    assert not at.exception
    sample = streamlit_app.SAMPLE_TRANSCRIPTION.read_text(encoding="utf-8").strip()
    assert sample in fake_llm[0][1]["content"]


def test_app_remembers_the_project_across_three_turns(run_app, fake_llm):
    at = run_app(TestClient(app))
    for message in [
        DESCRIPTION,
        "Añadimos un programa de puntos para los socios más fieles.",
        "El equipo será de tres personas y la app se publica en las dos stores.",
    ]:
        at.text_area(key="description").input(message)
        submit(at)

    assert not at.exception
    assert [len(messages) for messages in fake_llm] == [2, 4, 6]  # system + historial + nuevo
    assert len(at.main.chat_message) == 6  # tres pares mensaje / estimación
    memory = at.sidebar.caption[0].value
    assert memory.startswith("Historial: 3 de 6 turnos")
    facts = at.sidebar.markdown[0].value
    assert "**Proyecto:** App del gimnasio" in facts
    assert "`Tecnología 1` `Tecnología 2` `Tecnología 3`" in facts
    assert at.sidebar.markdown[1].value == "**Alcance acordado:** Alcance tras el turno 3"


def test_new_conversation_starts_a_new_session(run_app, fake_llm):
    at = run_app(TestClient(app))
    first_session = at.session_state.session_id
    at.text_area(key="description").input(DESCRIPTION)
    submit(at)

    next(b for b in at.button if b.label == "Nueva conversación").click().run()

    assert not at.exception
    assert at.session_state.session_id != first_session
    assert at.session_state.turns == []
    assert len(at.main.chat_message) == 0
    assert at.sidebar.caption[0].value.startswith("Historial: 0 de 6 turnos")


def test_app_recovers_when_the_server_lost_the_session(run_app, fake_llm):
    at = run_app(TestClient(app))
    lost_session = at.session_state.session_id
    get_session_store()._sessions.clear()  # como si uvicorn --reload hubiera reiniciado la API

    at.run()

    assert not at.exception
    assert "se perdió" in at.info[0].value
    assert at.session_state.session_id != lost_session
