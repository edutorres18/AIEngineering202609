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
from streamlit_app import (
    ApiError,
    error_message,
    read_description,
    request_estimation,
    validation_message,
)

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
DESCRIPTION = "App móvil para un gimnasio: reserva de clases, pago de cuotas y avisos push."
ESTIMATION = "## Estimación: App del gimnasio\n\n### Desglose por fases\n\n| Fase | Horas |"


@pytest.fixture
def fake_llm(monkeypatch):
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

    monkeypatch.setattr(llm_service, "_call_openai", fake_call)
    return calls


def unreachable_client() -> httpx.Client:
    def refuse(request):
        raise httpx.ConnectError("Connection refused", request=request)

    return httpx.Client(base_url="http://localhost:8000", transport=httpx.MockTransport(refuse))


# --- Cliente de la API ---


def test_client_sends_the_typed_request_and_gets_the_estimation(fake_llm):
    request = EstimationRequest(
        description=DESCRIPTION,
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.NARRATIVE,
    )
    body = request_estimation(TestClient(app), request)

    assert body["text"] == ESTIMATION
    assert body["prompt_version"] == "v1"
    assert body["usage"]["output_tokens"] == 500
    system = fake_llm[0][0]["content"]
    assert "sin tablas ni listas" in system  # el formato elegido llegó a la plantilla


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
        request_estimation(unreachable_client(), request)


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
    # El sidebar muestra el system prompt que recibiría el modelo con esos parámetros
    expected = render_system_prompt(
        ProjectType.WEB_SAAS, DetailLevel.MEDIUM, OutputFormat.PHASES_TABLE
    )
    assert [block.value for block in at.sidebar.code] == [expected]
    assert len(at.sidebar.expander) == 5  # system prompt + tarifas + 3 referencias


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
    assert at.main.markdown[0].value == ESTIMATION
    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Tokens entrada"] == "2.000"
    assert metrics["Prompt"] == "v1"
    # El contexto del sidebar pasa a ser el de los parámetros enviados
    assert at.sidebar.code[0].value == system["content"]


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
