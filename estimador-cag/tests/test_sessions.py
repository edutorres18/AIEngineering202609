"""Conversación de varios turnos por HTTP (POST /sessions…), con el proveedor simulado.

Los tres primeros son los tests de integración del Paso 7 del ejercicio: la ficha se actualiza
entre turnos, un adjunto llega al modelo y la ventana deslizante limita el historial.
"""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm_service
from app.sessions.models import ProjectMetadata
from tests.documents import docx_bytes, pdf_bytes

client = TestClient(app)

FORM = {
    "transcript": "Reunión con el cliente: quieren un CRM para el equipo comercial.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def new_session() -> str:
    response = client.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def send(session_id: str, transcript: str = FORM["transcript"], **kwargs):
    return client.post(
        f"/sessions/{session_id}/estimate", data={**FORM, "transcript": transcript}, **kwargs
    )


def system_of(messages: list[dict[str, str]]) -> str:
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


# --- Paso 7 del ejercicio ---


def test_two_turns_update_the_project_metadata(session_store, fake_provider):
    fake_provider.facts = [
        ProjectMetadata(
            project_name="Nimbus CRM",
            mentioned_technologies=["React"],
            agreed_scope="CRM con contactos y oportunidades",
        ),
        ProjectMetadata(
            assumed_team_size=3,
            mentioned_technologies=["PostgreSQL", "react"],
            agreed_scope="CRM con contactos, oportunidades y facturación",
        ),
    ]
    session_id = new_session()

    first = send(session_id, "Nimbus CRM: contactos y oportunidades, frontend en React.")
    after_first = client.get(f"/sessions/{session_id}").json()
    second = send(session_id, "Añadimos facturación; base de datos PostgreSQL, equipo de 3.")
    after_second = client.get(f"/sessions/{session_id}").json()

    assert first.status_code == second.status_code == 200
    assert second.json()["prompt_version"] == "v2"
    assert after_first["metadata"]["project_name"] == "Nimbus CRM"
    assert after_first["metadata"]["assumed_team_size"] is None
    assert after_second["metadata"] == {
        "project_name": "Nimbus CRM",  # el segundo turno no lo menciona: se conserva
        "assumed_team_size": 3,
        "mentioned_technologies": ["React", "PostgreSQL"],
        "agreed_scope": "CRM con contactos, oportunidades y facturación",
    }
    assert after_second["turns"] == 2
    # El primer turno arranca sin ficha; el segundo recibe la ficha y el turno anterior
    first_call, second_call = fake_provider.estimations
    assert "es el primer turno" in system_of(first_call)
    assert "- Nombre del proyecto: Nimbus CRM" in system_of(second_call)
    assert "- Tecnologías mencionadas: React" in system_of(second_call)
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert "Nimbus CRM: contactos" in second_call[1]["content"]
    assert second_call[2]["content"] == first.json()["text"]


def test_a_pdf_attachment_reaches_the_estimation(session_store, fake_provider):
    session_id = new_session()
    spec = pdf_bytes("Modulo de facturacion con Stripe", "Exportacion de facturas a PDF")

    without = send(session_id, "Primera versión del CRM, sin documentación adjunta.")
    with_pdf = send(
        session_id,
        "Te paso la especificación técnica del módulo de facturación.",
        files=[("attachments", ("spec.pdf", spec, "application/pdf"))],
    )

    assert without.status_code == with_pdf.status_code == 200
    plain_message = fake_provider.estimations[0][-1]["content"]
    enriched_message = fake_provider.estimations[1][-1]["content"]
    assert "attachment" not in plain_message
    # Lo que cambia en la petición al modelo es exactamente el texto del documento
    assert (
        "--- attachment: spec.pdf ---\n"
        "Modulo de facturacion con Stripe\nExportacion de facturas a PDF\n"
        "--- end attachment ---"
    ) in enriched_message
    # El extractor de la ficha también lo ve, así que puede recoger «Stripe»
    assert "Modulo de facturacion con Stripe" in fake_provider.extractions[1][-1]["content"]


def test_eight_turns_never_exceed_the_sliding_window(session_store, fake_provider):
    fake_provider.facts = [ProjectMetadata(project_name="Nimbus CRM")]  # solo en el turno 1
    session_id = new_session()

    for turn in range(1, 9):
        assert send(session_id, f"Turno {turn}: seguimos precisando el alcance.").status_code == 200

    max_turns = session_store.get(session_id).history.max_turns  # 3 en los tests
    sizes = [len(messages) for messages in fake_provider.estimations]
    # system + hasta 3 pares (user, assistant) del historial + el mensaje nuevo
    assert sizes == [2, 4, 6, 8, 8, 8, 8, 8]
    assert max(sizes) == 1 + 2 * max_turns + 1
    last_call = fake_provider.estimations[-1]
    assert "Turno 5" in last_call[1]["content"]  # el par más antiguo que sigue en la ventana
    assert not any("Turno 1:" in m["content"] for m in last_call)
    # El turno 1 ya salió del historial, pero el nombre del proyecto sigue en la ficha
    assert "- Nombre del proyecto: Nimbus CRM" in system_of(last_call)
    assert client.get(f"/sessions/{session_id}").json()["turns"] == max_turns


# --- Otros comportamientos ---


def test_create_session_returns_a_new_uuid_each_time(session_store):
    first, second = new_session(), new_session()

    assert first != second
    assert len(first) == 36
    state = client.get(f"/sessions/{first}").json()
    assert state == {
        "session_id": first,
        "metadata": {
            "project_name": None,
            "assumed_team_size": None,
            "mentioned_technologies": [],
            "agreed_scope": None,
        },
        "turns": 0,
        "max_turns": 3,
    }


def test_word_attachments_are_accepted_too(session_store, fake_provider):
    session_id = new_session()
    proposal = docx_bytes(["Propuesta previa: app para comerciales en Flutter"])
    spec = pdf_bytes("Integracion con HubSpot")

    response = send(
        session_id,
        files=[
            ("attachments", ("propuesta.docx", proposal, "application/octet-stream")),
            ("attachments", ("spec.pdf", spec, "application/pdf")),
        ],
    )

    assert response.status_code == 200
    message = fake_provider.estimations[0][-1]["content"]
    assert message.index("propuesta.docx") < message.index("spec.pdf")
    assert "app para comerciales en Flutter" in message
    assert "Integracion con HubSpot" in message


def test_unknown_session_is_404(session_store, fake_provider):
    assert client.get("/sessions/no-existe").status_code == 404
    response = send("no-existe")

    assert response.status_code == 404
    assert "crea una nueva" in response.json()["detail"]
    assert fake_provider.estimations == []


@pytest.mark.parametrize(
    ("filename", "content", "status"),
    [("diagrama.png", b"\x89PNG", 415), ("spec.pdf", b"no es un pdf", 422)],
)
def test_bad_attachments_are_rejected_before_calling_the_llm(
    session_store, fake_provider, filename, content, status
):
    session_id = new_session()

    response = send(session_id, files=[("attachments", (filename, content, "application/pdf"))])

    assert response.status_code == status
    assert filename in response.json()["detail"]
    assert fake_provider.estimations == []


@pytest.mark.parametrize(
    "form",
    [
        {**FORM, "transcript": "   corta   "},
        {**FORM, "transcript": " " * 30},
        {**FORM, "transcript": "x" * 80_001},
        {**FORM, "project_type": "videojuego"},
        {key: value for key, value in FORM.items() if key != "detail_level"},
    ],
)
def test_invalid_forms_are_422(session_store, fake_provider, form):
    session_id = new_session()

    assert client.post(f"/sessions/{session_id}/estimate", data=form).status_code == 422
    assert fake_provider.estimations == []


def test_a_failed_estimation_leaves_the_session_untouched(session_store, monkeypatch):
    async def rate_limited(settings, messages):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(429, request=request)
        raise openai.RateLimitError("rate limit", response=response, body=None)

    monkeypatch.setattr(llm_service, "_call_openai", rate_limited)
    session_id = new_session()

    response = send(session_id)

    assert response.status_code == 503
    assert session_store.get(session_id).history.turns == 0  # se puede reintentar el turno


def test_a_failed_extraction_keeps_the_previous_metadata(session_store, fake_provider):
    fake_provider.facts = [
        ProjectMetadata(project_name="Nimbus CRM"),
        llm_service.LLMServiceError("timeout", status_code=504),
    ]
    session_id = new_session()

    send(session_id)
    response = send(session_id, "Segundo turno, con el extractor caído.")

    assert response.status_code == 200  # la estimación del turno no se pierde
    state = client.get(f"/sessions/{session_id}").json()
    assert state["metadata"]["project_name"] == "Nimbus CRM"
    assert state["turns"] == 2


def test_extractor_uses_its_own_model(session_store, fake_provider):
    send(new_session())

    assert fake_provider.extraction_models == ["gpt-4o-mini-extractor"]


def test_context_shows_the_system_prompt_of_the_next_turn(session_store, fake_provider):
    fake_provider.facts = [ProjectMetadata(project_name="Nimbus CRM")]
    session_id = new_session()
    send(session_id)

    context = client.get("/api/v1/context", params={"session_id": session_id}).json()
    without_session = client.get("/api/v1/context").json()

    assert context["prompt_version"] == "v2"
    assert "- Nombre del proyecto: Nimbus CRM" in context["system_prompt"]
    assert without_session["prompt_version"] == "v1"
    assert "project_metadata" not in without_session["system_prompt"]
    assert client.get("/api/v1/context", params={"session_id": "x"}).status_code == 404
