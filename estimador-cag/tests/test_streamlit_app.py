"""Interfaz Streamlit: cliente SSE contra la API real (LLM simulado) y la app con AppTest."""

from pathlib import Path

import httpx
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import streamlit_app
from app.main import app
from app.services import llm_service
from app.services.llm_service import LLMResult
from streamlit_app import ApiError, iter_estimation, parse_sse, start_estimation

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
TRANSCRIPTION = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing "
    "page con formulario de contacto e integración con HubSpot."
)
ESTIMATION = "## Estimación: Landing page\n\n### Desglose de tareas\n\n| Tarea | Horas |"


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def stream(settings, messages):
        calls.append(messages)
        for i in range(0, len(ESTIMATION), 10):
            yield ESTIMATION[i : i + 10]
        yield LLMResult(
            text=ESTIMATION,
            finish_reason="completed",
            truncated=False,
            input_tokens=2000,
            output_tokens=500,
        )

    monkeypatch.setattr(llm_service, "_stream_openai", stream)
    return calls


def unreachable_client() -> httpx.Client:
    def refuse(request):
        raise httpx.ConnectError("Connection refused", request=request)

    return httpx.Client(base_url="http://localhost:8000", transport=httpx.MockTransport(refuse))


# --- Cliente de la API ---


def test_parse_sse_groups_lines_into_events():
    lines = [
        ": ping",
        "event: delta",
        'data: {"text": "Hola"}',
        "",
        "event: done",
        "data: {",
        'data: "ok": true}',
        "",
    ]
    assert list(parse_sse(lines)) == [("delta", {"text": "Hola"}), ("done", {"ok": True})]


def test_client_streams_estimation_from_the_api(fake_llm):
    metrics = {}
    response = start_estimation(TestClient(app), TRANSCRIPTION)
    chunks = list(iter_estimation(response, metrics))

    assert len(chunks) > 1
    assert "".join(chunks) == ESTIMATION
    assert metrics["model"] == "gpt-4o-mini"
    assert metrics["usage"]["output_tokens"] == 500
    assert response.is_closed


def test_client_translates_validation_errors(fake_llm):
    with pytest.raises(ApiError, match="demasiado corta: escribe al menos 50 caracteres"):
        start_estimation(TestClient(app), "hola")
    assert fake_llm == []


def test_client_raises_on_error_event():
    response = httpx.Response(
        200, text='event: error\ndata: {"detail": "Error al llamar al proveedor LLM"}\n\n'
    )
    with pytest.raises(ApiError, match="Error al llamar al proveedor LLM"):
        list(iter_estimation(response, {}))


def test_client_detects_a_stream_cut_without_done_event():
    response = httpx.Response(200, text='event: delta\ndata: {"text": "## Estim"}\n\n')
    with pytest.raises(ApiError, match="sin completarse"):
        list(iter_estimation(response, {}))


def test_client_reports_unreachable_api():
    with pytest.raises(ApiError, match="No se pudo conectar con la API"):
        start_estimation(unreachable_client(), TRANSCRIPTION)


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


def test_app_shows_chat_and_cag_context(run_app):
    at = run_app(TestClient(app))

    assert not at.exception
    assert len(at.chat_input) == 1
    sidebar_code = [block.value for block in at.sidebar.code]
    assert llm_service.build_system_prompt() in sidebar_code  # system prompt de solo lectura
    assert len(sidebar_code) == 4  # system prompt + 3 estimaciones de referencia


def test_app_streams_estimation_and_keeps_history(run_app, fake_llm):
    at = run_app(TestClient(app))
    at.chat_input[0].set_value(TRANSCRIPTION).run()
    at.chat_input[0].set_value(TRANSCRIPTION + " También quieren un blog.").run()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert len(fake_llm) == 2
    answer = at.chat_message[3].markdown[0].value
    assert answer == ESTIMATION
    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Tokens entrada"] == "2.000"
    assert metrics["Tokens salida"] == "500"


def test_app_reports_unreachable_api(run_app):
    at = run_app(unreachable_client())
    at.chat_input[0].set_value(TRANSCRIPTION).run()

    assert not at.exception
    assert "No se pudo conectar con la API" in at.sidebar.warning[0].value
    assert "No se pudo conectar con la API" in at.chat_message[1].error[0].value


def test_app_offers_sample_transcription_on_empty_chat(run_app, fake_llm):
    at = run_app(TestClient(app))
    sample = next(b for b in at.button if "ejemplo" in b.label)
    sample.click().run()

    assert not at.exception
    sent = fake_llm[0][1]["content"]
    assert streamlit_app.SAMPLE_TRANSCRIPTION.read_text(encoding="utf-8").strip() in sent
