"""Endpoints HTTP con el proveedor LLM simulado (sin llamadas reales)."""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm_service
from app.services.llm_service import LLMResult

TRANSCRIPTION = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing "
    "page con formulario de contacto e integración con HubSpot."
)

client = TestClient(app)


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def fake_call(settings, messages):
        calls.append(messages)
        return LLMResult(
            text="## Estimación: Landing page\n### Desglose de tareas\n...",
            finish_reason="completed",
            truncated=False,
            input_tokens=2000,
            output_tokens=500,
            cached_input_tokens=1024,
        )

    monkeypatch.setattr(llm_service, "_call_openai", fake_call)
    return calls


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_swagger_docs_available():
    assert client.get("/docs").status_code == 200
    assert "/api/v1/estimate" in client.get("/openapi.json").json()["paths"]


def test_estimate_returns_estimation(fake_llm):
    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPTION})
    assert response.status_code == 200
    body = response.json()
    assert body["estimation"].startswith("## Estimación")
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["usage"] == {
        "input_tokens": 2000,
        "output_tokens": 500,
        "cached_input_tokens": 1024,
    }
    assert body["estimated_cost_usd"] == pytest.approx(0.0006)
    assert body["truncated"] is False
    # El contexto CAG viaja en el system prompt y la transcripción en el mensaje de usuario
    system, user = fake_llm[0]
    assert "ESTIMACIÓN DE REFERENCIA 1" in system["content"]
    assert TRANSCRIPTION in user["content"]


def test_estimate_rejects_too_short_transcription(fake_llm):
    response = client.post("/api/v1/estimate", json={"transcription": "muy corta"})
    assert response.status_code == 422
    assert fake_llm == []  # no se gasta una llamada al LLM


def test_estimate_maps_provider_rate_limit_to_503(monkeypatch):
    async def rate_limited(settings, messages):
        raise openai.RateLimitError(
            "rate limit",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")),
            body=None,
        )

    monkeypatch.setattr(llm_service, "_call_openai", rate_limited)
    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPTION})
    assert response.status_code == 503
    assert "Límite de peticiones" in response.json()["detail"]
