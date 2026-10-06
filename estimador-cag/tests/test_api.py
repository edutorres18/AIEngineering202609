"""Endpoints HTTP con el proveedor LLM simulado (sin llamadas reales)."""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm_service
from app.services.llm_service import LLMResult

DESCRIPTION = (
    "El equipo de marketing necesita una landing page con formulario de contacto e "
    "integración con HubSpot."
)
REQUEST = {
    "description": DESCRIPTION,
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

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
    response = client.post("/api/v1/estimate", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["text"].startswith("## Estimación")
    assert body["prompt_version"] == "v1"
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["usage"] == {
        "input_tokens": 2000,
        "output_tokens": 500,
        "cached_input_tokens": 1024,
    }
    assert body["estimated_cost_usd"] == pytest.approx(0.0006)
    assert body["truncated"] is False
    # Dos mensajes separados: el contexto CAG en el system y la descripción en el de usuario
    system, user = fake_llm[0]
    assert (system["role"], user["role"]) == ("system", "user")
    assert '<example number="1">' in system["content"]
    assert "Confianza (%)" in system["content"]  # instrucciones del formato phases_table
    assert f"<project_description>\n{DESCRIPTION}\n</project_description>" in user["content"]


def test_estimate_renders_the_prompt_for_the_requested_parameters(fake_llm):
    narrative = REQUEST | {"output_format": "narrative", "detail_level": "detailed"}
    assert client.post("/api/v1/estimate", json=narrative).status_code == 200

    system = fake_llm[0][0]["content"]
    assert "sin tablas ni listas" in system
    assert "supuestos por fase" in system
    assert "Confianza (%)" not in system


@pytest.mark.parametrize(
    "invalid",
    [
        {"description": "muy corta"},
        {"description": "   " + "x" * 10 + "   "},  # los espacios no cuentan
        {"description": "x" * 80_001},
        {"project_type": "videojuego"},
        {"detail_level": "MEDIUM"},  # el valor del enum, no el nombre
        {"output_format": None},
    ],
)
def test_estimate_rejects_invalid_requests(fake_llm, invalid):
    response = client.post("/api/v1/estimate", json=REQUEST | invalid)
    assert response.status_code == 422
    assert fake_llm == []  # no se gasta una llamada al LLM


def test_estimate_requires_every_parameter(fake_llm):
    response = client.post("/api/v1/estimate", json={"description": DESCRIPTION})
    assert response.status_code == 422
    missing = {error["loc"][-1] for error in response.json()["detail"]}
    assert missing == {"project_type", "detail_level", "output_format"}


def test_estimate_maps_provider_rate_limit_to_503(monkeypatch):
    async def rate_limited(settings, messages):
        raise openai.RateLimitError(
            "rate limit",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")),
            body=None,
        )

    monkeypatch.setattr(llm_service, "_call_openai", rate_limited)
    response = client.post("/api/v1/estimate", json=REQUEST)
    assert response.status_code == 503
    assert "Límite de peticiones" in response.json()["detail"]
