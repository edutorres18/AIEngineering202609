"""Streaming SSE y contexto CAG, con los proveedores simulados (sin llamadas reales)."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.services import llm_service
from app.services.llm_service import LLMResult, build_system_prompt

TRANSCRIPTION = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing "
    "page con formulario de contacto e integración con HubSpot."
)
ESTIMATION = "## Estimación: Landing page\n### Desglose de tareas\n| Tarea | ... |"

client = TestClient(app)


def sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def fake_stream(*items):
    async def stream(settings, messages):
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item

    return stream


def api_error(message: str = "boom") -> openai.APIError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.APIError(message, request=request, body=None)


RESULT = LLMResult(
    text=ESTIMATION,
    finish_reason="completed",
    truncated=False,
    input_tokens=2000,
    output_tokens=500,
    cached_input_tokens=1024,
)


def test_stream_sends_deltas_then_done(monkeypatch):
    monkeypatch.setattr(
        llm_service, "_stream_openai", fake_stream(ESTIMATION[:20], ESTIMATION[20:], RESULT)
    )
    response = client.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert [name for name, _ in events] == ["delta", "delta", "done"]
    assert "".join(data["text"] for name, data in events if name == "delta") == ESTIMATION
    done = events[-1][1]
    assert done["provider"] == "openai"
    assert done["usage"] == {
        "input_tokens": 2000,
        "output_tokens": 500,
        "cached_input_tokens": 1024,
    }
    assert done["estimated_cost_usd"] == pytest.approx(0.0006)
    assert done["truncated"] is False


def test_stream_error_before_first_token_is_an_http_error(monkeypatch):
    rate_limit = openai.RateLimitError(
        "rate limit",
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")),
        body=None,
    )
    monkeypatch.setattr(llm_service, "_stream_openai", fake_stream(rate_limit))
    response = client.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPTION})

    assert response.status_code == 503
    assert "Límite de peticiones" in response.json()["detail"]


def test_stream_error_after_first_token_is_an_error_event(monkeypatch):
    monkeypatch.setattr(llm_service, "_stream_openai", fake_stream("## Estimación", api_error()))
    response = client.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPTION})

    assert response.status_code == 200  # el stream ya había empezado
    assert sse_events(response.text) == [
        ("delta", {"text": "## Estimación"}),
        ("error", {"detail": "Error al llamar al proveedor LLM", "status_code": 502}),
    ]


def test_stream_without_text_is_an_error(monkeypatch):
    empty = LLMResult(
        text="", finish_reason="completed", truncated=False, input_tokens=1, output_tokens=0
    )
    monkeypatch.setattr(llm_service, "_stream_openai", fake_stream(empty))
    response = client.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPTION})

    assert response.status_code == 502
    assert "vacía" in response.json()["detail"]


def test_stream_cut_by_the_provider_ends_with_an_error_event(monkeypatch):
    monkeypatch.setattr(llm_service, "_stream_openai", fake_stream("## Estimación"))
    response = client.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPTION})

    name, data = sse_events(response.text)[-1]
    assert name == "error"
    assert "antes de terminarla" in data["detail"]


def test_stream_rejects_too_short_transcription(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_service, "_stream_openai", lambda *args: calls.append(args))
    response = client.post("/api/v1/estimate/stream", json={"transcription": "muy corta"})

    assert response.status_code == 422
    assert calls == []  # no se gasta una llamada al LLM


def test_stream_endpoint_is_documented_as_sse():
    operation = client.get("/openapi.json").json()["paths"]["/api/v1/estimate/stream"]["post"]
    assert "text/event-stream" in operation["responses"]["200"]["content"]
    assert {"422", "502", "503", "504"} <= operation["responses"].keys()


def test_context_exposes_the_same_prompt_the_model_receives():
    response = client.get("/api/v1/context")

    assert response.status_code == 200
    body = response.json()
    assert body["system_prompt"] == build_system_prompt()
    assert (body["provider"], body["model"]) == ("openai", "gpt-4o-mini")
    assert body["hourly_rates_eur"]["QA"] == 40
    assert len(body["examples"]) == 3
    for example in body["examples"]:
        assert example["content"] in body["system_prompt"]


# --- Normalización de los eventos de cada SDK ---


class FakeOpenAIStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def __aiter__(self):
        for event in self.events:
            yield event


def openai_response(status="completed", reason=None, error=None):
    usage = SimpleNamespace(
        input_tokens=2500,
        output_tokens=600,
        input_tokens_details=SimpleNamespace(cached_tokens=2048),
    )
    details = SimpleNamespace(reason=reason) if reason else None
    return SimpleNamespace(
        status=status,
        incomplete_details=details,
        usage=usage,
        output_text="Hola mundo",
        error=error,
    )


def fake_openai_client(monkeypatch, events) -> tuple[FakeOpenAIStream, list[dict]]:
    stream = FakeOpenAIStream(events)
    requests = []

    async def create(**request):
        requests.append(request)
        return stream

    fake = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(llm_service, "_openai_client", lambda: fake)
    return stream, requests


async def collect(settings):
    return [item async for item in llm_service.stream_estimation(TRANSCRIPTION, settings)]


def test_openai_stream_yields_text_deltas_and_usage(monkeypatch):
    stream, requests = fake_openai_client(
        monkeypatch,
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Hola"),
            SimpleNamespace(type="response.output_text.delta", delta=" mundo"),
            SimpleNamespace(type="response.completed", response=openai_response()),
        ],
    )
    *deltas, metadata = asyncio.run(collect(Settings()))

    assert deltas == ["Hola", " mundo"]
    assert metadata.usage.model_dump() == {
        "input_tokens": 2500,
        "output_tokens": 600,
        "cached_input_tokens": 2048,
    }
    assert metadata.finish_reason == "completed"
    assert requests[0]["stream"] is True
    assert requests[0]["store"] is False
    assert stream.closed


def test_openai_incomplete_stream_is_marked_as_truncated(monkeypatch):
    fake_openai_client(
        monkeypatch,
        [
            SimpleNamespace(type="response.output_text.delta", delta="Hola mundo"),
            SimpleNamespace(
                type="response.incomplete",
                response=openai_response("incomplete", "max_output_tokens"),
            ),
        ],
    )
    *_, metadata = asyncio.run(collect(Settings()))

    assert metadata.truncated is True
    assert metadata.finish_reason == "max_output_tokens"


def test_openai_failed_stream_raises_service_error(monkeypatch):
    fake_openai_client(
        monkeypatch,
        [
            SimpleNamespace(type="response.output_text.delta", delta="Hola"),
            SimpleNamespace(
                type="response.failed",
                response=openai_response("failed", error=SimpleNamespace(code="server_error")),
            ),
        ],
    )
    with pytest.raises(llm_service.LLMServiceError):
        asyncio.run(collect(Settings()))


def test_anthropic_stream_yields_text_deltas_and_usage(monkeypatch):
    requests = []

    class FakeMessageStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        @property
        async def text_stream(self):
            for text in ["Hola", " mundo"]:
                yield text

        async def get_final_message(self):
            usage = SimpleNamespace(
                input_tokens=100,
                output_tokens=600,
                cache_read_input_tokens=2000,
                cache_creation_input_tokens=0,
            )
            content = [SimpleNamespace(type="text", text="Hola mundo")]
            return SimpleNamespace(content=content, usage=usage, stop_reason="end_turn")

    def stream(**request):
        requests.append(request)
        return FakeMessageStream()

    fake = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    monkeypatch.setattr(llm_service, "_anthropic_client", lambda: fake)
    settings = Settings(LLM_PROVIDER="anthropic", LLM_MODEL="claude-haiku-4-5")
    *deltas, metadata = asyncio.run(collect(settings))

    assert deltas == ["Hola", " mundo"]
    assert metadata.provider == "anthropic"
    assert metadata.usage.input_tokens == 2100  # entrada sin cachear + leída de caché
    assert metadata.usage.cached_input_tokens == 2000
    # El system prompt va como parámetro aparte y la transcripción como único mensaje
    assert requests[0]["system"] == build_system_prompt()
    assert [m["role"] for m in requests[0]["messages"]] == ["user"]
