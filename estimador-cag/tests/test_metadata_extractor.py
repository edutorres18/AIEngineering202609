"""Salida estructurada de cada SDK y extractor de la ficha (sin llamadas reales)."""

import asyncio
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services import llm_service
from app.services.llm_service import LLMServiceError
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import ProjectMetadata

MESSAGES = [
    {"role": "system", "content": "Eres un extractor de hechos."},
    {"role": "user", "content": "Nimbus CRM con React."},
]
FACTS = ProjectMetadata(project_name="Nimbus CRM", mentioned_technologies=["React"])


def openai_parsed_response(parsed):
    usage = SimpleNamespace(
        input_tokens=900, output_tokens=60, input_tokens_details=SimpleNamespace(cached_tokens=0)
    )
    return SimpleNamespace(
        status="completed",
        incomplete_details=None,
        usage=usage,
        output_text=parsed.model_dump_json() if parsed else "",
        output_parsed=parsed,
    )


def fake_openai_parse(monkeypatch, parsed) -> list[dict]:
    requests = []

    async def parse(**request):
        requests.append(request)
        return openai_parsed_response(parsed)

    fake = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    monkeypatch.setattr(llm_service, "_openai_client", lambda: fake)
    return requests


def extract(settings: Settings, model: str = "modelo-barato"):
    return asyncio.run(
        llm_service.extract_structured(MESSAGES, ProjectMetadata, model=model, settings=settings)
    )


def test_openai_structured_output_uses_text_format(monkeypatch):
    requests = fake_openai_parse(monkeypatch, FACTS)

    assert extract(Settings()) == FACTS
    request = requests[0]
    assert request["text_format"] is ProjectMetadata
    assert request["model"] == "modelo-barato"
    assert request["max_output_tokens"] == 1000
    assert request["input"] == MESSAGES
    assert request["store"] is False


def test_anthropic_structured_output_uses_output_format(monkeypatch):
    requests = []

    async def parse(**request):
        requests.append(request)
        usage = SimpleNamespace(
            input_tokens=900,
            output_tokens=60,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        content = [SimpleNamespace(type="text", text=FACTS.model_dump_json())]
        return SimpleNamespace(
            parsed_output=FACTS, content=content, usage=usage, stop_reason="end_turn"
        )

    fake = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    monkeypatch.setattr(llm_service, "_anthropic_client", lambda: fake)
    settings = Settings(LLM_PROVIDER="anthropic", LLM_MODEL="claude-haiku-4-5")

    assert extract(settings, model="claude-haiku-4-5") == FACTS
    request = requests[0]
    assert request["output_format"] is ProjectMetadata
    assert request["system"] == MESSAGES[0]["content"]  # el system va aparte en Anthropic
    assert request["messages"] == [MESSAGES[1]]


def test_structured_output_without_a_parsed_object_is_an_error(monkeypatch):
    fake_openai_parse(monkeypatch, None)  # p. ej. el modelo se negó o se cortó el JSON

    with pytest.raises(LLMServiceError):
        extract(Settings())


def test_update_metadata_merges_the_extracted_facts(monkeypatch):
    requests = fake_openai_parse(monkeypatch, FACTS)
    previous = ProjectMetadata(mentioned_technologies=["PostgreSQL"], assumed_team_size=2)

    merged = asyncio.run(update_metadata(previous, "Nimbus CRM con React.", "## Estimación"))

    assert merged == ProjectMetadata(
        project_name="Nimbus CRM",
        assumed_team_size=2,
        mentioned_technologies=["PostgreSQL", "React"],
    )
    assert requests[0]["model"] == "gpt-4o-mini-extractor"  # METADATA_EXTRACTOR_MODEL


def test_update_metadata_keeps_the_previous_one_when_extraction_fails(monkeypatch):
    fake_openai_parse(monkeypatch, None)
    previous = ProjectMetadata(project_name="Nimbus CRM")

    assert asyncio.run(update_metadata(previous, "...", "...")) is previous


def test_extractor_model_defaults_to_the_estimation_model(monkeypatch):
    monkeypatch.delenv("METADATA_EXTRACTOR_MODEL")

    assert Settings(LLM_MODEL="gpt-4o-mini").METADATA_EXTRACTOR_MODEL == "gpt-4o-mini"
