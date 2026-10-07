import os

# Configuración de test fija: no depende del .env local ni llama a ningún LLM real.
# Las variables de entorno tienen prioridad sobre el .env en pydantic-settings.
os.environ.update(
    {
        "OPENAI_API_KEY": "sk-test-dummy",
        "ANTHROPIC_API_KEY": "sk-ant-test-dummy",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o-mini",
        "METADATA_EXTRACTOR_MODEL": "gpt-4o-mini-extractor",
        "APP_ENV": "test",
        "LOG_LEVEL": "WARNING",
    }
)

import pytest  # noqa: E402 (las variables de entorno van antes de importar la app)

from app.main import app  # noqa: E402
from app.services import llm_service  # noqa: E402
from app.services.llm_service import LLMResult  # noqa: E402
from app.sessions.models import ProjectMetadata  # noqa: E402
from app.sessions.store import SessionStore, get_session_store  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    """Red de seguridad: si un test no simula el proveedor, falla en vez de llamar a la API."""

    def refuse():
        raise AssertionError("Un test ha intentado llamar a un proveedor LLM real")

    monkeypatch.setattr(llm_service, "_openai_client", refuse)
    monkeypatch.setattr(llm_service, "_anthropic_client", refuse)


class FakeLLM:
    """Proveedor simulado para la conversación: guarda los mensajes de cada llamada y responde
    con estimaciones numeradas y con las fichas que el test ponga en `facts`."""

    def __init__(self) -> None:
        self.estimations: list[list[dict[str, str]]] = []  # mensajes de cada estimación
        self.extractions: list[list[dict[str, str]]] = []  # mensajes de cada extracción
        self.extraction_models: list[str] = []
        self.facts: list[ProjectMetadata | Exception] = []  # respuestas del extractor, en orden

    async def call(self, settings, messages):
        self.estimations.append(messages)
        return LLMResult(
            text=f"## Estimación: turno {len(self.estimations)}\n| Fase | Horas |",
            finish_reason="completed",
            truncated=False,
            input_tokens=3000,
            output_tokens=800,
        )

    async def parse(self, settings, messages, response_model):
        self.extractions.append(messages)
        self.extraction_models.append(settings.LLM_MODEL)
        facts = self.facts.pop(0) if self.facts else ProjectMetadata()
        if isinstance(facts, Exception):
            raise facts
        result = LLMResult(
            text=facts.model_dump_json(),
            finish_reason="completed",
            truncated=False,
            input_tokens=900,
            output_tokens=60,
        )
        return facts, result


@pytest.fixture
def fake_provider(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(llm_service, "_call_openai", fake.call)
    monkeypatch.setattr(llm_service, "_parse_openai", fake.parse)
    return fake


@pytest.fixture
def session_store():
    """Almacén propio del test, con una ventana de 3 turnos, en lugar del global del proceso."""
    store = SessionStore(max_turns=3)
    app.dependency_overrides[get_session_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_session_store, None)
