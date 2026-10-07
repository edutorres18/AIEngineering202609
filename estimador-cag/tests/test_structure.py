"""Valida que el scaffolding respeta la estructura pedida en el ejercicio."""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/routers/__init__.py",
    "app/routers/estimations.py",
    "app/services/__init__.py",
    "app/services/llm_service.py",
    "app/context/__init__.py",
    "app/context/examples.py",
    "app/schemas/__init__.py",
    "app/schemas/estimation.py",
    "app/prompts/__init__.py",
    "app/prompts/loader.py",
    "app/prompts/estimation/v1/system.j2",
    "app/prompts/estimation/v1/user.j2",
    "app/prompts/estimation/v1/examples.j2",
    "tests/prompts/test_estimation_v1.py",
    # Sesión 5: conversación con memoria y adjuntos
    "app/routers/sessions.py",
    "app/schemas/session.py",
    "app/services/conversation_service.py",
    "app/sessions/__init__.py",
    "app/sessions/models.py",
    "app/sessions/store.py",
    "app/sessions/metadata_extractor.py",
    "app/attachments/__init__.py",
    "app/attachments/extractor.py",
    "app/prompts/estimation/v2/system.j2",
    "app/prompts/estimation/v2/user.j2",
    "app/prompts/metadata_extraction/v1/system.j2",
    "app/prompts/metadata_extraction/v1/user.j2",
    "streamlit_app.py",
    ".env.example",
    ".gitignore",
    "pyproject.toml",
    "README.md",
]

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "APP_ENV",
    "LOG_LEVEL",
    "MAX_CONVERSATION_TURNS",
    "MAX_ATTACHMENT_CHARS",
    "METADATA_EXTRACTOR_MODEL",
]


@pytest.mark.parametrize("relative_path", REQUIRED_PATHS)
def test_required_path_exists(relative_path):
    assert (PROJECT_ROOT / relative_path).is_file(), f"Falta {relative_path}"


def test_env_is_gitignored():
    patterns = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in patterns


def test_env_example_documents_variables_without_secrets():
    content = (PROJECT_ROOT / ".env.example").read_text()
    for var in REQUIRED_ENV_VARS:
        assert re.search(rf"^{var}=", content, re.MULTILINE), f"{var} no está en .env.example"
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert re.search(rf"^{key}=$", content, re.MULTILINE), f"{key} no debe tener valor"


def test_no_api_keys_hardcoded_in_code():
    key_pattern = re.compile(r"sk-(ant-|proj-)?[A-Za-z0-9_-]{20,}")
    for path in [*(PROJECT_ROOT / "app").rglob("*.py"), PROJECT_ROOT / "streamlit_app.py"]:
        assert not key_pattern.search(path.read_text()), f"Posible API key en {path}"


def test_at_least_one_meeting_transcription():
    assert any((PROJECT_ROOT / "transcripciones").glob("*.md"))
