"""Configuración centralizada: variables de entorno validadas al arrancar (fail fast)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modelos económicos por defecto para cada proveedor.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "VAR=" vacía en .env equivale a no definirla
        extra="ignore",
    )

    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    LLM_PROVIDER: Literal["openai", "anthropic"] = "openai"
    LLM_MODEL: str = ""  # vacío → modelo por defecto del proveedor (gpt-4o-mini / claude-haiku-4-5)
    LLM_MAX_OUTPUT_TOKENS: int = 3000
    LLM_TEMPERATURE: float | None = 0.2  # solo OpenAI; None para modelos que no lo admiten
    LLM_TIMEOUT_SECONDS: float = 60.0
    APP_ENV: Literal["development", "test", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    @model_validator(mode="after")
    def _validate_provider(self) -> "Settings":
        key = self.OPENAI_API_KEY if self.LLM_PROVIDER == "openai" else self.ANTHROPIC_API_KEY
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"Falta {self.LLM_PROVIDER.upper()}_API_KEY: es obligatoria con "
                f"LLM_PROVIDER={self.LLM_PROVIDER}. Defínela en .env (ver .env.example)."
            )
        if not self.LLM_MODEL:
            self.LLM_MODEL = DEFAULT_MODELS[self.LLM_PROVIDER]
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
