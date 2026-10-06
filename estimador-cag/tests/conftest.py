import os

# Configuración de test fija: no depende del .env local ni llama a ningún LLM real.
# Las variables de entorno tienen prioridad sobre el .env en pydantic-settings.
os.environ.update(
    {
        "OPENAI_API_KEY": "sk-test-dummy",
        "ANTHROPIC_API_KEY": "sk-ant-test-dummy",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o-mini",
        "APP_ENV": "test",
        "LOG_LEVEL": "WARNING",
    }
)
