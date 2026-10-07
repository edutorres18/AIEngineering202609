"""Lógica de negocio: compone el prompt, llama al LLM y normaliza la respuesta.

Estructura de mensajes (CAG, single-turn):
    [system]    → rol + tarifas + estimaciones de referencia + reglas + parámetros del formulario
    [user]      → descripción del proyecto (o transcripción de la reunión) a estimar
    [assistant] → estimación generada por el modelo

El prompt sale de las plantillas Jinja2 versionadas de app/prompts (ver loader.py).
La estimación se puede pedir completa (generate_estimation) o en streaming
(stream_estimation); ambas usan el mismo prompt, los mismos errores y las mismas métricas.

En una conversación (sesión 5, ver conversation_service.py) el array lleva además los turnos
anteriores entre el system y el mensaje nuevo; complete() acepta cualquier array de mensajes.
extract_structured() pide al modelo un objeto Pydantic (salida estructurada del proveedor).
"""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import TypeVar

import anthropic
import openai
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.context.examples import HOURLY_RATES_EUR, reference_estimations
from app.prompts.loader import (
    CONVERSATION_PROMPT_VERSION,
    PROMPT_VERSION,
    render_estimation_prompt,
    render_system_prompt,
)
from app.schemas.estimation import (
    CAGContext,
    DetailLevel,
    EstimationMetadata,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    TokenUsage,
)
from app.sessions.models import ProjectMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Precios aproximados en USD por millón de tokens (entrada, salida). Revisar periódicamente.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
}


class LLMServiceError(Exception):
    """Fallo al obtener una estimación del proveedor LLM."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class LLMResult:
    """Respuesta de un proveedor normalizada a un formato común."""

    text: str
    finish_reason: str
    truncated: bool
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


def get_cag_context(
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    settings: Settings | None = None,
    metadata: ProjectMetadata | None = None,
) -> CAGContext:
    """Contexto que recibe el modelo con estos parámetros, para mostrarlo en la interfaz.

    Con `metadata` (la ficha de una sesión), el system prompt es el v2 de la conversación.
    """
    settings = settings or get_settings()
    version = PROMPT_VERSION if metadata is None else CONVERSATION_PROMPT_VERSION
    return CAGContext(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        prompt_version=version,
        system_prompt=render_system_prompt(
            project_type, detail_level, output_format, version, metadata
        ),
        hourly_rates_eur=HOURLY_RATES_EUR,
        examples=list(reference_estimations()),
    )


def build_messages(request: EstimationRequest) -> list[dict[str, str]]:
    """System y user como mensajes separados, renderizados desde las plantillas."""
    system, user = render_estimation_prompt(request, PROMPT_VERSION)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@lru_cache
def _openai_client() -> openai.AsyncOpenAI:
    settings = get_settings()
    return openai.AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )


@lru_cache
def _anthropic_client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    return anthropic.AsyncAnthropic(
        api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )


def _openai_request(settings: Settings, messages: list[dict[str, str]]) -> dict:
    request = {
        "model": settings.LLM_MODEL,
        "input": messages,
        "max_output_tokens": settings.LLM_MAX_OUTPUT_TOKENS,
        "store": False,  # no guardar en OpenAI lo que cuentan los clientes
    }
    if settings.LLM_TEMPERATURE is not None:
        request["temperature"] = settings.LLM_TEMPERATURE
    return request


def _openai_result(response) -> LLMResult:
    reason = response.status or "unknown"
    if response.status == "incomplete" and response.incomplete_details:
        reason = response.incomplete_details.reason or reason
    usage = response.usage
    cached = usage.input_tokens_details.cached_tokens if usage and usage.input_tokens_details else 0
    return LLMResult(
        text=response.output_text,
        finish_reason=reason,
        truncated=reason == "max_output_tokens",
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        cached_input_tokens=cached or 0,
    )


async def _call_openai(settings: Settings, messages: list[dict[str, str]]) -> LLMResult:
    response = await _openai_client().responses.create(**_openai_request(settings, messages))
    return _openai_result(response)


async def _parse_openai(
    settings: Settings, messages: list[dict[str, str]], response_model: type[T]
) -> tuple[T | None, LLMResult]:
    """Salida estructurada: OpenAI restringe la respuesta al JSON Schema del modelo Pydantic."""
    response = await _openai_client().responses.parse(
        **_openai_request(settings, messages), text_format=response_model
    )
    return response.output_parsed, _openai_result(response)


async def _stream_openai(
    settings: Settings, messages: list[dict[str, str]]
) -> AsyncIterator[str | LLMResult]:
    """Produce los fragmentos de texto y, al final, el resultado completo con el uso de tokens."""
    stream = await _openai_client().responses.create(
        **_openai_request(settings, messages), stream=True
    )
    async with stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type in ("response.completed", "response.incomplete"):
                yield _openai_result(event.response)
            elif event.type in ("response.failed", "error"):
                # Solo el código: el evento completo incluye texto generado a partir del cliente.
                error = event.response.error if event.type == "response.failed" else event
                logger.error("OpenAI interrumpió el stream (code=%s)", getattr(error, "code", None))
                raise LLMServiceError("El proveedor LLM interrumpió la respuesta")


def _anthropic_request(settings: Settings, messages: list[dict[str, str]]) -> dict:
    # Anthropic recibe el system prompt como parámetro aparte, fuera del array de mensajes.
    return {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_OUTPUT_TOKENS,
        "system": next(m["content"] for m in messages if m["role"] == "system"),
        "messages": [m for m in messages if m["role"] != "system"],
    }


def _anthropic_result(message) -> LLMResult:
    usage = message.usage
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    reason = message.stop_reason or "unknown"
    return LLMResult(
        text="".join(block.text for block in message.content if block.type == "text"),
        finish_reason=reason,
        truncated=reason == "max_tokens",
        input_tokens=usage.input_tokens + cache_read + cache_write,
        output_tokens=usage.output_tokens,
        cached_input_tokens=cache_read,
    )


async def _call_anthropic(settings: Settings, messages: list[dict[str, str]]) -> LLMResult:
    response = await _anthropic_client().messages.create(**_anthropic_request(settings, messages))
    return _anthropic_result(response)


async def _parse_anthropic(
    settings: Settings, messages: list[dict[str, str]], response_model: type[T]
) -> tuple[T | None, LLMResult]:
    """Salida estructurada: Anthropic restringe la respuesta al JSON Schema del modelo Pydantic."""
    message = await _anthropic_client().messages.parse(
        **_anthropic_request(settings, messages), output_format=response_model
    )
    return message.parsed_output, _anthropic_result(message)


async def _stream_anthropic(
    settings: Settings, messages: list[dict[str, str]]
) -> AsyncIterator[str | LLMResult]:
    """Produce los fragmentos de texto y, al final, el resultado completo con el uso de tokens."""
    request = _anthropic_request(settings, messages)
    async with _anthropic_client().messages.stream(**request) as stream:
        async for text in stream.text_stream:
            yield text
        yield _anthropic_result(await stream.get_final_message())


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING_USD_PER_MTOK.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)


@contextmanager
def _provider_errors(provider: str) -> Iterator[None]:
    """Traduce las excepciones de los SDK a LLMServiceError sin exponer detalles internos."""
    try:
        yield
    except (openai.APITimeoutError, anthropic.APITimeoutError) as exc:
        logger.warning("Timeout del proveedor %s", provider)
        raise LLMServiceError("El proveedor LLM no respondió a tiempo", status_code=504) from exc
    except (openai.RateLimitError, anthropic.RateLimitError) as exc:
        logger.warning("Rate limit del proveedor %s", provider)
        raise LLMServiceError(
            "Límite de peticiones del proveedor LLM alcanzado, reintenta en unos segundos",
            status_code=503,
        ) from exc
    except (openai.AuthenticationError, anthropic.AuthenticationError) as exc:
        logger.error("El proveedor %s rechazó la API key", provider)
        raise LLMServiceError("Error de configuración del proveedor LLM") from exc
    except (openai.APIError, anthropic.APIError) as exc:
        logger.exception("Error del proveedor %s", provider)
        raise LLMServiceError("Error al llamar al proveedor LLM") from exc


def _log_request(settings: Settings, messages: list[dict[str, str]], *, streaming: bool) -> None:
    logger.debug(
        "Llamando a %s/%s (%d mensajes, system=%d chars, último user=%d chars, streaming=%s)",
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        len(messages),
        len(messages[0]["content"]),
        len(messages[-1]["content"]),
        streaming,
    )


def _build_metadata(
    settings: Settings,
    result: LLMResult,
    start: float,
    *,
    streaming: bool,
    prompt_version: str = PROMPT_VERSION,
) -> EstimationMetadata:
    """Valida el resultado, registra las métricas de la llamada y las devuelve."""
    latency_ms = int((time.perf_counter() - start) * 1000)
    if not result.text.strip():
        raise LLMServiceError("El proveedor LLM devolvió una respuesta vacía")
    if result.truncated:
        logger.warning(
            "Respuesta truncada por límite de tokens (LLM_MAX_OUTPUT_TOKENS=%d)",
            settings.LLM_MAX_OUTPUT_TOKENS,
        )
    logger.info(
        "Estimación generada prompt=%s provider=%s model=%s streaming=%s input_tokens=%d "
        "output_tokens=%d cached_tokens=%d latency_ms=%d finish_reason=%s",
        prompt_version,
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        streaming,
        result.input_tokens,
        result.output_tokens,
        result.cached_input_tokens,
        latency_ms,
        result.finish_reason,
    )
    return EstimationMetadata(
        prompt_version=prompt_version,
        model=settings.LLM_MODEL,
        provider=settings.LLM_PROVIDER,
        usage=TokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_input_tokens=result.cached_input_tokens,
        ),
        estimated_cost_usd=estimate_cost_usd(
            settings.LLM_MODEL, result.input_tokens, result.output_tokens
        ),
        latency_ms=latency_ms,
        finish_reason=result.finish_reason,
        truncated=result.truncated,
        created_at=datetime.now(UTC),
    )


async def complete(
    messages: list[dict[str, str]],
    *,
    prompt_version: str = PROMPT_VERSION,
    settings: Settings | None = None,
) -> EstimationResponse:
    """Llama al proveedor con un array de mensajes ya montado (system primero)."""
    settings = settings or get_settings()
    call = _call_openai if settings.LLM_PROVIDER == "openai" else _call_anthropic
    _log_request(settings, messages, streaming=False)

    start = time.perf_counter()
    with _provider_errors(settings.LLM_PROVIDER):
        result = await call(settings, messages)
    metadata = _build_metadata(
        settings, result, start, streaming=False, prompt_version=prompt_version
    )
    return EstimationResponse(**metadata.model_dump(), text=result.text)


async def generate_estimation(
    request: EstimationRequest, settings: Settings | None = None
) -> EstimationResponse:
    return await complete(build_messages(request), settings=settings)


async def extract_structured(
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    model: str,
    max_output_tokens: int = 1000,
    settings: Settings | None = None,
) -> T:
    """Pide al modelo un objeto `response_model` validado (salida estructurada del proveedor).

    Se usa para llamadas auxiliares y baratas, como el extractor de la ficha del proyecto: por
    eso admite otro modelo y otro límite de tokens que los de la estimación.
    """
    settings = (settings or get_settings()).model_copy(
        update={"LLM_MODEL": model, "LLM_MAX_OUTPUT_TOKENS": max_output_tokens}
    )
    parse = _parse_openai if settings.LLM_PROVIDER == "openai" else _parse_anthropic
    _log_request(settings, messages, streaming=False)

    start = time.perf_counter()
    with _provider_errors(settings.LLM_PROVIDER):
        parsed, result = await parse(settings, messages, response_model)
    if parsed is None:  # el modelo se negó o la respuesta se cortó antes de cerrar el JSON
        raise LLMServiceError("El proveedor LLM no devolvió una respuesta estructurada válida")
    logger.info(
        "Salida estructurada %s model=%s input_tokens=%d output_tokens=%d latency_ms=%d",
        response_model.__name__,
        settings.LLM_MODEL,
        result.input_tokens,
        result.output_tokens,
        int((time.perf_counter() - start) * 1000),
    )
    return parsed


async def stream_estimation(
    request: EstimationRequest, settings: Settings | None = None
) -> AsyncIterator[str | EstimationMetadata]:
    """Como generate_estimation, pero entrega el texto a medida que el modelo lo genera.

    Produce fragmentos de texto (str) y, al terminar, un único EstimationMetadata.
    """
    settings = settings or get_settings()
    messages = build_messages(request)
    stream = _stream_openai if settings.LLM_PROVIDER == "openai" else _stream_anthropic
    _log_request(settings, messages, streaming=True)

    start = time.perf_counter()
    result = None
    with _provider_errors(settings.LLM_PROVIDER):
        # aclosing: si el cliente abandona el stream, se cierra también la conexión con el proveedor
        async with aclosing(stream(settings, messages)) as items:
            async for item in items:
                if isinstance(item, LLMResult):
                    result = item
                elif item:
                    yield item
    if result is None:  # el stream terminó sin el evento final con el uso de tokens
        logger.error("El stream de %s terminó sin completarse", settings.LLM_PROVIDER)
        raise LLMServiceError("El proveedor LLM cortó la respuesta antes de terminarla")
    yield _build_metadata(settings, result, start, streaming=True)
