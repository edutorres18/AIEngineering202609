"""Capa de transporte HTTP: recibe, delega en el servicio y devuelve."""

import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event

from app.schemas.estimation import (
    CAGContext,
    DetailLevel,
    EstimationMetadata,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.services import llm_service
from app.services.llm_service import LLMServiceError
from app.sessions.store import SessionStore, get_session_store

router = APIRouter(tags=["estimations"])

ERROR_RESPONSES = {
    502: {"description": "Error del proveedor LLM"},
    503: {"description": "Rate limit del proveedor LLM"},
    504: {"description": "Timeout del proveedor LLM"},
}

STREAM_DESCRIPTION = """\
Misma estimación que `POST /estimate`, enviada con Server-Sent Events a medida que se genera.

Eventos (el campo `data` siempre es JSON):
- `delta` → `{"text": "..."}`: siguiente fragmento del Markdown de la estimación.
- `done` → metadatos de la llamada (versión del prompt, modelo, tokens, coste, latencia,
  truncado). Cierra el stream.
- `error` → `{"detail": "...", "status_code": 502}` si el proveedor falla con el stream ya empezado.

Si el proveedor falla antes del primer fragmento, la respuesta es un error HTTP normal
(502/503/504), igual que en `POST /estimate`."""


class SSEResponse(StreamingResponse):
    """Respuesta text/event-stream.

    No se usa EventSourceResponse como response_class porque obliga a que el endpoint sea un
    generador, y entonces un fallo del proveedor antes del primer token ya no podría
    devolverse como error HTTP.
    """

    media_type = "text/event-stream"


def _sse(item: str | EstimationMetadata) -> bytes:
    if isinstance(item, str):
        return format_sse_event(
            event="delta", data_str=json.dumps({"text": item}, ensure_ascii=False)
        )
    return format_sse_event(event="done", data_str=item.model_dump_json())


async def _sse_stream(
    first: str | EstimationMetadata,
    items: AsyncIterator[str | EstimationMetadata],
) -> AsyncIterator[bytes]:
    async with aclosing(items):
        try:
            yield _sse(first)
            async for item in items:
                yield _sse(item)
        except LLMServiceError as exc:
            # Con el stream empezado ya no se puede cambiar el código HTTP: el error va como evento.
            error = {"detail": exc.message, "status_code": exc.status_code}
            yield format_sse_event(event="error", data_str=json.dumps(error, ensure_ascii=False))


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    summary="Genera una estimación de software a partir de la descripción del proyecto",
    responses=ERROR_RESPONSES,
)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    return await llm_service.generate_estimation(request)


@router.post(
    "/estimate/stream",
    summary="Genera la estimación en streaming (Server-Sent Events)",
    description=STREAM_DESCRIPTION,
    response_class=SSEResponse,
    responses=ERROR_RESPONSES,
)
async def estimate_stream(request: EstimationRequest) -> SSEResponse:
    items = llm_service.stream_estimation(request)
    # Se espera al primer fragmento antes de responder: si el proveedor falla de entrada
    # (API key, rate limit, timeout), el cliente recibe el código HTTP de error habitual.
    first = await anext(items)
    return SSEResponse(
        _sse_stream(first, items),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/context",
    response_model=CAGContext,
    summary="Contexto CAG que recibe el modelo: system prompt, tarifas y referencias",
    description="El system prompt depende de los parámetros del formulario: se pueden indicar "
    "aquí para ver el que recibe el modelo con cada combinación. Con `session_id`, es el system "
    "prompt v2 que recibirá el próximo turno de esa conversación, con su ficha del proyecto.",
    responses={404: {"description": "La sesión no existe"}},
)
async def context(
    store: Annotated[SessionStore, Depends(get_session_store)],
    project_type: Annotated[ProjectType, Query()] = ProjectType.WEB_SAAS,
    detail_level: Annotated[DetailLevel, Query()] = DetailLevel.MEDIUM,
    output_format: Annotated[OutputFormat, Query()] = OutputFormat.PHASES_TABLE,
    session_id: Annotated[str | None, Query(description="Sesión de conversación")] = None,
) -> CAGContext:
    metadata = store.get(session_id).metadata if session_id else None
    return llm_service.get_cag_context(project_type, detail_level, output_format, metadata=metadata)
