"""Endpoints de conversación (sesión 5): una sesión recuerda el proyecto entre turnos.

POST /sessions                  → crea una sesión vacía y devuelve su session_id
GET  /sessions/{id}             → ficha del proyecto y tamaño del historial (depuración)
POST /sessions/{id}/estimate    → un turno: transcripción + adjuntos (multipart/form-data)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import StringConstraints

from app.attachments.extractor import enrich_transcript, extract_text
from app.config import get_settings
from app.schemas.estimation import DetailLevel, EstimationResponse, OutputFormat, ProjectType
from app.schemas.session import CreateSessionResponse, SessionState
from app.services import conversation_service
from app.sessions.store import SessionStore, get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

Store = Annotated[SessionStore, Depends(get_session_store)]
# Mismos límites que la descripción de POST /api/v1/estimate (EstimationRequest).
Transcript = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=20, max_length=80_000)
]

NOT_FOUND = {404: {"description": "La sesión no existe (o se perdió al reiniciar el servidor)"}}


async def _read_attachments(uploads: list[UploadFile]) -> list[tuple[str, str]]:
    """(nombre, texto) de cada adjunto, por el camino B (extracción local).

    Un adjunto no admitido o ilegible lanza AttachmentError (415/422, ver main.py).
    """
    max_chars = get_settings().MAX_ATTACHMENT_CHARS
    return [
        (upload.filename, extract_text(upload.filename, await upload.read(), max_chars=max_chars))
        for upload in uploads
        if upload.filename  # un campo de archivo enviado vacío no trae nombre
    ]


@router.post(
    "",
    status_code=201,
    response_model=CreateSessionResponse,
    summary="Crea una sesión de conversación vacía",
)
async def create_session(store: Store) -> CreateSessionResponse:
    session = store.create()
    logger.info(
        "Sesión creada session_id=%s (sesiones en memoria: %d)", session.session_id, len(store)
    )
    return CreateSessionResponse(session_id=session.session_id)


@router.get(
    "/{session_id}",
    response_model=SessionState,
    summary="Ficha del proyecto y tamaño del historial de una sesión",
    responses=NOT_FOUND,
)
async def get_session(session_id: str, store: Store) -> SessionState:
    session = store.get(session_id)  # SessionNotFoundError → 404 (main.py)
    return SessionState(
        session_id=session.session_id,
        metadata=session.metadata,
        turns=session.history.turns,
        max_turns=session.history.max_turns,
    )


@router.post(
    "/{session_id}/estimate",
    response_model=EstimationResponse,
    summary="Un turno de la conversación: transcripción y adjuntos opcionales (PDF o Word)",
    description="La respuesta es la misma EstimationResponse que la de `POST /api/v1/estimate`, "
    "con `prompt_version` v2. El texto de los adjuntos se extrae en el servidor y se añade a la "
    "transcripción. Tras responder, la sesión guarda el turno y actualiza su ficha.",
    responses={
        **NOT_FOUND,
        415: {"description": "Adjunto en un formato no admitido (solo .pdf y .docx)"},
        422: {"description": "Formulario no válido o adjunto ilegible"},
        502: {"description": "Error del proveedor LLM"},
        503: {"description": "Rate limit del proveedor LLM"},
        504: {"description": "Timeout del proveedor LLM"},
    },
)
async def estimate_in_session(
    session_id: str,
    store: Store,
    transcript: Annotated[Transcript, Form(description="Transcripción o mensaje de este turno")],
    project_type: Annotated[ProjectType, Form()],
    detail_level: Annotated[DetailLevel, Form()],
    output_format: Annotated[OutputFormat, Form()],
    attachments: Annotated[
        list[UploadFile] | None, File(description="Documentación complementaria: PDF o Word")
    ] = None,
) -> EstimationResponse:
    session = store.get(session_id)  # SessionNotFoundError → 404 (main.py)
    extracted = await _read_attachments(attachments or [])
    logger.info(
        "Turno recibido session_id=%s transcript_chars=%d adjuntos=%s",
        session_id,
        len(transcript),
        [(name, len(text)) for name, text in extracted],
    )
    return await conversation_service.estimate_in_session(
        session,
        enrich_transcript(transcript, extracted),
        project_type,
        detail_level,
        output_format,
    )
