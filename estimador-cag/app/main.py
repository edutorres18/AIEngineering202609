"""Punto de entrada: crea la aplicación FastAPI y conecta las piezas."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.attachments.extractor import AttachmentError
from app.config import get_settings
from app.routers import estimations, sessions
from app.services.llm_service import LLMServiceError
from app.sessions.store import SessionNotFoundError

settings = get_settings()  # falla al arrancar si la configuración no es válida

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Estimador CAG",
    description=(
        "Genera estimaciones de proyectos de software a partir de la descripción del proyecto "
        "(o la transcripción de una reunión con el cliente) y de tres parámetros: tipo de "
        "proyecto, nivel de detalle y formato. Arquitectura CAG: las estimaciones históricas "
        "de referencia se inyectan en el prompt en cada llamada al LLM. El prompt sale de "
        "plantillas Jinja2 versionadas. Las sesiones (/sessions) mantienen una conversación de "
        "varios turnos: recuerdan el proyecto en curso y aceptan adjuntos PDF o Word."
    ),
    version="0.1.0",
)

app.include_router(estimations.router, prefix="/api/v1")
app.include_router(sessions.router)  # /sessions, como en el enunciado y la solución de referencia


@app.exception_handler(LLMServiceError)
async def llm_service_error_handler(request: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    # Las sesiones viven en memoria: tras reiniciar el servidor, el cliente debe crear otra.
    detail = "La sesión no existe o se perdió al reiniciar el servidor: crea una nueva."
    return JSONResponse(status_code=404, content={"detail": detail})


@app.exception_handler(AttachmentError)
async def attachment_error_handler(request: Request, exc: AttachmentError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.APP_ENV,
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL,
    }
