"""Punto de entrada: crea la aplicación FastAPI y conecta las piezas."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import estimations
from app.services.llm_service import LLMServiceError

settings = get_settings()  # falla al arrancar si la configuración no es válida

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Estimador CAG",
    description=(
        "Genera estimaciones de proyectos de software a partir de transcripciones de "
        "reuniones con clientes. Arquitectura CAG: las estimaciones históricas de "
        "referencia se inyectan en el prompt en cada llamada al LLM."
    ),
    version="0.1.0",
)

app.include_router(estimations.router, prefix="/api/v1")


@app.exception_handler(LLMServiceError)
async def llm_service_error_handler(request: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.APP_ENV,
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL,
    }
