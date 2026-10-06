"""Contratos de datos de los endpoints de estimación."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class EstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=50,
        max_length=50_000,  # ~12K tokens: acota el presupuesto de contexto del mensaje de usuario
        description="Transcripción de la reunión con el cliente",
        examples=[
            "En la reunión con el equipo de marketing, el cliente explicó que necesita una "
            "landing page con formulario de contacto, integración con su CRM actual (HubSpot) "
            "y una sección de blog con editor WYSIWYG. El diseño ya existe en Figma."
        ],
    )


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


class EstimationMetadata(BaseModel):
    """Métricas de una llamada. En streaming se envían en el evento final `done`."""

    model: str
    provider: Literal["openai", "anthropic"]
    usage: TokenUsage
    estimated_cost_usd: float | None = Field(
        description="Coste aproximado en USD; null si el modelo no está en la tabla de precios"
    )
    latency_ms: int
    finish_reason: str
    truncated: bool = Field(description="True si la respuesta se cortó por el límite de tokens")
    created_at: datetime


class EstimationResponse(EstimationMetadata):
    estimation: str = Field(description="Estimación generada por el modelo, en Markdown")


class ReferenceEstimation(BaseModel):
    project: str
    project_type: str
    content: str = Field(description="Bloque en Markdown tal como se inyecta en el system prompt")


class CAGContext(BaseModel):
    """Contexto estático que recibe el modelo en cada llamada (para mostrarlo en la interfaz)."""

    provider: Literal["openai", "anthropic"]
    model: str
    system_prompt: str
    hourly_rates_eur: dict[str, int]
    examples: list[ReferenceEstimation]
