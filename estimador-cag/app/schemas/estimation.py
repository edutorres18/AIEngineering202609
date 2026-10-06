"""Contratos de datos de los endpoints de estimación."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProjectType(StrEnum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(StrEnum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(StrEnum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    """Lo que envía el formulario: el texto a estimar y tres parámetros cerrados."""

    model_config = ConfigDict(
        str_strip_whitespace=True,  # "   " no cuenta como descripción
        json_schema_extra={
            "examples": [
                {
                    "description": (
                        "El cliente necesita una landing page con formulario de contacto, "
                        "integración con su CRM actual (HubSpot) y una sección de blog con "
                        "editor WYSIWYG. El diseño ya existe en Figma."
                    ),
                    "project_type": "web_saas",
                    "detail_level": "medium",
                    "output_format": "phases_table",
                }
            ]
        },
    )

    description: str = Field(
        min_length=20,
        # Como en la solución de referencia del curso: admite también la transcripción completa
        # de una reunión (~20K tokens), no solo una descripción corta.
        max_length=80_000,
        description="Descripción del proyecto o transcripción de la reunión con el cliente",
    )
    project_type: ProjectType = Field(description="Tipo de proyecto")
    detail_level: DetailLevel = Field(description="Nivel de detalle de la estimación")
    output_format: OutputFormat = Field(description="Formato de la estimación")


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


class EstimationMetadata(BaseModel):
    """Métricas de una llamada. En streaming se envían en el evento final `done`."""

    prompt_version: str = Field(description="Versión de las plantillas de prompt (p. ej. v1)")
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
    text: str = Field(description="Estimación generada por el modelo, en Markdown")


class ReferenceTask(BaseModel):
    task: str
    role: str
    hours: int
    rate_eur: int
    cost_eur: int


class ReferenceEstimation(BaseModel):
    """Estimación histórica con los costes y totales ya calculados (se inyecta en el prompt)."""

    project: str
    project_type: str
    meeting_summary: str
    tasks: list[ReferenceTask]
    total_hours: int
    total_cost_eur: int
    team: str
    duration: str
    risks: list[str]


class CAGContext(BaseModel):
    """Contexto que recibe el modelo con unos parámetros dados (para mostrarlo en la interfaz)."""

    provider: Literal["openai", "anthropic"]
    model: str
    prompt_version: str
    system_prompt: str
    hourly_rates_eur: dict[str, int]
    examples: list[ReferenceEstimation]
