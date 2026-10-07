"""Carga y renderiza las plantillas Jinja2 versionadas de los prompts.

Estructura en disco: app/prompts/<caso de uso>/<versión>/<rol>.j2. Para probar otro prompt se
crea una carpeta nueva (v2/, v3/…) al lado de las anteriores y se cambia `version`: el resto
del código no se toca. Este módulo es el único punto donde Python toca las plantillas.
"""

import hashlib
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.context.examples import HOURLY_RATES_EUR, reference_estimations
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.sessions.models import ProjectMetadata

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"  # estimación de un solo turno (POST /api/v1/estimate)
CONVERSATION_PROMPT_VERSION = "v2"  # estimación dentro de una sesión (POST /sessions/…)
METADATA_PROMPT_VERSION = "v1"  # extractor de la ficha del proyecto

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    undefined=StrictUndefined,  # una variable que falta rompe el render en vez de quedar vacía
    trim_blocks=True,  # sin el salto de línea que deja cada {% ... %}
    lstrip_blocks=True,  # sin la indentación que hay delante de cada {% ... %}
    keep_trailing_newline=False,
    autoescape=False,  # el destino es un LLM, no HTML: el texto va tal cual
)
_env.filters["eur"] = lambda amount: f"{amount:,}".replace(",", ".")  # 1450 → "1.450"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def render_system_prompt(
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    version: str = PROMPT_VERSION,
    metadata: ProjectMetadata | None = None,
) -> str:
    """System prompt para unos parámetros dados (no depende de la descripción).

    `metadata` es la ficha del proyecto que inyecta v2; v1 no la usa.
    """
    template = _env.get_template(f"estimation/{version}/system.j2")
    return template.render(
        project_type=project_type.value,
        detail_level=detail_level.value,
        output_format=output_format.value,
        hourly_rates=HOURLY_RATES_EUR,
        reference_estimations=reference_estimations(),
        metadata=metadata or ProjectMetadata(),
    )


def render_estimation_prompt(
    request: EstimationRequest, version: str = PROMPT_VERSION
) -> tuple[str, str]:
    """Devuelve (system, user), listos para enviarse al modelo como mensajes separados."""
    system = render_system_prompt(
        request.project_type, request.detail_level, request.output_format, version
    )
    user = _env.get_template(f"estimation/{version}/user.j2").render(
        description=request.description
    )
    # Versión y hash permiten saber qué prompt exacto recibió el modelo sin registrar su contenido.
    logger.debug(
        "Prompt renderizado version=%s system_sha256=%s user_chars=%d",
        version,
        _short_hash(system),
        len(user),
    )
    return system, user


def render_conversation_prompt(
    description: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    metadata: ProjectMetadata,
    version: str = CONVERSATION_PROMPT_VERSION,
) -> tuple[str, str]:
    """(system, user) de un turno de conversación: el system lleva la ficha del proyecto.

    `description` es la transcripción del turno ya enriquecida con el texto de los adjuntos.
    """
    system = render_system_prompt(project_type, detail_level, output_format, version, metadata)
    user = _env.get_template(f"estimation/{version}/user.j2").render(description=description)
    logger.debug(
        "Prompt de conversación renderizado version=%s system_sha256=%s user_chars=%d",
        version,
        _short_hash(system),
        len(user),
    )
    return system, user


def render_metadata_extraction_prompt(
    previous: ProjectMetadata,
    transcript: str,
    estimation: str,
    version: str = METADATA_PROMPT_VERSION,
) -> tuple[str, str]:
    """(system, user) del extractor que actualiza la ficha tras cada turno."""
    system = _env.get_template(f"metadata_extraction/{version}/system.j2").render()
    user = _env.get_template(f"metadata_extraction/{version}/user.j2").render(
        previous=previous, transcript=transcript, estimation=estimation
    )
    return system, user
