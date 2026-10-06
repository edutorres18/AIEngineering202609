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

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_VERSION = "v1"  # versión que usan los endpoints

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
) -> str:
    """System prompt para unos parámetros dados (no depende de la descripción)."""
    template = _env.get_template(f"estimation/{version}/system.j2")
    return template.render(
        project_type=project_type.value,
        detail_level=detail_level.value,
        output_format=output_format.value,
        hourly_rates=HOURLY_RATES_EUR,
        reference_estimations=reference_estimations(),
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
