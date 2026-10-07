"""Actualiza la ficha del proyecto tras cada turno con una segunda llamada al LLM.

Se eligió un extractor LLM frente a una heurística (regex sobre la respuesta): la estimación es
Markdown libre y el cliente escribe como quiere, así que las regex se romperían con cualquier
paráfrasis. La llamada usa un prompt corto y un modelo barato (METADATA_EXTRACTOR_MODEL), y la
salida estructurada del proveedor devuelve directamente un ProjectMetadata validado.

Si la extracción falla, se conserva la ficha anterior: perder la actualización de un turno es
aceptable; que la conversación se caiga por ello, no.
"""

import logging

from app.config import Settings, get_settings
from app.prompts.loader import render_metadata_extraction_prompt
from app.services import llm_service
from app.sessions.models import ProjectMetadata

logger = logging.getLogger(__name__)


async def update_metadata(
    previous: ProjectMetadata,
    transcript: str,
    estimation: str,
    settings: Settings | None = None,
) -> ProjectMetadata:
    """Ficha anterior fusionada con los hechos del último turno (o la anterior si falla)."""
    settings = settings or get_settings()
    system, user = render_metadata_extraction_prompt(previous, transcript, estimation)
    try:
        extracted = await llm_service.extract_structured(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            ProjectMetadata,
            model=settings.METADATA_EXTRACTOR_MODEL,
            settings=settings,
        )
    except Exception as exc:  # cualquier fallo deja la ficha como estaba
        logger.warning("No se pudo actualizar la ficha del proyecto: %s", type(exc).__name__)
        return previous
    merged = previous.merge_with(extracted)
    logger.info(
        "Ficha actualizada project_name=%r team_size=%s technologies=%d scope=%s",
        merged.project_name,
        merged.assumed_team_size,
        len(merged.mentioned_technologies),
        merged.agreed_scope is not None,
    )
    return merged
