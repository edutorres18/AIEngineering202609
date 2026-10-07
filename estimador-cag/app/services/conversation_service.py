"""Un turno de conversación (sesión 5): estimación con memoria dentro de una sesión.

Pasos de cada turno:
    1. Renderizar el prompt v2: el system lleva la ficha actual del proyecto (<project_metadata>).
    2. Montar el array: system nuevo + turnos de la ventana deslizante + mensaje nuevo.
    3. Llamar al LLM (mismo wrapper, errores y métricas que la estimación de un solo turno).
    4. Guardar el turno en el historial (la ventana descarta los pares más antiguos).
    5. Segunda llamada: el extractor actualiza la ficha con los hechos del turno.

Si el paso 3 falla, la sesión no cambia: el cliente puede reintentar el mismo turno.
"""

from app.config import Settings, get_settings
from app.prompts.loader import CONVERSATION_PROMPT_VERSION, render_conversation_prompt
from app.schemas.estimation import DetailLevel, EstimationResponse, OutputFormat, ProjectType
from app.services import llm_service
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import Session


async def estimate_in_session(
    session: Session,
    transcript: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    settings: Settings | None = None,
) -> EstimationResponse:
    """`transcript` llega ya enriquecido con el texto de los adjuntos."""
    settings = settings or get_settings()
    system, user = render_conversation_prompt(
        transcript, project_type, detail_level, output_format, session.metadata
    )
    messages = session.history.to_messages_list(system, user)
    response = await llm_service.complete(
        messages, prompt_version=CONVERSATION_PROMPT_VERSION, settings=settings
    )
    session.history.append(user=user, assistant=response.text)
    session.metadata = await update_metadata(
        session.metadata, transcript, response.text, settings=settings
    )
    return response
