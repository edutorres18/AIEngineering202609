"""Contratos de datos de los endpoints de conversación (sesión 5)."""

from pydantic import BaseModel, Field

from app.sessions.models import ProjectMetadata


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID v4 de la sesión; viaja en cada petición posterior")


class SessionState(BaseModel):
    """Vista de depuración de una sesión: la ficha y el tamaño del historial."""

    session_id: str
    metadata: ProjectMetadata = Field(description="Ficha del proyecto (memoria conversacional)")
    turns: int = Field(description="Pares user+assistant que hay ahora en el historial")
    max_turns: int = Field(description="Tamaño de la ventana deslizante (MAX_CONVERSATION_TURNS)")
