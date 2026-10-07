"""Estado de una conversación: el historial y la ficha del proyecto, por separado.

- ConversationHistory es el historial: los mensajes user/assistant que viajan al LLM en cada
  llamada, con ventana deslizante. Responde a «¿qué se dijo en los últimos turnos?».
- ProjectMetadata es la memoria: los hechos del proyecto en curso (nombre, equipo,
  tecnologías, alcance). Va en el system prompt de cada turno y no depende del historial, así
  que sobrevive cuando la ventana descarta los turnos donde se mencionaron.

El system prompt no se guarda en el historial: se vuelve a generar en cada turno a partir de la
ficha actual (ver prompts/estimation/v2/system.j2).
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationHistory(BaseModel):
    """Ventana deslizante de pares user+assistant.

    `max_turns` cuenta pares. Al superarlo se descartan los pares más antiguos, siempre de dos en
    dos para no dejar una pregunta sin su respuesta (los proveedores esperan que se alternen).
    """

    max_turns: int = Field(default=6, ge=1)
    messages: list[Message] = Field(default_factory=list)

    def append(self, *, user: str, assistant: str) -> None:
        """Añade un turno completo y recorta la ventana."""
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))
        overflow = len(self.messages) - self.max_turns * 2
        if overflow > 0:
            del self.messages[:overflow]

    def to_messages_list(self, system_prompt: str, user: str) -> list[dict[str, str]]:
        """Array `messages` listo para el LLM: system + historial de la ventana + turno nuevo."""
        return [
            {"role": "system", "content": system_prompt},
            *({"role": m.role, "content": m.content} for m in self.messages),
            {"role": "user", "content": user},
        ]

    @property
    def turns(self) -> int:
        return len(self.messages) // 2


class ProjectMetadata(BaseModel):
    """Ficha del proyecto en curso, que se rellena turno a turno.

    Todos los campos son opcionales: en el primer turno todavía no se sabe nada. Es también el
    formato que devuelve el extractor (salida estructurada), por eso cada campo lleva una
    descripción para el modelo.
    """

    project_name: str | None = Field(
        default=None, description="Nombre propio del proyecto; null si no se ha dicho"
    )
    assumed_team_size: int | None = Field(
        default=None, ge=1, le=50, description="Personas del equipo que asume la estimación"
    )
    mentioned_technologies: list[str] = Field(
        default_factory=list, description="Tecnologías, lenguajes, servicios o plataformas"
    )
    agreed_scope: str | None = Field(
        default=None, description="Un párrafo con el alcance acordado hasta ahora"
    )

    def is_empty(self) -> bool:
        return self == ProjectMetadata()

    def merge_with(self, update: "ProjectMetadata") -> "ProjectMetadata":
        """Ficha nueva: los datos no nulos de `update` sustituyen a los anteriores y las
        tecnologías se acumulan (sin repetir, sin distinguir mayúsculas).

        Sustituir es lo correcto para nombre, equipo y alcance, que la conversación va
        precisando. Las tecnologías se suman: mencionar una nueva no borra las anteriores.
        """
        technologies = list(self.mentioned_technologies)
        known = {tech.casefold() for tech in technologies}
        for tech in (t.strip() for t in update.mentioned_technologies):
            if tech and tech.casefold() not in known:
                technologies.append(tech)
                known.add(tech.casefold())
        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


class Session(BaseModel):
    """Una conversación de estimación: historial + ficha, identificados por un UUID v4."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
