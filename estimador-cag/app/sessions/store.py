"""Almacén de sesiones en la memoria del proceso: un dict indexado por session_id.

La volatilidad es deliberada en esta fase:
- Si se reinicia el servidor (o uvicorn --reload recarga el código), se pierden todas las
  conversaciones. El cliente lo detecta (404) y abre una sesión nueva.
- Con varios workers, cada uno tendría su propio dict y una conversación podría caer en un
  worker que no la conoce. Con uvicorn y un único worker, el modo de desarrollo, no pasa.
Persistir en Redis o Postgres cambiaría solo esta clase: routers y servicio no la ven por dentro.
"""

from functools import lru_cache

from app.config import get_settings
from app.sessions.models import ConversationHistory, Session


class SessionNotFoundError(KeyError):
    """No hay ninguna sesión con ese id (no existe o se perdió al reiniciar)."""


class SessionStore:
    def __init__(self, *, max_turns: int = 6) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self._max_turns))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def __len__(self) -> int:
        return len(self._sessions)


@lru_cache
def get_session_store() -> SessionStore:
    """Almacén único del proceso (dependencia de FastAPI; los tests la sustituyen)."""
    return SessionStore(max_turns=get_settings().MAX_CONVERSATION_TURNS)
