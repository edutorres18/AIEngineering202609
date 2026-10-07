"""Estado de la sesión: ventana deslizante del historial y fusión de la ficha del proyecto."""

import pytest

from app.sessions.models import ConversationHistory, ProjectMetadata
from app.sessions.store import SessionNotFoundError, SessionStore


def fill(history: ConversationHistory, turns: int) -> None:
    for n in range(1, turns + 1):
        history.append(user=f"pregunta {n}", assistant=f"respuesta {n}")


def test_history_keeps_every_turn_below_the_limit():
    history = ConversationHistory(max_turns=3)
    fill(history, 2)

    assert history.turns == 2
    assert [m.content for m in history.messages] == [
        "pregunta 1",
        "respuesta 1",
        "pregunta 2",
        "respuesta 2",
    ]


def test_history_drops_the_oldest_pairs_beyond_the_limit():
    history = ConversationHistory(max_turns=3)
    fill(history, 5)

    assert history.turns == 3
    # Se descartan pares completos: el historial sigue empezando por una pregunta del usuario
    assert [m.role for m in history.messages] == ["user", "assistant"] * 3
    assert history.messages[0].content == "pregunta 3"
    assert history.messages[-1].content == "respuesta 5"


def test_messages_list_is_fresh_system_then_window_then_new_message():
    history = ConversationHistory(max_turns=2)
    fill(history, 3)

    messages = history.to_messages_list("system con la ficha", "pregunta 4")

    assert messages[0] == {"role": "system", "content": "system con la ficha"}
    assert messages[-1] == {"role": "user", "content": "pregunta 4"}
    assert [m["content"] for m in messages[1:-1]] == [
        "pregunta 2",
        "respuesta 2",
        "pregunta 3",
        "respuesta 3",
    ]
    assert len(messages) == 1 + 2 * 2 + 1


def test_new_metadata_is_empty():
    assert ProjectMetadata().is_empty()
    assert not ProjectMetadata(project_name="Nimbus").is_empty()


def test_merge_overwrites_known_facts_and_keeps_the_rest():
    previous = ProjectMetadata(
        project_name="Nimbus", assumed_team_size=2, agreed_scope="CRM con contactos"
    )
    update = ProjectMetadata(assumed_team_size=4, agreed_scope="CRM con contactos y facturación")

    merged = previous.merge_with(update)

    assert merged.project_name == "Nimbus"  # null en la actualización: se conserva
    assert merged.assumed_team_size == 4
    assert merged.agreed_scope == "CRM con contactos y facturación"


def test_merge_accumulates_technologies_without_duplicates():
    previous = ProjectMetadata(mentioned_technologies=["React", "PostgreSQL"])
    update = ProjectMetadata(mentioned_technologies=["postgresql", " Stripe ", ""])

    merged = previous.merge_with(update)

    assert merged.mentioned_technologies == ["React", "PostgreSQL", "Stripe"]
    assert previous.mentioned_technologies == ["React", "PostgreSQL"]  # no muta la anterior


def test_team_size_must_be_plausible():
    with pytest.raises(ValueError):
        ProjectMetadata(assumed_team_size=0)


def test_store_creates_independent_sessions_with_its_window():
    store = SessionStore(max_turns=4)
    first, second = store.create(), store.create()

    assert first.session_id != second.session_id
    assert store.get(first.session_id) is first
    assert first.history.max_turns == 4
    assert len(store) == 2


def test_store_reports_unknown_sessions():
    with pytest.raises(SessionNotFoundError):
        SessionStore().get("no-existe")
