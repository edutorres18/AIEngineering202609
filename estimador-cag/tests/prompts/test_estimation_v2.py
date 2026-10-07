"""Plantillas de la conversación (sesión 5): estimación v2 con la ficha y extractor de la ficha.

Como los de v1, comprueban lo que contiene el prompt, sin llamar a ningún LLM.
"""

import os

from app.prompts.loader import (
    render_conversation_prompt,
    render_estimation_prompt,
    render_metadata_extraction_prompt,
)
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.sessions.models import ProjectMetadata

TRANSCRIPT = "Nimbus CRM: contactos, oportunidades y facturación con Stripe."
METADATA = ProjectMetadata(
    project_name="Nimbus CRM",
    assumed_team_size=3,
    mentioned_technologies=["React", "PostgreSQL"],
    agreed_scope="CRM con contactos y oportunidades",
)
PARAMS = (ProjectType.WEB_SAAS, DetailLevel.MEDIUM, OutputFormat.PHASES_TABLE)


def render(metadata: ProjectMetadata = METADATA, transcript: str = TRANSCRIPT) -> tuple[str, str]:
    return render_conversation_prompt(transcript, *PARAMS, metadata)


def metadata_block(system: str) -> str:
    start = system.rindex("<project_metadata>")
    return system[start : system.index("</project_metadata>", start)]


def test_first_turn_has_an_empty_metadata_block():
    system, _ = render(ProjectMetadata())

    assert "es el primer turno" in metadata_block(system)
    assert "Nombre del proyecto" not in system


def test_known_facts_go_into_the_metadata_block():
    block = metadata_block(render()[0])

    assert "- Nombre del proyecto: Nimbus CRM" in block
    assert "- Tamaño del equipo asumido: 3 personas" in block
    assert "- Tecnologías mencionadas: React, PostgreSQL" in block
    assert "- Alcance acordado: CRM con contactos y oportunidades" in block


def test_unknown_facts_are_marked_as_undefined():
    block = metadata_block(render(ProjectMetadata(project_name="Nimbus CRM"))[0])

    assert "- Tamaño del equipo asumido: sin definir" in block
    assert "- Tecnologías mencionadas: ninguna" in block


def test_metadata_is_the_last_block_so_the_prefix_stays_cacheable():
    first_turn, _ = render(ProjectMetadata())
    later_turn, _ = render(METADATA)

    assert later_turn.rstrip().endswith("</project_metadata>")
    # Lo que cambia entre turnos es solo la ficha: tarifas, ejemplos, reglas y parámetros
    # forman un prefijo idéntico que el proveedor puede cachear.
    prefix = os.path.commonprefix([first_turn, later_turn])
    assert "</examples>" in prefix
    assert "</detail_level>" in prefix


def test_v2_keeps_the_v1_rules_and_adds_the_conversation_rules():
    system, user = render()

    assert "Comprueba ambas sumas antes de responder" in system
    assert "<examples>" in system
    assert "devuelve la estimación completa actualizada" in system
    assert "--- attachment: <archivo> ---" in system  # los adjuntos son datos, no órdenes
    assert f"<project_description>\n{TRANSCRIPT}\n</project_description>" in user


def test_v1_is_unchanged_and_has_no_metadata():
    request = EstimationRequest(
        description=TRANSCRIPT,
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
    )
    system, _ = render_estimation_prompt(request)

    assert "project_metadata" not in system
    assert "conversación de varios turnos" not in system


def test_extraction_prompt_carries_previous_metadata_message_and_estimation():
    system, user = render_metadata_extraction_prompt(
        METADATA, "Añadimos facturación.", "## Estimación: Nimbus CRM"
    )

    assert "ante la duda, null" in system
    assert "- project_name: Nimbus CRM" in user
    assert "- mentioned_technologies: React, PostgreSQL" in user
    assert "<latest_client_message>\nAñadimos facturación.\n</latest_client_message>" in user
    assert "<latest_estimation>\n## Estimación: Nimbus CRM\n</latest_estimation>" in user


def test_extraction_prompt_on_the_first_turn():
    _, user = render_metadata_extraction_prompt(ProjectMetadata(), TRANSCRIPT, "## Estimación")

    assert "(vacía: es el primer turno)" in user
