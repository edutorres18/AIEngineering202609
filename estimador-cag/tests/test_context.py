"""Contexto CAG y mensajes que se envían al modelo: se testea sin llamar al LLM."""

from app.context.examples import (
    ESTIMATION_EXAMPLES,
    HOURLY_RATES_EUR,
    build_reference_estimation,
    reference_estimations,
)
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest
from app.services.llm_service import build_messages

REQUEST = EstimationRequest(
    description="  Landing page con formulario de contacto e integración con HubSpot.  ",
    project_type="web_saas",
    detail_level="summary",
    output_format="line_items",
)


def test_examples_are_enough_and_use_known_roles():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"]
        for task in example["tasks"]:
            assert task["role"] in HOURLY_RATES_EUR


def test_reference_costs_and_totals_are_precomputed():
    example = {
        "project": "Proyecto X",
        "project_type": "Web",
        "meeting_summary": "Resumen",
        "tasks": [
            {"task": "Backend", "role": "Desarrollador Backend", "hours": 10},
            {"task": "Diseño", "role": "Diseñador UX/UI", "hours": 20},
        ],
        "team": "1 backend",
        "duration": "2 semanas",
        "risks": ["Ninguno"],
    }
    reference = build_reference_estimation(
        example, {"Desarrollador Backend": 55, "Diseñador UX/UI": 45}
    )
    assert [(t.rate_eur, t.cost_eur) for t in reference.tasks] == [(55, 550), (45, 900)]
    assert (reference.total_hours, reference.total_cost_eur) == (30, 1450)


def test_every_example_becomes_a_reference_estimation():
    references = reference_estimations()
    assert [r.project for r in references] == [e["project"] for e in ESTIMATION_EXAMPLES]


def test_description_is_stripped_before_validating():
    assert REQUEST.description.startswith("Landing page")
    assert REQUEST.description.endswith("HubSpot.")


def test_messages_are_system_then_user_from_the_templates():
    messages = build_messages(REQUEST)

    assert [m["role"] for m in messages] == ["system", "user"]
    system, user = render_estimation_prompt(REQUEST)
    assert messages[0]["content"] == system
    assert messages[1]["content"] == user
    assert REQUEST.description not in system  # la descripción solo viaja en el mensaje de usuario
