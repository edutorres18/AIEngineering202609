"""Construcción del contexto CAG: se testea sin llamar al LLM."""

from app.context.examples import ESTIMATION_EXAMPLES, HOURLY_RATES_EUR
from app.services.llm_service import (
    build_messages,
    build_system_prompt,
    format_example,
)


def test_examples_are_enough_and_use_known_roles():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"]
        for task in example["tasks"]:
            assert task["role"] in HOURLY_RATES_EUR


def test_example_costs_and_totals_are_precomputed():
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
    text = format_example(example, {"Desarrollador Backend": 55, "Diseñador UX/UI": 45})
    assert "| Backend | Desarrollador Backend | 10 | 55 | 550 |" in text
    assert "Horas totales: 30 h" in text
    assert "Coste total: 1.450 EUR" in text


def test_system_prompt_injects_every_example_with_delimiters():
    prompt = build_system_prompt()
    for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        assert f"===== ESTIMACIÓN DE REFERENCIA {i} =====" in prompt
        assert example["project"] in prompt
    assert "===== FIN DE ESTIMACIONES DE REFERENCIA =====" in prompt


def test_system_prompt_order_instructions_examples_rules():
    prompt = build_system_prompt()
    role = prompt.index("Eres un consultor senior")
    output_format = prompt.index("## Formato de la respuesta")
    first_example = prompt.index("===== ESTIMACIÓN DE REFERENCIA 1 =====")
    rules = prompt.index("## Reglas")
    assert role < output_format < first_example < rules


def test_messages_follow_system_then_user():
    messages = build_messages("  Transcripción de prueba  ")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "<transcripcion>\nTranscripción de prueba\n</transcripcion>" in messages[1]["content"]
