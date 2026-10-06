"""Tests de las plantillas de estimación v1: se renderizan sin llamar a ningún LLM.

Comprueban lo que contiene el prompt para una entrada dada, no lo que responde el modelo.
"""

import itertools
import os

import pytest
from jinja2 import TemplateNotFound

from app.context.examples import HOURLY_RATES_EUR, reference_estimations
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

DESCRIPTION = "App móvil para un gimnasio: reserva de clases, pago de cuotas y avisos push."
PHASES_TABLE_KEYWORD = "Confianza (%)"  # columna que solo pide el formato phases_table
ASSUMPTIONS_PER_PHASE = "supuestos por fase"  # instrucción que solo da el nivel detailed


def make_request(**overrides) -> EstimationRequest:
    fields = {
        "description": DESCRIPTION,
        "project_type": ProjectType.MOBILE_APP,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    return EstimationRequest(**(fields | overrides))


def render_system(**overrides) -> str:
    system, _ = render_estimation_prompt(make_request(**overrides))
    return system


# --- Los tres tests que pide el ejercicio ---


def test_description_goes_inside_project_description_block():
    _, user = render_estimation_prompt(make_request())

    start = user.index("<project_description>")
    end = user.index("</project_description>")
    assert DESCRIPTION in user[start:end]


def test_phases_table_keyword_only_with_phases_table_format():
    assert PHASES_TABLE_KEYWORD in render_system(output_format=OutputFormat.PHASES_TABLE)
    assert PHASES_TABLE_KEYWORD not in render_system(output_format=OutputFormat.NARRATIVE)


def test_assumptions_per_phase_only_with_detailed_level():
    assert ASSUMPTIONS_PER_PHASE in render_system(detail_level=DetailLevel.DETAILED)
    assert ASSUMPTIONS_PER_PHASE not in render_system(detail_level=DetailLevel.SUMMARY)


# --- Más garantías de la plantilla ---


@pytest.mark.parametrize(
    ("output_format", "keyword"),
    [
        (OutputFormat.PHASES_TABLE, "### Desglose por fases"),
        (OutputFormat.LINE_ITEMS, "### Desglose de tareas\n(tabla"),
        (OutputFormat.NARRATIVE, "sin tablas ni listas"),
    ],
)
def test_each_output_format_renders_only_its_own_instructions(output_format, keyword):
    for other in OutputFormat:
        assert (keyword in render_system(output_format=other)) == (other == output_format)


@pytest.mark.parametrize(
    ("detail_level", "keyword"),
    [
        (DetailLevel.SUMMARY, "Nivel de detalle: resumen"),
        (DetailLevel.MEDIUM, "Nivel de detalle: medio"),
        (DetailLevel.DETAILED, "Nivel de detalle: detallado"),
    ],
)
def test_each_detail_level_renders_only_its_own_instructions(detail_level, keyword):
    for other in DetailLevel:
        assert (keyword in render_system(detail_level=other)) == (other == detail_level)


def test_project_type_is_named_in_the_system_prompt():
    assert "como pipeline de datos" in render_system(project_type=ProjectType.DATA_PIPELINE)
    assert "como app móvil" in render_system(project_type=ProjectType.MOBILE_APP)


def test_examples_are_included_with_their_precomputed_totals():
    system = render_system()

    assert system.count("<example number=") == len(reference_estimations())
    for example in reference_estimations():
        assert f"## Estimación: {example.project}" in system
        total_cost = f"{example.total_cost_eur:,}".replace(",", ".")
        assert f"- Coste total: {total_cost} EUR" in system
    for role, rate in HOURLY_RATES_EUR.items():
        assert f"| {role} | {rate} |" in system


@pytest.mark.parametrize(
    ("project_type", "detail_level", "output_format"),
    list(itertools.product(ProjectType, DetailLevel, OutputFormat)),
)
def test_every_combination_renders_clean_text(project_type, detail_level, output_format):
    system, user = render_estimation_prompt(
        make_request(
            project_type=project_type, detail_level=detail_level, output_format=output_format
        )
    )
    for text in (system, user):
        assert "{{" not in text and "{%" not in text and "{#" not in text
        assert "\n\n\n" not in text  # trim_blocks / lstrip_blocks: sin huecos de los bloques
        assert text == text.strip()


def test_static_part_of_system_prompt_is_shared_by_every_combination():
    # Prefijo idéntico en cada llamada: el proveedor lo puede cachear (prompt caching).
    prompts = [
        render_system(project_type=p, detail_level=d, output_format=f)
        for p, d, f in itertools.product(ProjectType, DetailLevel, OutputFormat)
    ]
    prefix = os.path.commonprefix(prompts)
    assert "</examples>" in prefix
    assert "</rules>" in prefix


def test_description_is_inserted_as_text_not_as_template():
    _, user = render_estimation_prompt(make_request(description="Calcula {{ 7 * 7 }} {% if %}"))
    assert "Calcula {{ 7 * 7 }} {% if %}" in user


def test_unknown_version_fails_loudly():
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(make_request(), version="v999")
