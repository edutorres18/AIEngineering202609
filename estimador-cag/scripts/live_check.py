"""Prueba end-to-end contra el servidor en marcha y con el LLM real.

Envía una transcripción como descripción a POST /api/v1/estimate, con el formato
de desglose de tareas (line_items) y el nivel de detalle medio, y evalúa la respuesta
con comprobaciones deterministas (sin LLM juez): estructura, tarifas del contexto,
aritmética y uso de las referencias inyectadas.

Uso:
    uv run python scripts/live_check.py [--url http://localhost:8000] [--file transcripciones/x.md]
"""

import argparse
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.context.examples import ESTIMATION_EXAMPLES, HOURLY_RATES_EUR  # noqa: E402

REQUIRED_SECTIONS = [
    "## Estimación:",
    "### Resumen del proyecto",
    "### Desglose de tareas",
    "### Totales",
    "### Equipo recomendado",
    "### Duración estimada",
    "### Supuestos y riesgos",
    "### Referencia utilizada",
]
DEFAULT_FILE = Path(__file__).resolve().parent.parent / "transcripciones/reunion_red_veterinaria.md"
# Las comprobaciones de abajo son las del formato line_items (tabla de tareas con tarifas).
PARAMETERS = {"project_type": "web_saas", "detail_level": "medium", "output_format": "line_items"}


def parse_number(text: str) -> int | None:
    digits = re.sub(r"[^\d]", "", text.split(",")[0])
    return int(digits) if digits else None


def task_rows(estimation: str) -> list[list[str]]:
    rows = []
    for line in estimation.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        is_task = len(cells) == 5 and "---" not in line
        if is_task and parse_number(cells[2]) is not None and parse_number(cells[3]) is not None:
            rows.append(cells)
    return rows


def declared_total(estimation: str, label: str) -> int | None:
    match = re.search(rf"{label}:\W*([\d.,]+)", estimation, re.IGNORECASE)
    return parse_number(match.group(1)) if match else None


def totals_match(rows: list[list[str]], estimation: str) -> bool:
    """Compara los totales declarados con la suma real del desglose."""
    hours = sum(parse_number(r[2]) for r in rows)
    cost = sum(parse_number(r[4]) or 0 for r in rows)
    declared_hours = declared_total(estimation, "Horas totales")
    declared_cost = declared_total(estimation, "Coste total")
    if (declared_hours, declared_cost) != (hours, cost):
        print(
            f"  ⚠ Totales declarados: {declared_hours} h / {declared_cost} EUR — "
            f"suma real del desglose: {hours} h / {cost} EUR"
        )
        return False
    return True


def evaluate(estimation: str) -> dict[str, bool]:
    rows = task_rows(estimation)
    rates = set(HOURLY_RATES_EUR.values())
    lower = estimation.lower()
    references = [e["project_type"].lower() for e in ESTIMATION_EXAMPLES] + [
        " ".join(e["project"].lower().split()[:2]) for e in ESTIMATION_EXAMPLES
    ]
    return {
        "Tiene todas las secciones": all(s in estimation for s in REQUIRED_SECTIONS),
        "Desglose con al menos 5 tareas": len(rows) >= 5,
        "Usa solo tarifas del contexto": bool(rows)
        and all(parse_number(r[3]) in rates for r in rows),
        "Horas en múltiplos de 5": bool(rows) and all(parse_number(r[2]) % 5 == 0 for r in rows),
        "Coste = horas × tarifa": bool(rows)
        and all(parse_number(r[2]) * parse_number(r[3]) == parse_number(r[4]) for r in rows),
        "Totales = suma del desglose": bool(rows) and totals_match(rows, estimation),
        "Cita una estimación de referencia": any(ref in lower for ref in references),
        "Moneda EUR": "eur" in lower or "€" in estimation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--min-score", type=float, default=0.7)
    args = parser.parse_args()

    health = httpx.get(f"{args.url}/health", timeout=10)
    print(f"GET /health → {health.status_code} {health.json()}")

    payload = {"description": args.file.read_text(encoding="utf-8")} | PARAMETERS
    response = httpx.post(f"{args.url}/api/v1/estimate", json=payload, timeout=120)
    print(f"POST /api/v1/estimate {PARAMETERS} → {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return 1
    body = response.json()

    print("\n" + body["text"] + "\n")
    print(
        f"Prompt: {body['prompt_version']} | proveedor/modelo: {body['provider']} / {body['model']}"
    )
    print(f"Tokens: {body['usage']} | latencia: {body['latency_ms']} ms")
    print(
        f"Coste aprox.: {body['estimated_cost_usd']} USD | finish_reason: {body['finish_reason']}"
    )

    checks = evaluate(body["text"])
    checks["Respuesta no truncada"] = not body["truncated"]
    print("\nEvaluación determinista:")
    for name, ok in checks.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    score = sum(checks.values()) / len(checks)
    print(f"\nScore: {score:.2f} (mínimo {args.min_score})")
    return 0 if score >= args.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
