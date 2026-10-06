"""Interfaz del Estimador CAG con Streamlit: un formulario con parámetros tipados.

Es un cliente HTTP de la API FastAPI: no importa el servicio de IA ni los SDK de los
proveedores. El prompt, las API keys y el manejo de errores del LLM viven solo en el backend,
así que este frontend se puede sustituir por otro sin tocarlo. Del backend solo reutiliza el
contrato (app/schemas): el formulario produce exactamente el EstimationRequest que espera la API.

Uso, con la API en marcha (make run):
    streamlit run streamlit_app.py      # o: make ui
La URL de la API se cambia con la variable de entorno ESTIMADOR_API_URL.
"""

import os
from pathlib import Path

import httpx
import streamlit as st
from pydantic import ValidationError

from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

API_URL = os.environ.get("ESTIMADOR_API_URL", "http://localhost:8000")
# La API responde cuando el modelo ha terminado la estimación completa.
TIMEOUT = httpx.Timeout(5.0, read=120.0)
SAMPLE_TRANSCRIPTION = Path(__file__).parent / "transcripciones" / "reunion_red_veterinaria.md"
MAX_DESCRIPTION_CHARS = EstimationRequest.model_json_schema()["properties"]["description"][
    "maxLength"
]

# Etiquetas de la interfaz; a la API viaja el valor del enum ("mobile_app", "summary"…).
PROJECT_TYPES = {
    ProjectType.MOBILE_APP: "App móvil",
    ProjectType.WEB_SAAS: "Aplicación web / SaaS",
    ProjectType.INTERNAL_TOOL: "Herramienta interna",
    ProjectType.DATA_PIPELINE: "Pipeline de datos",
}
DETAIL_LEVELS = {
    DetailLevel.SUMMARY: "Resumen",
    DetailLevel.MEDIUM: "Medio",
    DetailLevel.DETAILED: "Detallado",
}
OUTPUT_FORMATS = {
    OutputFormat.PHASES_TABLE: "Tabla por fases",
    OutputFormat.LINE_ITEMS: "Desglose de tareas",
    OutputFormat.NARRATIVE: "Texto narrativo",
}
FORM_DEFAULTS = {
    "description": "",
    "project_type": ProjectType.WEB_SAAS,
    "detail_level": DetailLevel.MEDIUM,
    "output_format": OutputFormat.PHASES_TABLE,
}

# Los límites los define el contrato (EstimationRequest); aquí solo se traducen sus errores.
VALIDATION_MESSAGES = {
    "string_too_short": (
        "La descripción es demasiado corta: escribe al menos {min_length} caracteres "
        "para que haya algo que estimar."
    ),
    "string_too_long": (
        "La descripción es demasiado larga: el máximo son {max_length} caracteres."
    ),
}


# --- Cliente de la API (sin Streamlit: se testea por separado) ---


class ApiError(Exception):
    """La API devolvió un error o no se pudo hablar con ella."""


def _connection_error(exc: httpx.TransportError, base_url: httpx.URL) -> ApiError:
    if isinstance(exc, httpx.TimeoutException):
        return ApiError("La API no respondió a tiempo. Vuelve a intentarlo.")
    return ApiError(f"No se pudo conectar con la API en {base_url}. ¿Está arrancada? (`make run`)")


def validation_message(errors: list[dict]) -> str:
    """Errores de validación de Pydantic (locales o del 422 de la API) en lenguaje natural."""
    messages = []
    for error in errors:
        template = VALIDATION_MESSAGES.get(error.get("type"))
        messages.append(template.format(**error.get("ctx", {})) if template else error["msg"])
    return "; ".join(messages)


def error_message(response: httpx.Response) -> str:
    """Mensaje de una respuesta de error de FastAPI ({"detail": ...})."""
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return f"La API respondió con un error HTTP {response.status_code}."
    if isinstance(detail, list):  # 422: errores de validación de Pydantic
        return validation_message(detail)
    return str(detail)


def get_context(client: httpx.Client, params: dict[str, str]) -> dict:
    """System prompt (para esos parámetros), tarifas y estimaciones de referencia."""
    try:
        response = client.get("/api/v1/context", params=params)
    except httpx.TransportError as exc:
        raise _connection_error(exc, client.base_url) from exc
    if response.status_code != 200:
        raise ApiError(error_message(response))
    return response.json()


def request_estimation(client: httpx.Client, request: EstimationRequest) -> dict:
    """Envía el formulario a POST /api/v1/estimate y devuelve la EstimationResponse."""
    try:
        response = client.post("/api/v1/estimate", json=request.model_dump(mode="json"))
    except httpx.TransportError as exc:
        raise _connection_error(exc, client.base_url) from exc
    if response.status_code != 200:
        raise ApiError(error_message(response))
    return response.json()


# --- Interfaz ---


def _n(number: int) -> str:
    return f"{number:,}".replace(",", ".")


@st.cache_resource
def get_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


@st.cache_data(ttl=60, show_spinner=False)
def load_context(_client: httpx.Client, project_type: str, detail_level: str, output_format: str):
    params = {
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    return get_context(_client, params)


def load_sample() -> None:
    """Rellena el formulario con la transcripción de ejemplo (antes de pintar los widgets)."""
    st.session_state.description = SAMPLE_TRANSCRIPTION.read_text(encoding="utf-8")
    st.session_state.project_type = ProjectType.WEB_SAAS


def read_description(text: str, file) -> str:
    """Texto del formulario más el contenido del fichero adjunto, si lo hay."""
    parts = [text, file.getvalue().decode("utf-8", errors="replace") if file else ""]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def render_form() -> dict | None:
    """Pinta el formulario y, cuando se envía, devuelve sus valores."""
    with st.form("estimation_form", border=True):
        text = st.text_area(
            "Descripción del proyecto",
            key="description",
            height=180,
            max_chars=MAX_DESCRIPTION_CHARS,
            placeholder=(
                "Ej.: app móvil para que los socios de un gimnasio reserven clases, paguen la "
                "cuota y reciban avisos. Ya tienen la marca y un backend con la lista de socios."
            ),
            help="Lo que hay que construir. También puedes pegar la transcripción de una reunión.",
        )
        file = st.file_uploader(
            "…o adjunta la transcripción de una reunión", type=["txt", "md"], key="file"
        )
        left, right = st.columns(2)
        project_type = left.selectbox(
            "Tipo de proyecto",
            options=list(PROJECT_TYPES),
            format_func=PROJECT_TYPES.get,
            key="project_type",
        )
        output_format = right.selectbox(
            "Formato de salida",
            options=list(OUTPUT_FORMATS),
            format_func=OUTPUT_FORMATS.get,
            key="output_format",
        )
        detail_level = st.pills(
            "Nivel de detalle",
            options=list(DETAIL_LEVELS),
            format_func=DETAIL_LEVELS.get,
            required=True,
            key="detail_level",
        )
        if st.form_submit_button("Generar estimación", type="primary", icon=":material/calculate:"):
            return {
                "description": read_description(text, file),
                "project_type": project_type,
                "detail_level": detail_level,
                "output_format": output_format,
            }
    return None


def estimate(client: httpx.Client, fields: dict) -> dict:
    """Valida el formulario con el contrato de la API, la llama y devuelve el resultado."""
    result = {"response": None, "error": None}
    try:
        request = EstimationRequest(**fields)
    except ValidationError as exc:  # se avisa sin gastar una petición
        result["error"] = validation_message(exc.errors())
        return result
    with st.spinner("Estimando el proyecto…"):
        try:
            result["response"] = request_estimation(client, request)
        except ApiError as exc:
            result["error"] = str(exc)
    return result


def format_metrics(response: dict) -> str:
    usage = response["usage"]
    parts = [
        f"prompt {response['prompt_version']}",
        f"{response['provider']} / {response['model']}",
        f"{_n(usage['input_tokens'])} tokens de entrada",
        f"{_n(usage['output_tokens'])} de salida",
        f"{response['latency_ms'] / 1000:.1f} s",
    ]
    if response["estimated_cost_usd"] is not None:
        parts.append(f"~{response['estimated_cost_usd']:.4f} USD")
    return " · ".join(parts)


def render_result(result: dict) -> None:
    if result["error"]:
        st.error(result["error"], icon="⚠️")
        return
    response = result["response"]
    with st.container(border=True):
        if response["truncated"]:
            st.warning(
                "La estimación se cortó por el límite de tokens de salida (LLM_MAX_OUTPUT_TOKENS).",
                icon="✂️",
            )
        st.markdown(response["text"])
        st.caption(format_metrics(response))


def render_last_call(response: dict | None) -> None:
    st.subheader("Última llamada")
    if not response:
        st.caption("Todavía no hay ninguna estimación.")
        return
    usage = response["usage"]
    st.markdown(f"Modelo: `{response['provider']} / {response['model']}`")
    left, right = st.columns(2)
    left.metric("Tokens entrada", _n(usage["input_tokens"]))
    right.metric("Tokens salida", _n(usage["output_tokens"]))
    left.metric("Tiempo total", f"{response['latency_ms'] / 1000:.1f} s")
    right.metric("Prompt", response["prompt_version"])
    details = [f"{_n(usage['cached_input_tokens'])} tokens de entrada cacheados"]
    if response["estimated_cost_usd"] is not None:
        details.append(f"coste aprox. {response['estimated_cost_usd']:.4f} USD")
    st.caption(" · ".join(details))


def format_reference(example: dict) -> str:
    rows = [
        f"| {task['task']} | {task['role']} | {task['hours']} | {_n(task['cost_eur'])} |"
        for task in example["tasks"]
    ]
    return "\n".join(
        [
            example["meeting_summary"],
            "",
            "| Tarea | Perfil | Horas | Coste (EUR) |",
            "| --- | --- | --: | --: |",
            *rows,
            f"| **Total** | | **{example['total_hours']}** | **{_n(example['total_cost_eur'])}** |",
        ]
    )


def render_context(client: httpx.Client) -> None:
    st.subheader("Contexto CAG")
    # Los widgets de un formulario solo cambian su valor al enviarlo: es lo último que se estimó.
    state = st.session_state
    try:
        context = load_context(
            client, state.project_type.value, state.detail_level.value, state.output_format.value
        )
    except ApiError as exc:
        st.warning(str(exc), icon="🔌")
        return
    st.caption(
        f"Lo que recibe el modelo · `{context['provider']} / {context['model']}` en {API_URL}"
    )
    with st.expander(f"System prompt {context['prompt_version']}"):
        st.caption(
            f"{PROJECT_TYPES[state.project_type]} · {DETAIL_LEVELS[state.detail_level]} · "
            f"{OUTPUT_FORMATS[state.output_format]}"
        )
        st.code(context["system_prompt"], language="markdown", wrap_lines=True, height=400)
    with st.expander("Tarifas por perfil"):
        rows = [f"| {role} | {rate} |" for role, rate in context["hourly_rates_eur"].items()]
        st.markdown("\n".join(["| Perfil | EUR/h |", "| --- | --: |", *rows]))
    st.markdown("**Estimaciones de referencia**")
    for example in context["examples"]:
        with st.expander(example["project"]):
            st.caption(example["project_type"])
            st.markdown(format_reference(example))


def main() -> None:
    st.set_page_config(page_title="Estimador CAG", page_icon="📐", initial_sidebar_state=420)
    for key, value in FORM_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    client = get_client()

    st.title("📐 Estimador de proyectos")
    st.caption(
        "Describe el proyecto (o pega la transcripción de la reunión con el cliente), elige el "
        "tipo, el formato y el nivel de detalle, y recibe una estimación de desarrollo."
    )
    st.button(
        "Probar con una transcripción de ejemplo",
        icon=":material/pets:",
        type="tertiary",
        on_click=load_sample,  # los callbacks se ejecutan antes de pintar el formulario
    )

    if fields := render_form():
        st.session_state.result = estimate(client, fields)
        if response := st.session_state.result["response"]:
            st.session_state.last_response = response
    if result := st.session_state.get("result"):
        render_result(result)

    with st.sidebar:
        render_last_call(st.session_state.get("last_response"))
        st.divider()
        render_context(client)


if __name__ == "__main__":
    main()
