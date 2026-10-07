"""Interfaz del Estimador CAG con Streamlit: una conversación con formulario de parámetros tipados.

Es un cliente HTTP de la API FastAPI: no importa el servicio de IA ni los SDK de los
proveedores. El prompt, las API keys, la memoria de la conversación y la lectura de los
adjuntos viven solo en el backend, así que este frontend se puede sustituir por otro sin
tocarlo. Del backend solo reutiliza el contrato (app/schemas): el formulario se valida con el
mismo EstimationRequest que usa la API.

Sesión 5: al cargar la página se crea una sesión (POST /sessions) y su session_id se guarda en
st.session_state. Cada envío es un turno (POST /sessions/{id}/estimate, multipart con los
adjuntos) y el panel lateral muestra la ficha del proyecto que recuerda la API
(GET /sessions/{id}).

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
# Cada turno son dos llamadas al LLM: la estimación y la actualización de la ficha.
TIMEOUT = httpx.Timeout(5.0, read=180.0)
SAMPLE_TRANSCRIPTION = Path(__file__).parent / "transcripciones" / "reunion_red_veterinaria.md"
MAX_DESCRIPTION_CHARS = EstimationRequest.model_json_schema()["properties"]["description"][
    "maxLength"
]
LONG_MESSAGE_CHARS = 600  # a partir de aquí, un mensaje del historial se muestra plegado

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


class SessionLost(ApiError):
    """La API ya no conoce la sesión (404): las sesiones viven en memoria y se pierden al
    reiniciar el servidor, p. ej. cuando uvicorn --reload recarga el código."""


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


def _call(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    """Hace la petición y devuelve el JSON, o lanza ApiError / SessionLost."""
    try:
        response = client.request(method, url, **kwargs)
    except httpx.TransportError as exc:
        raise _connection_error(exc, client.base_url) from exc
    if response.status_code == 404 and url.startswith("/sessions/"):
        raise SessionLost(error_message(response))
    if not response.is_success:
        raise ApiError(error_message(response))
    return response.json()


def get_context(client: httpx.Client, params: dict[str, str]) -> dict:
    """System prompt (para esos parámetros y esa sesión), tarifas y estimaciones de referencia."""
    return _call(client, "GET", "/api/v1/context", params=params)


def create_session(client: httpx.Client) -> str:
    """POST /sessions: abre una conversación vacía y devuelve su session_id."""
    return _call(client, "POST", "/sessions")["session_id"]


def get_session(client: httpx.Client, session_id: str) -> dict:
    """GET /sessions/{id}: ficha del proyecto y tamaño del historial."""
    return _call(client, "GET", f"/sessions/{session_id}")


def request_turn(
    client: httpx.Client, session_id: str, request: EstimationRequest, attachments: list
) -> dict:
    """Envía un turno (multipart/form-data) y devuelve la EstimationResponse.

    `attachments` son los archivos de st.file_uploader (o cualquier objeto con name,
    getvalue() y type).
    """
    form = request.model_dump(mode="json")
    form["transcript"] = form.pop("description")  # en la API de sesiones se llama transcript
    files = [
        ("attachments", (file.name, file.getvalue(), file.type or "application/octet-stream"))
        for file in attachments
    ]
    return _call(client, "POST", f"/sessions/{session_id}/estimate", data=form, files=files or None)


# --- Interfaz ---


def _n(number: int) -> str:
    return f"{number:,}".replace(",", ".")


@st.cache_resource
def get_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


@st.cache_data(ttl=60, show_spinner=False)
def load_context(
    _client: httpx.Client,
    project_type: str,
    detail_level: str,
    output_format: str,
    session_id: str | None,
    turns: int,  # solo invalida la caché: tras cada turno cambia la ficha del system prompt
):
    params = {
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    if session_id:
        params["session_id"] = session_id
    return get_context(_client, params)


def load_sample() -> None:
    """Rellena el formulario con la transcripción de ejemplo (antes de pintar los widgets)."""
    st.session_state.description = SAMPLE_TRANSCRIPTION.read_text(encoding="utf-8")
    st.session_state.project_type = ProjectType.WEB_SAAS


def reset_conversation(notice: str | None = None, *, keep_form: bool = False) -> None:
    """Olvida la conversación de este navegador; la próxima ejecución abre una sesión nueva.

    Con `keep_form`, el formulario conserva lo escrito y los adjuntos (para reenviarlos).
    """
    state = st.session_state
    for key in ("session_id", "last_response", "error"):
        state.pop(key, None)
    state.turns = []
    if not keep_form:
        clear_form()
    if notice:
        state.notice = notice


def clear_form() -> None:
    """Vacía el formulario en la próxima ejecución: un widget ya pintado no se puede cambiar."""
    st.session_state.clear_input = True
    st.session_state.upload_round += 1  # claves nuevas → selectores de archivos vacíos


def ensure_session(client: httpx.Client) -> str:
    """session_id de esta conversación; la crea en la API la primera vez (o lanza ApiError)."""
    if not st.session_state.get("session_id"):
        st.session_state.session_id = create_session(client)
    return st.session_state.session_id


def read_description(text: str, file) -> str:
    """Texto del formulario más el contenido del fichero de transcripción, si lo hay."""
    parts = [text, file.getvalue().decode("utf-8", errors="replace") if file else ""]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def render_form() -> dict | None:
    """Pinta el formulario y, cuando se envía, devuelve sus valores."""
    first_turn = not st.session_state.turns
    upload_round = st.session_state.upload_round
    with st.form("estimation_form", border=True):
        text = st.text_area(
            "Descripción del proyecto" if first_turn else "¿Qué más sabes del proyecto?",
            key="description",
            height=180 if first_turn else 110,
            max_chars=MAX_DESCRIPTION_CHARS,
            placeholder=(
                "Ej.: app móvil para que los socios de un gimnasio reserven clases, paguen la "
                "cuota y reciban avisos. Ya tienen la marca y un backend con la lista de socios."
                if first_turn
                else "Ej.: el cliente quiere añadir pagos con tarjeta y el equipo será de 3 "
                "personas. No hace falta repetir lo anterior: el estimador lo recuerda."
            ),
            help="Lo que hay que construir, o la transcripción de una reunión. En los turnos "
            "siguientes, solo lo que cambia o se añade.",
        )
        left, right = st.columns(2)
        file = left.file_uploader(
            "Transcripción en un archivo",
            type=["txt", "md"],
            key=f"file_{upload_round}",
            help="Su texto se añade a la descripción.",
        )
        attachments = right.file_uploader(
            "Documentación adjunta (PDF o Word)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"attachments_{upload_round}",
            help="Especificaciones, propuestas previas… La API extrae su texto y lo añade al "
            "mensaje.",
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
                "attachments": attachments or [],
            }
    return None


def send_turn(client: httpx.Client, fields: dict) -> str | None:
    """Valida y envía un turno. Devuelve el mensaje de error, o None si ha ido bien."""
    attachments = fields.pop("attachments")
    try:
        request = EstimationRequest(**fields)
    except ValidationError as exc:  # se avisa sin gastar una petición
        return validation_message(exc.errors())
    with st.spinner("Estimando y actualizando la memoria del proyecto…"):
        try:
            response = request_turn(client, ensure_session(client), request, attachments)
        except SessionLost:
            reset_conversation(keep_form=True)
            return (
                "El servidor se reinició y la conversación se perdió (vive en su memoria). "
                "Se ha abierto una nueva: vuelve a enviar el mensaje con todo el contexto."
            )
        except ApiError as exc:
            return str(exc)
    st.session_state.turns.append(
        {
            "message": request.description,
            "attachments": [file.name for file in attachments],
            "response": response,
        }
    )
    st.session_state.last_response = response
    clear_form()
    return None


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


def render_message(text: str, attachments: list[str]) -> None:
    if len(text) <= LONG_MESSAGE_CHARS:
        st.text(text)
    else:
        st.text(text[:LONG_MESSAGE_CHARS] + "…")
        with st.expander(f"Mensaje completo ({_n(len(text))} caracteres)"):
            st.text(text)
    if attachments:
        st.caption("📎 " + " · ".join(attachments))


def render_response(response: dict, number: int) -> None:
    if response["truncated"]:
        st.warning(
            "La estimación se cortó por el límite de tokens de salida (LLM_MAX_OUTPUT_TOKENS).",
            icon="✂️",
        )
    st.markdown(response["text"])
    st.caption(f"Turno {number} · {format_metrics(response)}")


def render_conversation(turns: list[dict]) -> None:
    for number, turn in enumerate(turns, start=1):
        with st.chat_message("user"):
            render_message(turn["message"], turn["attachments"])
        with st.chat_message("assistant"):
            render_response(turn["response"], number)


def render_memory(session: dict | None) -> None:
    """Panel de la ficha del proyecto: la memoria, que vive aparte del historial."""
    st.subheader("Memoria del proyecto")
    if session is None:
        st.caption("Sin sesión: la API no está disponible.")
        return
    st.caption(
        f"Historial: {session['turns']} de {session['max_turns']} turnos · "
        f"sesión `{session['session_id'][:8]}`"
    )
    metadata = session["metadata"]
    if not any(metadata.values()):
        st.caption("Vacía: se rellena con los hechos del proyecto tras el primer turno.")
        return
    team = metadata["assumed_team_size"]
    technologies = metadata["mentioned_technologies"]
    st.markdown(
        "  \n".join(  # dos espacios + salto: salto de línea en Markdown
            [
                f"**Proyecto:** {metadata['project_name'] or '—'}",
                f"**Equipo:** {f'{team} personas' if team else '—'}",
                "**Tecnologías:** "
                + (" ".join(f"`{tech}`" for tech in technologies) if technologies else "—"),
            ]
        )
    )
    st.markdown(f"**Alcance acordado:** {metadata['agreed_scope'] or '—'}")
    st.caption(
        "Va en el bloque <project_metadata> del system prompt de cada turno (ver Contexto CAG)."
    )


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


def render_context(client: httpx.Client, session_id: str | None, turns: int) -> None:
    st.subheader("Contexto CAG")
    # Los widgets de un formulario solo cambian su valor al enviarlo: es lo último que se estimó.
    state = st.session_state
    try:
        context = load_context(
            client,
            state.project_type.value,
            state.detail_level.value,
            state.output_format.value,
            session_id,
            turns,
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
            f"{OUTPUT_FORMATS[state.output_format]} · el del próximo turno, con la ficha al final"
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
    state = st.session_state
    for key, value in FORM_DEFAULTS.items():
        state.setdefault(key, value)
    state.setdefault("turns", [])
    state.setdefault("upload_round", 0)
    if state.pop("clear_input", False):  # tras un turno, antes de pintar el cuadro de texto
        state.description = ""
    client = get_client()

    st.title("📐 Estimador de proyectos")
    st.caption(
        "Describe el proyecto (o pega la transcripción de la reunión con el cliente) y recibe "
        "una estimación. Después puedes seguir precisando el alcance o adjuntar documentos: el "
        "estimador recuerda la conversación."
    )
    left, right = st.columns([3, 2])
    left.button(
        "Probar con una transcripción de ejemplo",
        icon=":material/pets:",
        type="tertiary",
        on_click=load_sample,  # los callbacks se ejecutan antes de pintar el formulario
    )
    right.button(
        "Nueva conversación",
        icon=":material/add_comment:",
        on_click=reset_conversation,
        width="stretch",
    )

    session_id, session = None, None
    try:
        session_id = ensure_session(client)
        session = get_session(client, session_id)
    except SessionLost:
        reset_conversation(
            "El servidor se reinició y la conversación anterior se perdió: empieza una nueva."
        )
        st.rerun()
    except ApiError as exc:
        st.warning(str(exc), icon="🔌")
    if notice := state.pop("notice", None):
        st.info(notice, icon="🔄")

    render_conversation(state.turns)
    if fields := render_form():
        state.error = send_turn(client, fields)
        if state.error is None:
            st.rerun()  # vuelve a pintar con el turno nuevo, la ficha nueva y el cuadro vacío
    if error := state.get("error"):
        st.error(error, icon="⚠️")

    with st.sidebar:
        render_memory(session)
        st.divider()
        render_last_call(state.get("last_response"))
        st.divider()
        render_context(client, session_id, len(state.turns))


if __name__ == "__main__":
    main()
