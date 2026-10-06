"""Interfaz de chat del Estimador CAG con Streamlit.

Es un cliente HTTP de la API FastAPI: no importa el servicio de IA ni los SDK de los
proveedores. El prompt, las API keys y el manejo de errores del LLM viven solo en el backend,
así que este frontend se puede sustituir por otro sin tocarlo.

Uso, con la API en marcha (make run):
    streamlit run streamlit_app.py      # o: make ui
La URL de la API se cambia con la variable de entorno ESTIMADOR_API_URL.
"""

import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("ESTIMADOR_API_URL", "http://localhost:8000")
# La primera lectura espera al primer token del modelo; las siguientes, entre fragmentos.
TIMEOUT = httpx.Timeout(5.0, read=120.0)
SAMPLE_TRANSCRIPTION = Path(__file__).parent / "transcripciones" / "reunion_red_veterinaria.md"
MAX_TRANSCRIPTION_CHARS = 50_000  # mismo límite que valida la API
PREVIEW_CHARS = 300
# Los límites los decide la API; aquí solo se traducen sus errores de validación de Pydantic.
VALIDATION_MESSAGES = {
    "string_too_short": (
        "La transcripción es demasiado corta: escribe al menos {min_length} caracteres "
        "para que haya algo que estimar."
    ),
    "string_too_long": (
        "La transcripción es demasiado larga: el máximo son {max_length} caracteres."
    ),
}


# --- Cliente de la API (sin Streamlit: se testea por separado) ---


class ApiError(Exception):
    """La API devolvió un error o no se pudo hablar con ella."""


def _connection_error(exc: httpx.TransportError, base_url: httpx.URL) -> ApiError:
    if isinstance(exc, httpx.TimeoutException):
        return ApiError("La API no respondió a tiempo. Vuelve a intentarlo.")
    return ApiError(f"No se pudo conectar con la API en {base_url}. ¿Está arrancada? (`make run`)")


def _validation_message(error: dict) -> str:
    template = VALIDATION_MESSAGES.get(error.get("type"))
    return template.format(**error.get("ctx", {})) if template else error["msg"]


def error_message(response: httpx.Response) -> str:
    """Mensaje de una respuesta de error de FastAPI ({"detail": ...})."""
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return f"La API respondió con un error HTTP {response.status_code}."
    if isinstance(detail, list):  # 422: errores de validación de Pydantic
        return "; ".join(_validation_message(error) for error in detail)
    return str(detail)


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Agrupa las líneas de un stream text/event-stream en pares (evento, datos JSON)."""
    event, data = "message", []
    for line in lines:
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").removeprefix(" "))
        elif not line:  # una línea vacía cierra el evento
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        # el resto son comentarios (": ping" de keep-alive) y se ignoran


def get_context(client: httpx.Client) -> dict:
    """System prompt, tarifas y estimaciones de referencia que usa el backend."""
    try:
        response = client.get("/api/v1/context")
    except httpx.TransportError as exc:
        raise _connection_error(exc, client.base_url) from exc
    if response.status_code != 200:
        raise ApiError(error_message(response))
    return response.json()


def start_estimation(client: httpx.Client, transcription: str) -> httpx.Response:
    """Envía la transcripción al endpoint de streaming y devuelve la respuesta abierta.

    La API contesta cuando el modelo ya tiene el primer fragmento, así que los errores de
    entrada (validación, API key, rate limit) llegan aquí como códigos HTTP.
    """
    request = client.build_request(
        "POST", "/api/v1/estimate/stream", json={"transcription": transcription}
    )
    try:
        response = client.send(request, stream=True)
    except httpx.TransportError as exc:
        raise _connection_error(exc, client.base_url) from exc
    if response.status_code != 200:
        response.read()
        response.close()
        raise ApiError(error_message(response))
    return response


def iter_estimation(response: httpx.Response, metrics: dict) -> Iterator[str]:
    """Entrega el texto de la estimación según llega y deja en `metrics` los metadatos finales."""
    try:
        for event, data in parse_sse(response.iter_lines()):
            if event == "delta":
                yield data["text"]
            elif event == "done":
                metrics.update(data)
                return
            elif event == "error":
                raise ApiError(data["detail"])
    except httpx.TransportError as exc:
        raise ApiError("Se perdió la conexión con la API durante la respuesta.") from exc
    finally:
        response.close()
    raise ApiError("La respuesta de la API terminó sin completarse.")


# --- Interfaz ---


def _n(number: int) -> str:
    return f"{number:,}".replace(",", ".")


@st.cache_resource
def get_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


@st.cache_data(ttl=60, show_spinner=False)
def load_context(_client: httpx.Client) -> dict:
    return get_context(_client)


def read_chat_input(value) -> str:
    """Texto escrito en el chat más el contenido de los ficheros adjuntos."""
    parts = [value.text or ""]
    parts += [file.getvalue().decode("utf-8", errors="replace") for file in value.files]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def last_metrics(messages: list[dict]) -> dict | None:
    return next((m["metrics"] for m in reversed(messages) if m.get("metrics")), None)


def format_metrics(metrics: dict) -> str:
    usage = metrics["usage"]
    parts = [
        f"{metrics['provider']} / {metrics['model']}",
        f"{_n(usage['input_tokens'])} tokens de entrada",
        f"{_n(usage['output_tokens'])} de salida",
        f"{metrics['latency_ms'] / 1000:.1f} s",
    ]
    if metrics["estimated_cost_usd"] is not None:
        parts.append(f"~{metrics['estimated_cost_usd']:.4f} USD")
    return " · ".join(parts)


def render_transcription(text: str) -> None:
    if len(text) <= PREVIEW_CHARS:
        st.text(text)
        return
    st.text(text[:PREVIEW_CHARS].rstrip() + " …")
    with st.expander(f"Ver la transcripción completa ({_n(len(text))} caracteres)"):
        st.text(text)


def render_answer_footer(message: dict) -> None:
    if message["error"]:
        st.error(message["error"], icon="⚠️")
    if metrics := message["metrics"]:
        if metrics["truncated"]:
            st.warning(
                "La estimación se cortó por el límite de tokens de salida (LLM_MAX_OUTPUT_TOKENS).",
                icon="✂️",
            )
        st.caption(format_metrics(metrics))


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            render_transcription(message["content"])
        else:
            if message["content"]:
                st.markdown(message["content"])
            render_answer_footer(message)


def answer(client: httpx.Client, transcription: str) -> dict:
    """Muestra la estimación en streaming y devuelve el mensaje para el historial."""
    message = {"role": "assistant", "content": "", "metrics": None, "error": None}
    metrics: dict = {}
    start = time.perf_counter()
    try:
        with st.spinner("Analizando la transcripción…"):
            response = start_estimation(client, transcription)
        metrics["first_token_ms"] = int((time.perf_counter() - start) * 1000)
        message["content"] = st.write_stream(iter_estimation(response, metrics))
        message["metrics"] = metrics
    except ApiError as exc:
        # Un texto a medias se queda en pantalla, pero no en el historial: no es una estimación.
        message["error"] = str(exc)
    render_answer_footer(message)
    return message


def render_last_call(metrics: dict | None) -> None:
    st.subheader("Última llamada")
    if not metrics:
        st.caption("Todavía no hay ninguna estimación en esta conversación.")
        return
    usage = metrics["usage"]
    st.markdown(f"Modelo: `{metrics['provider']} / {metrics['model']}`")
    left, right = st.columns(2)
    left.metric("Tokens entrada", _n(usage["input_tokens"]))
    right.metric("Tokens salida", _n(usage["output_tokens"]))
    left.metric("Tiempo total", f"{metrics['latency_ms'] / 1000:.1f} s")
    right.metric("Primer token", f"{metrics['first_token_ms'] / 1000:.1f} s")
    details = [f"{_n(usage['cached_input_tokens'])} tokens de entrada cacheados"]
    if metrics["estimated_cost_usd"] is not None:
        details.append(f"coste aprox. {metrics['estimated_cost_usd']:.4f} USD")
    st.caption(" · ".join(details))


def render_context(client: httpx.Client) -> None:
    st.subheader("Contexto CAG")
    try:
        context = load_context(client)
    except ApiError as exc:
        st.warning(str(exc), icon="🔌")
        return
    st.caption(
        f"Lo que recibe el modelo en cada llamada · "
        f"`{context['provider']} / {context['model']}` en {API_URL}"
    )
    with st.expander("System prompt activo"):
        st.code(context["system_prompt"], language="markdown", wrap_lines=True, height=400)
    with st.expander("Tarifas por perfil"):
        rows = [f"| {role} | {rate} |" for role, rate in context["hourly_rates_eur"].items()]
        st.markdown("\n".join(["| Perfil | EUR/h |", "| --- | --: |", *rows]))
    st.markdown("**Estimaciones de referencia**")
    for example in context["examples"]:
        with st.expander(example["project"]):
            st.caption(example["project_type"])
            st.code(example["content"], language="markdown", wrap_lines=True)


def main() -> None:
    st.set_page_config(page_title="Estimador CAG", page_icon="📐", initial_sidebar_state=420)
    st.session_state.setdefault("messages", [])
    messages: list[dict] = st.session_state.messages
    client = get_client()

    with st.sidebar:
        # Se rellena al final, cuando la llamada de esta ejecución ya ha terminado.
        last_call_box = st.container()
        st.divider()
        render_context(client)

    st.title("📐 Estimador de proyectos")
    st.caption(
        "Pega la transcripción de una reunión con un cliente y recibe una estimación de "
        "desarrollo. Cada mensaje se estima por separado."
    )

    for message in messages:
        render_message(message)

    value = st.chat_input(
        "Pega aquí la transcripción o adjunta un .txt / .md",
        accept_file=True,
        file_type=["txt", "md"],
        max_chars=MAX_TRANSCRIPTION_CHARS,
    )
    transcription = read_chat_input(value) if value else ""
    if not messages:
        example_slot = st.empty()
        if example_slot.button("Probar con una transcripción de ejemplo", icon=":material/pets:"):
            transcription = SAMPLE_TRANSCRIPTION.read_text(encoding="utf-8")
        if transcription:
            example_slot.empty()

    if transcription:
        user_message = {"role": "user", "content": transcription}
        messages.append(user_message)
        render_message(user_message)
        with st.chat_message("assistant"):
            messages.append(answer(client, transcription))

    with last_call_box:
        st.button(
            "Nueva conversación",
            icon=":material/add_comment:",
            width="stretch",
            disabled=not messages,
            on_click=messages.clear,  # los callbacks se ejecutan antes de la siguiente ejecución
        )
        render_last_call(last_metrics(messages))


if __name__ == "__main__":
    main()
