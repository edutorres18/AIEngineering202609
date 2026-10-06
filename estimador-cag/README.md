# Estimador CAG

Servicio FastAPI que recibe la **transcripción de una reunión con un cliente** y devuelve una
**estimación de proyecto de software** (desglose de tareas, horas, costes, equipo, duración y
riesgos) generada por un LLM.

Es el Proyecto 1 del programa AI Engineering 2026/09 (LIDR), en su primera fase:
**arquitectura CAG (Cache Augmented Generation)**. Todo el contexto que necesita el modelo
(tarifas y estimaciones históricas de la empresa) viaja en cada llamada. No hay base de datos,
ni retrieval, ni persistencia.

## Arquitectura

```mermaid
flowchart LR
    C[Cliente<br/>curl / Swagger] -->|POST /api/v1/estimate| R[routers/estimations.py<br/>valida con Pydantic]
    R --> S[services/llm_service.py<br/>construye el prompt CAG]
    X[context/examples.py<br/>tarifas + estimaciones históricas] --> S
    S -->|system + user| L[(LLM<br/>OpenAI / Anthropic)]
    L --> S
    S -->|EstimationResponse| R --> C
```

| Capa | Archivo | Responsabilidad |
| --- | --- | --- |
| Entrada | `app/main.py` | Crea la app, registra el router con prefijo `/api/v1`, `/health`, traduce errores del LLM a HTTP |
| Configuración | `app/config.py` | `BaseSettings` que lee `.env` y valida al arrancar (si falta la API key del proveedor, no arranca) |
| Transporte | `app/routers/estimations.py` | Endpoint fino: recibe, delega y devuelve |
| Negocio | `app/services/llm_service.py` | Preprocesa el contexto, arma los mensajes, llama al proveedor y normaliza la respuesta |
| Contratos | `app/schemas/estimation.py` | Request/response con Pydantic (documentados en Swagger) |
| Contexto CAG | `app/context/examples.py` | Tarifas por perfil y 3 estimaciones históricas ficticias |

Estructura de mensajes enviada al modelo:

```
[system]    → rol + uso del contexto + formato de salida + tarifas + estimaciones de referencia + reglas
[user]      → <transcripcion>…</transcripcion>
[assistant] → estimación en Markdown
```

## Decisiones de diseño

- **CAG en lugar de RAG.** Tres estimaciones de referencia ocupan unos 1.800 tokens: caben de
  sobra en la ventana de contexto de `gpt-4o-mini` (128K) o `claude-haiku-4-5` (200K). Cuando
  el histórico crezca a cientos de presupuestos, `context/` se sustituirá por un servicio de
  búsqueda semántica sin tocar routers ni schemas.
- **Contexto estructurado y preprocesado.** Los ejemplos se guardan como datos (tareas, perfil,
  horas) y el servicio **calcula costes y totales** antes de inyectarlos. Así el modelo no tiene
  que hacer aritmética para entender las referencias, y las tarifas tienen una única fuente de verdad.
- **Ejemplos variados.** E-commerce web, app móvil e integración/herramienta interna, para
  calibrar sin sesgar el modelo hacia una sola tipología.
- **Orden del prompt por "lost in the middle".** Instrucciones y formato al principio, referencias
  (delimitadas con `===== ESTIMACIÓN DE REFERENCIA N =====`) en el medio, reglas al final y la
  transcripción como último mensaje.
- **Formato de los ejemplos igual al de la salida esperada** (Markdown con tabla), porque el
  modelo imita lo que ve en su contexto.
- **Prefijo estático.** El system prompt se construye una sola vez y es idéntico en cada llamada,
  lo que permite que el proveedor lo cachee (prompt caching) cuando supere el tamaño mínimo.
- **Dos proveedores tras una misma función.** `LLM_PROVIDER` elige OpenAI (Responses API) o
  Anthropic (Messages API). Las diferencias (system como mensaje o como parámetro, nombre de los
  campos de uso, `finish_reason`) se normalizan en `LLMResult`.
- **Clientes asíncronos** (`AsyncOpenAI` / `AsyncAnthropic`): una llamada al LLM tarda segundos y
  no debe bloquear el event loop de FastAPI.

## Latencia, coste, calidad y seguridad

- **Latencia:** cada respuesta incluye `latency_ms`. Hay timeout configurable
  (`LLM_TIMEOUT_SECONDS`) y 2 reintentos automáticos del SDK.
- **Coste:** se usan modelos económicos por defecto y la respuesta incluye `usage` (tokens de
  entrada, salida y cacheados) y `estimated_cost_usd`. `max_length` en la transcripción y
  `LLM_MAX_OUTPUT_TOKENS` acotan el presupuesto de tokens.
- **Calidad:** `scripts/live_check.py` evalúa una estimación real con comprobaciones
  deterministas (secciones, tarifas del contexto, horas múltiplo de 5, coste = horas × tarifa,
  cita de la referencia usada, no truncada) y devuelve un score.
- **Truncado:** si el modelo corta por límite de tokens, la respuesta lo indica con
  `truncated: true` y se registra un warning.
- **Seguridad:** las API keys solo se leen de `.env` (ignorado por git) y se manejan como
  `SecretStr`. La transcripción va delimitada y el prompt indica tratarla como datos, no como
  instrucciones. Con OpenAI se usa `store=False` para no almacenar transcripciones de clientes.
  Los logs registran métricas, no el contenido de la transcripción.
- **Errores del proveedor:** timeout → `504`, rate limit → `503`, otros errores → `502`, sin
  exponer detalles internos.

## Puesta en marcha

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) y una API key de OpenAI
o Anthropic. `uv` instala Python 3.13 automáticamente si no lo tienes.

```bash
cd estimador-cag
uv sync
cp .env.example .env   # completa OPENAI_API_KEY (o ANTHROPIC_API_KEY + LLM_PROVIDER=anthropic)
uv run uvicorn app.main:app --reload
```

- Swagger UI: http://localhost:8000/docs
- Health: `curl http://localhost:8000/health`

Ejemplo de petición:

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"transcription": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño ya existe en Figma."}'
```

Respuesta (resumida):

```json
{
  "estimation": "## Estimación: ...\n\n### Resumen del proyecto\n...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "usage": {"input_tokens": 2350, "output_tokens": 780, "cached_input_tokens": 0},
  "estimated_cost_usd": 0.00082,
  "latency_ms": 9120,
  "finish_reason": "completed",
  "truncated": false,
  "created_at": "2026-10-06T10:00:00Z"
}
```

### Transcripción de prueba

[`transcripciones/reunion_red_veterinaria.md`](transcripciones/reunion_red_veterinaria.md) es una
reunión ficticia (red de clínicas veterinarias: reservas online, integración con su software de
gestión y recordatorios por WhatsApp). Incluye charla irrelevante a propósito, para comprobar
que el modelo la ignora. Para estimarla y evaluar el resultado con el servidor en marcha:

```bash
uv run python scripts/live_check.py
```

## Tests y validación automática

```bash
uv run pytest        # estructura, prompt CAG y endpoints con el LLM simulado (sin coste)
uv run ruff check .  # lint
```

- `tests/test_structure.py`: la estructura de carpetas es la del ejercicio, `.env` está en
  `.gitignore`, `.env.example` documenta las variables sin valores y no hay keys en el código.
- `tests/test_prompt.py`: los ejemplos se inyectan con delimitadores, los costes se precalculan
  bien y el orden del prompt es instrucciones → referencias → reglas.
- `tests/test_api.py`: `/health`, `/docs`, `POST /api/v1/estimate` (con el proveedor simulado),
  validación de entrada (422) y traducción de errores del proveedor (503).

El pipeline [`.github/workflows/estimador-cag-ci.yml`](../.github/workflows/estimador-cag-ci.yml)
se ejecuta en cada push: comprueba que no haya `.env` versionado, instala con `uv sync --locked`,
pasa ruff y pytest y arranca el servidor para probar `/health`, `/docs` y el OpenAPI. Lanzado a
mano (`workflow_dispatch`) y con el secreto `OPENAI_API_KEY` configurado, también ejecuta la
prueba end-to-end con el LLM real.

## Variables de entorno

| Variable | Descripción | Por defecto |
| --- | --- | --- |
| `OPENAI_API_KEY` | API key de OpenAI | Requerida si se usa OpenAI |
| `ANTHROPIC_API_KEY` | API key de Anthropic | Requerida si se usa Anthropic |
| `LLM_PROVIDER` | `openai` o `anthropic` | `openai` |
| `LLM_MODEL` | Modelo a utilizar | `gpt-4o-mini` / `claude-haiku-4-5` |
| `LLM_MAX_OUTPUT_TOKENS` | Límite de tokens de salida | `3000` |
| `LLM_TEMPERATURE` | Temperatura (solo OpenAI) | `0.2` |
| `LLM_TIMEOUT_SECONDS` | Timeout de la llamada al LLM | `60` |
| `APP_ENV` | Entorno de ejecución | `development` |
| `LOG_LEVEL` | Nivel de logging | `DEBUG` |

## Limitaciones

- Single-turn: no se puede refinar una estimación en varios turnos (no hay historial).
- Los ejemplos de referencia son ficticios y estáticos. La calidad depende de lo representativos
  que sean.
- La aritmética la hace el modelo: los totales de la estimación generada pueden tener errores
  (el `live_check` los detecta, pero el servicio todavía no los corrige).
- La salida es Markdown libre, no datos estructurados validados.
- Sin autenticación, rate limiting propio ni guardrails de entrada/salida.
- La tabla de precios para `estimated_cost_usd` es aproximada y hay que actualizarla a mano.
