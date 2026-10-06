"""Datos de referencia que se inyectan en cada llamada al LLM (arquitectura CAG).

Son estimaciones históricas ficticias de la empresa. Se guardan como datos
estructurados (no como texto libre) para que el servicio pueda normalizarlos y
precalcular los campos derivados (costes y totales) antes de inyectarlos:
el modelo no tiene que hacer aritmética para entender las referencias.
La forma en que se presentan al modelo vive en la plantilla
`app/prompts/estimation/<versión>/examples.j2`.

Cuando el proyecto evolucione a RAG, esta capa se sustituirá por un servicio de
búsqueda semántica sin que el resto del sistema cambie.
"""

from functools import cache

from app.schemas.estimation import ReferenceEstimation, ReferenceTask

# Tarifas internas por perfil, en EUR/hora. Son la fuente de verdad para calcular costes.
HOURLY_RATES_EUR: dict[str, int] = {
    "Project Manager": 60,
    "Diseñador UX/UI": 45,
    "Desarrollador Backend": 55,
    "Desarrollador Frontend": 50,
    "Desarrollador Mobile": 55,
    "QA": 40,
    "DevOps": 60,
}

# Ordenados de mayor a menor relevancia general ("lost in the middle": lo más útil primero).
# Tipologías distintas a propósito para no sesgar al modelo hacia un único tipo de proyecto.
ESTIMATION_EXAMPLES: list[dict] = [
    {
        "project": "Tienda online B2C para marca de cosmética natural",
        "project_type": "E-commerce web",
        "meeting_summary": (
            "La marca vende hoy por Instagram y WhatsApp y quiere una tienda propia. Necesita "
            "catálogo con unas 150 referencias y variantes, carrito, pago con tarjeta y Bizum, "
            "cupones de descuento, área de cliente con historial de pedidos y un panel para "
            "gestionar stock. Los envíos se gestionan con una empresa de mensajería que ofrece "
            "API. Tienen la identidad visual definida pero no diseños de la web."
        ),
        "tasks": [
            {"task": "Discovery, alcance y planificación", "role": "Project Manager", "hours": 20},
            {
                "task": "Diseño UX/UI (catálogo, checkout, área cliente)",
                "role": "Diseñador UX/UI",
                "hours": 60,
            },
            {
                "task": "Catálogo, variantes y buscador",
                "role": "Desarrollador Backend",
                "hours": 50,
            },
            {
                "task": "Carrito, checkout y pasarela de pago (tarjeta + Bizum)",
                "role": "Desarrollador Backend",
                "hours": 60,
            },
            {
                "task": "Integración con API de mensajería",
                "role": "Desarrollador Backend",
                "hours": 30,
            },
            {
                "task": "Panel de administración (stock, pedidos, cupones)",
                "role": "Desarrollador Frontend",
                "hours": 55,
            },
            {"task": "Frontend tienda responsive", "role": "Desarrollador Frontend", "hours": 80},
            {"task": "Testing funcional y de pagos", "role": "QA", "hours": 40},
            {
                "task": "Infraestructura, CI/CD y puesta en producción",
                "role": "DevOps",
                "hours": 20,
            },
            {"task": "Gestión del proyecto y seguimiento", "role": "Project Manager", "hours": 30},
        ],
        "team": "1 PM (parcial), 1 diseñador UX/UI, 1 backend, 1 frontend, 1 QA (parcial), 1 DevOps (puntual)",
        "duration": "10-12 semanas",
        "risks": [
            "La pasarela de pago requiere alta del comercio en el banco (plazo externo).",
            "El catálogo y las fotos los aporta el cliente; retrasos en el contenido retrasan la salida.",
        ],
    },
    {
        "project": "App móvil de reservas para cadena de 6 centros de fisioterapia",
        "project_type": "App móvil + backend",
        "meeting_summary": (
            "El cliente gestiona las citas por teléfono y quiere una app iOS/Android para que los "
            "pacientes reserven, cancelen y reciban recordatorios push. Cada centro tiene varios "
            "fisioterapeutas con agenda propia. Necesitan un panel web sencillo para recepción y "
            "pago opcional de bonos de sesiones. No hay sistema previo que integrar."
        ),
        "tasks": [
            {"task": "Discovery, alcance y planificación", "role": "Project Manager", "hours": 15},
            {"task": "Diseño UX/UI app y panel", "role": "Diseñador UX/UI", "hours": 50},
            {
                "task": "API de agendas, reservas y bonos",
                "role": "Desarrollador Backend",
                "hours": 70,
            },
            {
                "task": "Notificaciones push y recordatorios",
                "role": "Desarrollador Backend",
                "hours": 20,
            },
            {
                "task": "App móvil multiplataforma (Flutter)",
                "role": "Desarrollador Mobile",
                "hours": 120,
            },
            {"task": "Panel web de recepción", "role": "Desarrollador Frontend", "hours": 40},
            {"task": "Testing en dispositivos y QA", "role": "QA", "hours": 35},
            {
                "task": "Publicación en stores, infraestructura y CI/CD",
                "role": "DevOps",
                "hours": 20,
            },
            {"task": "Gestión del proyecto y seguimiento", "role": "Project Manager", "hours": 25},
        ],
        "team": "1 PM (parcial), 1 diseñador UX/UI, 1 backend, 1 mobile, 1 frontend (parcial), 1 QA (parcial)",
        "duration": "12-14 semanas",
        "risks": [
            "La revisión de Apple/Google puede añadir 1-2 semanas a la publicación.",
            "Las reglas de reserva por centro (huecos, cancelaciones) no están cerradas.",
        ],
    },
    {
        "project": "Integración ERP-CRM y dashboard comercial para distribuidora",
        "project_type": "Integración + herramienta interna",
        "meeting_summary": (
            "La distribuidora usa un ERP on-premise para facturación y HubSpot como CRM, y el "
            "equipo comercial copia datos a mano entre ambos. Quieren sincronizar clientes y "
            "pedidos cada hora y un dashboard interno con ventas por comercial y zona. El ERP "
            "expone una API SOAP antigua y poco documentada."
        ),
        "tasks": [
            {
                "task": "Análisis de la API del ERP y mapeo de datos",
                "role": "Desarrollador Backend",
                "hours": 25,
            },
            {
                "task": "Servicio de sincronización ERP ↔ HubSpot",
                "role": "Desarrollador Backend",
                "hours": 60,
            },
            {
                "task": "Gestión de errores, reintentos y logs de sincronización",
                "role": "Desarrollador Backend",
                "hours": 20,
            },
            {
                "task": "Dashboard comercial (ventas por comercial y zona)",
                "role": "Desarrollador Frontend",
                "hours": 45,
            },
            {
                "task": "Testing de integración con datos reales anonimizados",
                "role": "QA",
                "hours": 25,
            },
            {
                "task": "Despliegue, programación de tareas y monitorización",
                "role": "DevOps",
                "hours": 15,
            },
            {"task": "Gestión del proyecto y seguimiento", "role": "Project Manager", "hours": 20},
        ],
        "team": "1 PM (parcial), 1 backend, 1 frontend (parcial), 1 QA (parcial), 1 DevOps (puntual)",
        "duration": "6-8 semanas",
        "risks": [
            "La documentación de la API SOAP del ERP es incompleta: puede requerir ingeniería inversa.",
            "Calidad de los datos de origen (duplicados de clientes entre ERP y CRM).",
        ],
    },
]


def build_reference_estimation(example: dict, rates: dict[str, int]) -> ReferenceEstimation:
    """Calcula el coste de cada tarea (horas × tarifa del perfil) y los totales del ejemplo."""
    tasks = [
        ReferenceTask(
            task=task["task"],
            role=task["role"],
            hours=task["hours"],
            rate_eur=rates[task["role"]],
            cost_eur=task["hours"] * rates[task["role"]],
        )
        for task in example["tasks"]
    ]
    return ReferenceEstimation(
        project=example["project"],
        project_type=example["project_type"],
        meeting_summary=example["meeting_summary"],
        tasks=tasks,
        total_hours=sum(task.hours for task in tasks),
        total_cost_eur=sum(task.cost_eur for task in tasks),
        team=example["team"],
        duration=example["duration"],
        risks=example["risks"],
    )


@cache
def reference_estimations() -> tuple[ReferenceEstimation, ...]:
    """Las estimaciones de referencia listas para inyectar (se calculan una sola vez)."""
    return tuple(build_reference_estimation(e, HOURLY_RATES_EUR) for e in ESTIMATION_EXAMPLES)
