# AI Engineering 2026/09 — LIDR

Proyecto práctico del programa AI Engineering. Un único sistema que evoluciona sesión a sesión:
un **estimador automático de proyectos de software** que recibe transcripciones de reuniones con
clientes y genera presupuestos basados en el histórico de la empresa.

| Fase | Arquitectura | Estado |
| --- | --- | --- |
| Módulo 2 · Sesiones 2-5 | CAG: contexto estático inyectado en el prompt; desde la sesión 5, conversación con memoria y adjuntos | ✅ [`estimador-cag/`](estimador-cag/) |
| Módulos 3-4 | RAG con búsqueda semántica | Pendiente |
| Módulo 5 | Orquestación de agentes | Pendiente |
| Módulo 6 | Producción | Pendiente |

Cada sesión se entrega en su propia rama (`sesion-N/...`) y se integra en `main` una vez revisada.
