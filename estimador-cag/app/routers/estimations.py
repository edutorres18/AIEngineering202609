"""Capa de transporte HTTP: recibe, delega en el servicio y devuelve."""

from fastapi import APIRouter

from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services import llm_service

router = APIRouter(tags=["estimations"])

ERROR_RESPONSES = {
    502: {"description": "Error del proveedor LLM"},
    503: {"description": "Rate limit del proveedor LLM"},
    504: {"description": "Timeout del proveedor LLM"},
}


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    summary="Genera una estimación de software a partir de una transcripción de reunión",
    responses=ERROR_RESPONSES,
)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    return await llm_service.generate_estimation(request.transcription)
