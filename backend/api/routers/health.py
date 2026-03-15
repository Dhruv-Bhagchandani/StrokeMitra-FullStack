"""Health check endpoints."""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends

from api.schemas import HealthResponse, ReadinessResponse
from api.dependencies import get_pipeline
from src.pipeline import SlurringDetectionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Basic liveness check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
    )


@router.get("/readyz", response_model=ReadinessResponse)
async def readiness_check(pipeline: SlurringDetectionPipeline = Depends(get_pipeline)):
    """Readiness check - verifies pipeline is loaded."""
    checks = {
        "pipeline_loaded": pipeline is not None,
        "model_loaded": pipeline.ensemble_model is not None,
    }

    ready = all(checks.values())

    return ReadinessResponse(
        ready=ready,
        checks=checks,
        timestamp=datetime.utcnow(),
    )
