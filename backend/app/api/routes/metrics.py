"""Metrics endpoint for monitoring."""
from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Get Prometheus-compatible metrics.

    Returns:
        Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

