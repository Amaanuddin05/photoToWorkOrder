"""
Prometheus metrics endpoint.
Unauthenticated by default; set METRICS_API_KEY to require auth.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger("siglip.routers.metrics")
router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus metrics scrape endpoint."""
    from app.config import get_settings

    settings = get_settings()
    metrics_key = settings.METRICS_API_KEY

    if metrics_key:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        provided = auth_header[7:].strip()
        if not secrets.compare_digest(provided, metrics_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

