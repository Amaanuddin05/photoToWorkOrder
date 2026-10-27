"""
Health endpoints: /health/live (liveness) and /health/ready (readiness).
Both are unauthenticated.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("siglip.routers.health")
router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """K8s/Coolify liveness probe. Always returns 200 if the process is alive."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """
    Readiness probe. Returns 200 only when all components are healthy.
    Returns 503 with per-component details otherwise.
    Reports: database, schema, siglip_model, tei_embed, tei_rerank.
    """
    state = request.app.state
    checks: dict[str, Any] = {}
    all_ok = True

    # Database connectivity
    try:
        db_pool = getattr(state, "db_pool", None)
        if db_pool is None:
            raise RuntimeError("Pool not initialised")
        async with db_pool.connection() as conn:
            await conn.execute("SELECT 1;")
        schema_info = getattr(state, "schema_info", {})
        checks["database"] = {
            "connected": True,
            "pgvector_version": schema_info.get("pgvector_version"),
            "schema_verified": True,
        }
    except Exception as exc:
        checks["database"] = {"connected": False, "error": str(exc)}
        all_ok = False

    # SigLIP model
    siglip = getattr(state, "siglip_service", None)
    if siglip and getattr(siglip, "_is_loaded", False):
        checks["siglip_model"] = {
            "loaded": True,
            "model_id": siglip.model_id,
            "dim": siglip.embedding_dim,
        }
    else:
        checks["siglip_model"] = {"loaded": False}
        all_ok = False

    # TEI dependencies
    tei = getattr(state, "tei_client", None)
    if tei:
        dep_status = await tei.check_health()
        embed_ok = dep_status.get("tei_embed", {}).get("reachable", False)
        rerank_ok = dep_status.get("tei_rerank", {}).get("reachable", False)
        checks["tei_embed"] = dep_status.get("tei_embed", {})
        checks["tei_rerank"] = dep_status.get("tei_rerank", {})
        if not embed_ok or not rerank_ok:
            all_ok = False
    else:
        checks["tei_embed"] = {"reachable": False, "error": "client not initialised"}
        checks["tei_rerank"] = {"reachable": False, "error": "client not initialised"}
        all_ok = False

    # Service profiles loaded
    profiles = getattr(state, "service_profiles", None)
    checks["service_profiles"] = {
        "loaded": profiles is not None,
        "count": len(profiles) if profiles else 0,
    }

    status = "ready" if all_ok else "not_ready"
    http_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=http_code,
        content={"status": status, "checks": checks},
    )

