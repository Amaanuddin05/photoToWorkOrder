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
    Reports: database, schema, siglip_model, tei_embed, tei_rerank, text_embedding.
    text_embedding is not-ready while the catalog is not embedded for the model
    TEI serves (a background re-embed is started automatically).
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
    settings = getattr(state, "settings", None)
    reranker_enabled = getattr(settings, "ENABLE_RERANKER", True) if settings else True

    tei = getattr(state, "tei_client", None)
    if tei:
        dep_status = await tei.check_health()
        embed_ok = dep_status.get("tei_embed", {}).get("reachable", False)
        rerank_ok = dep_status.get("tei_rerank", {}).get("reachable", False)
        checks["tei_embed"] = dep_status.get("tei_embed", {})
        rerank_info = dict(dep_status.get("tei_rerank", {}))
        if not reranker_enabled:
            rerank_info["skipped"] = True
            rerank_info["enabled"] = False
        checks["tei_rerank"] = rerank_info
        if not embed_ok or (reranker_enabled and not rerank_ok):
            all_ok = False
    else:
        checks["tei_embed"] = {"reachable": False, "error": "client not initialised"}
        checks["tei_rerank"] = {
            "reachable": False,
            "error": "client not initialised",
            "skipped": not reranker_enabled,
            "enabled": reranker_enabled,
        }
        all_ok = False

    # Text-embedding guard: the catalog must be embedded (ServiceTextEmbedding)
    # for the model TEI serves right now. Checked live on every readiness call.
    # If TEI's model changed (or services were added/edited elsewhere), one
    # background re-embed is started and the service is not-ready until it ends.
    db_pool = getattr(state, "db_pool", None)
    if tei and db_pool is not None and settings is not None:
        from app.text_embedding import STORE_TABLE, check_text_embedding, start_background_sync
        st = await check_text_embedding(db_pool, tei, settings)
        sync_task = getattr(state, "text_sync_task", None)
        if not st.ok:
            pin_violated = bool(st.pinned_model_id and st.tei_model_id and st.tei_model_id != st.pinned_model_id)
            store_missing = any(STORE_TABLE in p and "does not exist" in p for p in st.problems)
            if st.tei_model_id and not pin_violated and not store_missing:
                start_background_sync(state)
            sync_task = getattr(state, "text_sync_task", None)
        st.sync_in_progress = bool(sync_task is not None and not sync_task.done())
        st.last_sync = getattr(state, "text_sync_result", None)
        if st.sync_in_progress:
            st.message = f"Re-embedding catalog for '{st.tei_model_id}' in the background. " + st.message
        state.text_embed_status = st
        checks["text_embedding"] = st.to_dict()
        if not st.ok:
            all_ok = False
    else:
        checks["text_embedding"] = {"ok": False, "message": "TEI client or database pool not initialised"}
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

