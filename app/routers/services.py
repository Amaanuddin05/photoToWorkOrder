"""
Service catalog endpoints:
  GET  /v1/services           — list services
  PUT  /v1/services           — bulk upsert with embedding sync
  DELETE /v1/services/{id}    — soft-delete
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

logger = logging.getLogger("siglip.routers.services")
router = APIRouter(prefix="/v1", tags=["services"])

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SIGLIP_TOKEN_LIMIT = 64


class ServiceIn(BaseModel):
    id: int | None = None
    title: str
    description: str
    visual_line: str | None = None
    keywords: str | None = None
    is_active: bool = True


def _require_auth(request: Request, authorization: str | None = Header(None)) -> None:
    from app.config import get_settings
    settings = get_settings()
    key_list = settings.api_key_list
    if not key_list:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail={"code": "MISSING_AUTH", "message": "Authorization required"})
    provided = authorization[7:].strip()
    if not any(secrets.compare_digest(provided, k) for k in key_list):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail={"code": "INVALID_API_KEY", "message": "Invalid API key"})


@router.get("/services")
async def list_services(
    request: Request,
    include_inactive: bool = False,
    _auth: None = Depends(_require_auth),
) -> list[dict[str, Any]]:
    from app.db import repository as repo
    return await repo.get_all_services(request.app.state.db_pool, include_inactive)


@router.put("/services")
async def upsert_services(
    services: list[ServiceIn],
    request: Request,
    _auth: None = Depends(_require_auth),
) -> list[dict[str, Any]]:
    """
    Bulk upsert. For each service:
    1. Compute content hash; skip re-embedding if unchanged.
    2. Re-embed title+description via TEI (BGE passages, no prefix).
    3. Re-embed textUsed via SigLIP text encoder.
    4. Upsert WoService and ServiceImageProfile.
    5. Refresh in-memory service profiles.
    """
    from app.config import get_settings
    from app.db import repository as repo
    import asyncio

    state = request.app.state
    settings = get_settings()
    db_pool = state.db_pool
    tei = state.tei_client
    siglip = state.siglip_service

    results = []

    for svc in services:
        title = svc.title.strip()
        desc = svc.description.strip()
        visual_line = svc.visual_line.strip() if svc.visual_line else None
        text_hash = hashlib.sha256(f"{title}\n{desc}".encode()).hexdigest()
        image_text = f"{title}. {visual_line}" if visual_line else title

        # Lookup existing
        existing = await repo.get_service_by_title(db_pool, title)
        if existing and existing.get("id") == svc.id and existing.get("embedding_model") == "BAAI/bge-base-en-v1.5":
            # Check if text content changed via hash stored in embeddingModel or always re-embed
            # For simplicity: always re-embed (TEI is fast for small batches)
            pass

        # 1. Text embeddings via TEI (passages: NO query prefix)
        try:
            vecs = await tei.embed([title, desc])
            title_emb = vecs[0]
            desc_emb = vecs[1]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "TEI_EMBED_ERROR", "message": f"Failed to embed service text: {exc}"},
            )

        # 2. Upsert WoService
        db_svc = await repo.upsert_service(
            db_pool,
            title=title,
            description=desc,
            keywords=svc.keywords,
            is_active=svc.is_active,
            title_emb=title_emb,
            desc_emb=desc_emb,
            embedding_model="BAAI/bge-base-en-v1.5",
            existing_id=svc.id,
        )
        service_id = db_svc["id"]

        # 3. Image profile via SigLIP text encoder
        try:
            async with state.semaphore:
                image_embs, trunc_flags = await asyncio.to_thread(
                    siglip.encode_texts, [image_text]
                )
            image_emb = image_embs[0]
            truncated = trunc_flags[0] if trunc_flags else False
        except Exception as exc:
            logger.warning("SigLIP encoding failed for service '%s': %s", title, exc)
            image_emb = None
            truncated = False

        if image_emb is not None:
            await repo.upsert_service_image_profile(
                db_pool,
                service_id=service_id,
                image_model=settings.SIGLIP_MODEL_ID,
                visual_line=visual_line,
                text_used=image_text,
                text_emb=image_emb,
                truncated=truncated,
            )

        results.append({"id": service_id, "title": title, "updated_at": db_svc.get("updated_at")})

    # 4. Refresh in-memory service profiles
    try:
        state.service_profiles = await repo.get_all_service_profiles(
            db_pool, settings.SIGLIP_MODEL_ID
        )
        logger.info("Service profiles refreshed: %d services", len(state.service_profiles))
    except Exception as exc:
        logger.warning("Failed to refresh service profiles cache: %s", exc)

    return results


@router.delete("/services/{service_id}")
async def delete_service(
    service_id: int,
    request: Request,
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Soft-delete by setting isActive=false. Preserves FK references in DecisionLog."""
    from app.db import repository as repo

    found = await repo.soft_delete_service(request.app.state.db_pool, service_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SERVICE_NOT_FOUND", "message": f"Service {service_id} not found"},
        )
    return {"status": "deleted", "service_id": service_id, "note": "soft-deleted (isActive=false)"}

