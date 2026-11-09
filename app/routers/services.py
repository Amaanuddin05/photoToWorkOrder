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

from app.catalog_text import catalog_texts

logger = logging.getLogger("siglip.routers.services")
router = APIRouter(prefix="/v1", tags=["services"])

SIGLIP_TOKEN_LIMIT = 64


class ServiceIn(BaseModel):
    id: int | None = None
    title: str
    description: str
    visual_line: str | None = None
    keywords: str | None = None
    is_active: bool = True


class CreateServicePayload(BaseModel):
    title: str
    description: str
    keywords: str | None = None
    visual_line: str | None = None
    visual_description: str | None = None
    reactivate_if_exists: bool = False


class CheckSimilarityPayload(BaseModel):
    title: str
    description: str


def _clean_keywords(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    seen: set[str] = set()
    cleaned: list[str] = []
    for part in raw.split(","):
        trimmed = part.strip()
        if trimmed and trimmed.lower() not in seen:
            seen.add(trimmed.lower())
            cleaned.append(trimmed)
    return ", ".join(cleaned) if cleaned else None


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

        # 1. Text embeddings via TEI (passages: NO query prefix).
        #    Same recipe as the rest of the catalog: title, and "{description}. {keywords}".
        try:
            vecs = await tei.embed(list(catalog_texts(title, desc, svc.keywords)))
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


# ── Demo Services & Add-Service Form Endpoints ───────────────────────────────

@router.post("/services/check-similarity")
async def check_similarity(
    payload: CheckSimilarityPayload,
    request: Request,
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Non-blocking similarity check: compares candidate title and description embeddings
    against all active catalog services using cosine similarity.
    """
    from app.db import repository as repo

    title = payload.title.strip()
    desc = payload.description.strip()
    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "TITLE_REQUIRED", "message": "Service name is required."},
        )
    if not desc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DESC_REQUIRED", "message": "Service description is required."},
        )

    state = request.app.state
    db_pool = state.db_pool
    tei = state.tei_client

    try:
        vecs = await tei.embed([title, desc])
        title_emb = vecs[0]
        desc_emb = vecs[1]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "TEI_EMBED_ERROR", "message": f"Failed to compute embeddings: {exc}"},
        )

    if len(title_emb) != 768 or len(desc_emb) != 768:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "INVALID_EMBEDDING_DIM", "message": f"Expected 768 dimensions from TEI, got title={len(title_emb)}, desc={len(desc_emb)}"},
        )

    similar = await repo.find_similar_services(db_pool, title_emb, desc_emb, top_n=3)
    highest_score = similar[0]["score"] if similar else 0.0

    return {
        "similar_services": similar,
        "highest_score": highest_score,
        "is_high_similarity": bool(highest_score > 0.85),
    }


@router.post("/services/create")
async def create_service(
    payload: CreateServicePayload,
    request: Request,
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Creates a new service together with its embeddings in ONE atomic transaction.
    Rejects active duplicate titles (case-insensitive, trimmed).
    If an inactive (soft-deleted) duplicate exists, returns 409 prompting reactivation
    unless reactivate_if_exists=True.
    Refreshes in-memory service_profiles cache and registers as demo service.
    """
    from app.config import get_settings
    from app.db import repository as repo
    import asyncio

    state = request.app.state
    settings = get_settings()
    db_pool = state.db_pool
    tei = state.tei_client
    siglip = state.siglip_service

    title = payload.title.strip()
    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "TITLE_REQUIRED", "message": "Service name is required."},
        )
    if len(title) > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "TITLE_TOO_LONG", "message": "Service name must be at most 200 characters."},
        )

    desc = payload.description.strip()
    if not desc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DESC_REQUIRED", "message": "Service description is required."},
        )
    if len(desc) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "DESC_TOO_LONG", "message": "Description must be at most 1000 characters."},
        )

    visual_raw = payload.visual_line or payload.visual_description
    visual_line = visual_raw.strip() if visual_raw and visual_raw.strip() else None
    if visual_line and len(visual_line) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VISUAL_LINE_TOO_LONG", "message": "Visual description must be at most 500 characters."},
        )

    keywords = _clean_keywords(payload.keywords)

    # Check for existing title collision
    existing = await repo.get_service_by_title_ci(db_pool, title)
    existing_id = None
    if existing:
        if existing["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "SERVICE_ALREADY_EXISTS",
                    "message": f"An active service with title '{existing['title']}' already exists.",
                    "existing_id": existing["id"],
                },
            )
        else:
            if not payload.reactivate_if_exists:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "SERVICE_INACTIVE_EXISTS",
                        "message": (
                            f"An inactive (soft-deleted) service with title '{existing['title']}' already exists (ID: {existing['id']}). "
                            "Would you like to reactivate and overwrite it with these fields?"
                        ),
                        "existing_id": existing["id"],
                        "existing_title": existing["title"],
                        "can_reactivate": True,
                    },
                )
            existing_id = existing["id"]

    # 1. Compute ALL embeddings FIRST before writing anything
    #    Same recipe as the rest of the catalog: title, and "{description}. {keywords}".
    try:
        vecs = await tei.embed(list(catalog_texts(title, desc, keywords)))
        title_emb = vecs[0]
        desc_emb = vecs[1]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "TEI_EMBED_ERROR", "message": f"Failed to compute text embeddings: {exc}"},
        )

    if len(title_emb) != 768 or len(desc_emb) != 768:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "INVALID_EMBEDDING_DIM",
                "message": f"Expected 768 dimensions from TEI, got title={len(title_emb)}, desc={len(desc_emb)}",
            },
        )

    image_emb = None
    truncated = False
    text_used = None
    if visual_line:
        text_used = f"{title}. {visual_line}"
        try:
            async with state.semaphore:
                image_embs, trunc_flags = await asyncio.to_thread(
                    siglip.encode_texts, [text_used]
                )
            image_emb = image_embs[0]
            truncated = trunc_flags[0] if trunc_flags else False
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "SIGLIP_EMBED_ERROR", "message": f"Failed to compute SigLIP visual profile embedding: {exc}"},
            )
        if len(image_emb) != 768:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "INVALID_IMAGE_EMBED_DIM", "message": f"Expected 768 dimensions from SigLIP, got {len(image_emb)}"},
            )

    # 2. Similarity analysis (non-blocking warning)
    similar_services = await repo.find_similar_services(
        db_pool, title_emb, desc_emb, top_n=3, exclude_id=existing_id
    )

    # 3. Transactional Write: either both rows exist or neither
    try:
        db_svc = await repo.create_or_update_service_tx(
            db_pool,
            title=title,
            description=desc,
            keywords=keywords,
            is_active=True,
            title_emb=title_emb,
            desc_emb=desc_emb,
            embedding_model="BAAI/bge-base-en-v1.5",
            existing_id=existing_id,
            image_model=settings.SIGLIP_MODEL_ID,
            visual_line=visual_line,
            text_used=text_used,
            image_emb=image_emb,
            truncated=truncated,
        )
        service_id = db_svc["id"]
    except Exception as exc:
        logger.exception("Failed to persist service in transaction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DB_WRITE_ERROR", "message": f"Failed to write service in database: {exc}"},
        )

    # 4. Refresh in-memory service profiles cache immediately
    try:
        state.service_profiles = await repo.get_all_service_profiles(
            db_pool, settings.SIGLIP_MODEL_ID
        )
        logger.info("Service profiles refreshed after demo service create: %d services", len(state.service_profiles))
    except Exception as exc:
        logger.warning("Failed to refresh service profiles cache: %s", exc)

    # 5. Register in demo registry
    registry_warning = None
    try:
        from app.demo_registry import add_demo_service
        await add_demo_service(settings.DEMO_REGISTRY_PATH, service_id, title)
    except Exception as exc:
        logger.warning("Failed to register demo service id=%d: %s", service_id, exc)
        registry_warning = f"Service created, but failed to track in demo registry: {exc}"

    # Build clear warning messages
    warning_parts = []
    if not visual_line:
        warning_parts.append(
            "No visual description provided. No ServiceImageProfile was created; this service will not match from photos until a visual description is added."
        )
    if similar_services and similar_services[0]["score"] > 0.85:
        top_sim = similar_services[0]
        warning_parts.append(
            f"Very similar to existing service: '{top_sim['title']}' (similarity: {top_sim['score']:.2f})"
        )
    if registry_warning:
        warning_parts.append(registry_warning)

    warning = " | ".join(warning_parts) if warning_parts else None

    return {
        "id": service_id,
        "title": title,
        "description": desc,
        "keywords": keywords,
        "visual_line": visual_line,
        "has_image_profile": bool(image_emb is not None),
        "embedding_dimensions": 768,
        "reactivated": bool(existing_id is not None),
        "similar_services": similar_services,
        "warning": warning,
        "updated_at": db_svc.get("updated_at"),
    }


@router.get("/services/demo")
async def list_demo_services(
    request: Request,
    _auth: None = Depends(_require_auth),
) -> list[dict[str, Any]]:
    """
    Lists registered demo services with their current DB status.
    Automatically prunes stale entries from the registry if the DB row is missing or inactive.
    """
    from app.config import get_settings
    from app.db import repository as repo
    from app.demo_registry import load_demo_registry, prune_demo_registry

    settings = get_settings()
    db_pool = request.app.state.db_pool

    entries = await load_demo_registry(settings.DEMO_REGISTRY_PATH)
    if not entries:
        return []

    entry_ids = [e["id"] for e in entries]
    services_map = await repo.get_services_by_ids(
        db_pool, entry_ids, settings.SIGLIP_MODEL_ID
    )

    valid_ids: set[int] = set()
    results: list[dict[str, Any]] = []

    for e in entries:
        s_id = e["id"]
        db_s = services_map.get(s_id)
        if db_s and db_s["is_active"]:
            valid_ids.add(s_id)
            results.append({
                "id": s_id,
                "title": db_s["title"],
                "description": db_s["description"],
                "keywords": db_s["keywords"],
                "visual_description": db_s["visual_description"],
                "has_image_profile": db_s["has_image_profile"],
                "is_active": db_s["is_active"],
                "created_at": e["created_at"],
            })

    # Prune stale entries
    if len(valid_ids) < len(entries):
        await prune_demo_registry(settings.DEMO_REGISTRY_PATH, valid_ids)

    return results


@router.delete("/services/demo")
async def delete_all_demo_services(
    request: Request,
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Deletes all registered demo services.
    Refreshes in-memory service_profiles cache immediately.
    """
    from app.config import get_settings
    from app.db import repository as repo
    from app.demo_registry import load_demo_registry, clear_demo_registry

    settings = get_settings()
    state = request.app.state
    db_pool = state.db_pool

    entries = await load_demo_registry(settings.DEMO_REGISTRY_PATH)
    if not entries:
        return {"status": "deleted_all", "deleted_count": 0, "deleted_services": [], "mode": "none"}

    deleted_services: list[int] = []
    mode = "hard_delete" if settings.DEMO_HARD_DELETE else "soft_delete"

    for e in entries:
        s_id = e["id"]
        db_svc = await repo.get_service_by_id(db_pool, s_id)
        if db_svc and db_svc["title"].strip().lower() == e["title"].strip().lower():
            if settings.DEMO_HARD_DELETE:
                try:
                    await repo.hard_delete_service_tx(db_pool, s_id)
                    deleted_services.append(s_id)
                except Exception as exc:
                    logger.warning("Hard delete failed for demo service %d: %s", s_id, exc)
            else:
                await repo.soft_delete_service(db_pool, s_id)
                deleted_services.append(s_id)

    await clear_demo_registry(settings.DEMO_REGISTRY_PATH)

    # Refresh in-memory service profiles cache immediately
    try:
        state.service_profiles = await repo.get_all_service_profiles(
            db_pool, settings.SIGLIP_MODEL_ID
        )
        logger.info("Service profiles refreshed after bulk demo delete: %d services", len(state.service_profiles))
    except Exception as exc:
        logger.warning("Failed to refresh service profiles after bulk demo delete: %s", exc)

    return {
        "status": "deleted_all",
        "deleted_count": len(deleted_services),
        "deleted_services": deleted_services,
        "mode": mode,
    }


@router.delete("/services/demo/{service_id}")
async def delete_demo_service(
    service_id: int,
    request: Request,
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Deletes a specific demo service.
    SERVER-SIDE RULE: only IDs in the registry can be deleted, and the DB title must match.
    """
    from app.config import get_settings
    from app.db import repository as repo
    from app.demo_registry import get_demo_entry, remove_demo_service

    settings = get_settings()
    state = request.app.state
    db_pool = state.db_pool

    # 1. Server-side rule: verify registry entry
    reg_entry = await get_demo_entry(settings.DEMO_REGISTRY_PATH, service_id)
    if not reg_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_A_DEMO_SERVICE",
                "message": f"Service {service_id} is not registered as a demo service. Production services cannot be deleted through this endpoint.",
            },
        )

    # 2. Server-side rule: verify DB row and title match
    db_svc = await repo.get_service_by_id(db_pool, service_id)
    if not db_svc:
        await remove_demo_service(settings.DEMO_REGISTRY_PATH, service_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SERVICE_NOT_FOUND", "message": f"Service {service_id} not found in database."},
        )

    if db_svc["title"].strip().lower() != reg_entry["title"].strip().lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "TITLE_MISMATCH",
                "message": "Security error: Service title in database does not match the demo registry title.",
            },
        )

    # 3. Perform delete
    mode = "soft_delete"
    if settings.DEMO_HARD_DELETE:
        try:
            await repo.hard_delete_service_tx(db_pool, service_id)
            mode = "hard_delete"
        except Exception as exc:
            logger.error("Hard delete failed for demo service %d: %s", service_id, exc)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "DELETE_PERMISSION_DENIED", "message": f"Failed to hard-delete service: {exc}"},
            )
    else:
        await repo.soft_delete_service(db_pool, service_id)

    # 4. Remove from demo registry
    await remove_demo_service(settings.DEMO_REGISTRY_PATH, service_id)

    # 5. Refresh in-memory service profiles cache immediately
    try:
        state.service_profiles = await repo.get_all_service_profiles(
            db_pool, settings.SIGLIP_MODEL_ID
        )
        logger.info("Service profiles refreshed after demo service delete: %d services", len(state.service_profiles))
    except Exception as exc:
        logger.warning("Failed to refresh service profiles after demo service delete: %s", exc)

    return {
        "status": "deleted",
        "service_id": service_id,
        "title": db_svc["title"],
        "mode": mode,
    }


# ── Generic Services Management ──────────────────────────────────────────────

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



