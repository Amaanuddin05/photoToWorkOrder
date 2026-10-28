"""
POST /v1/posts/process — multipart endpoint for work order triage.
Handles idempotency, image validation, and delegates to pipeline.engine.
"""

from __future__ import annotations

import io
import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.pipeline.decision import CandidateOut, ProcessedPost

logger = logging.getLogger("siglip.routers.posts")
router = APIRouter(prefix="/v1", tags=["posts"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# Decompression-bomb protection (roughly 178 megapixels)
Image.MAX_IMAGE_PIXELS = 178_956_970


def _require_auth(request: Request, authorization: str | None = Header(None)) -> None:
    """Constant-time Bearer token verification. Skipped if API_KEYS is empty."""
    from app.config import get_settings

    settings = get_settings()
    key_list = settings.api_key_list
    if not key_list:
        return  # No auth configured (dev mode)

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_AUTH", "message": "Authorization header required"},
        )
    provided = authorization[7:].strip()
    # Constant-time comparison against all configured keys
    if not any(secrets.compare_digest(provided, k) for k in key_list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_API_KEY", "message": "Invalid API key"},
        )


def _build_response(result: ProcessedPost, idempotent_replay: bool = False) -> dict[str, Any]:
    candidates = [
        {
            "service_id": c.service_id,
            "title": c.title,
            "confidence": c.confidence,
            "confidence_pct": c.confidence_pct,
            "text_rank": c.text_rank,
            "image_rank": c.image_rank,
            "pool_rank": c.pool_rank,
            "rerank_score": c.rerank_score,
        }
        for c in result.top_candidates
    ]
    return {
        "decision_id": result.decision_id,
        "action": result.action,
        "would_be_action": result.would_be_action,
        "would_auto": result.would_be_action == "auto_create" and not result.action == "auto_create",
        "confidence": result.confidence,
        "gap": result.gap,
        "agree": result.agree,
        "reason": result.reason,
        "top_candidates": candidates,
        "idempotent_replay": idempotent_replay,
        "shadow": result.shadow,
    }


@router.post("/posts/process")
async def process_post(
    request: Request,
    image: UploadFile | None = File(None),
    description: str | None = Form(None),
    post_id: str | None = Form(None),
    location: str | None = Form(None),
    idempotency_key_header: str | None = Header(None, alias="Idempotency-Key"),
    _auth: None = Depends(_require_auth),
) -> JSONResponse:
    """
    Triage a maintenance post into a work order.
    Accepts multipart/form-data with optional image and/or description.
    At least one must be provided.
    """
    from app.config import get_settings
    from app.db import repository as repo
    from app.pipeline import engine

    settings = get_settings()
    state = request.app.state

    # Idempotency key: header takes precedence over post_id form field
    idem_key = idempotency_key_header or post_id

    # Fast idempotency check before reading image bytes
    if idem_key:
        db_pool = state.db_pool
        existing = await repo.get_decision_log_by_idempotency_key(db_pool, idem_key)
        if existing:
            # Reconstruct a minimal ProcessedPost from stored data
            reranked = existing.get("rerank_scores") or []
            top_candidates = []
            if reranked and isinstance(reranked, list):
                for item in reranked[:3]:
                    conf = item.get("confidence")
                    top_candidates.append(CandidateOut(
                        service_id=item.get("service_id"),
                        title=item.get("service", ""),
                        confidence=conf,
                        confidence_pct=f"{conf * 100:.1f}%" if conf is not None else None,
                        text_rank=item.get("text_rank"),
                        image_rank=item.get("image_rank"),
                        pool_rank=item.get("pool_rank"),
                        rerank_score=item.get("rerank_score"),
                    ))

            fake_result = ProcessedPost(
                decision_id=existing["id"],
                action=existing["action"],
                would_be_action=existing["would_be_action"],
                confidence=existing["confidence"],
                gap=existing["gap"],
                agree=existing["agree"],
                reason=existing.get("reason", ""),
                top_candidates=top_candidates,
                idempotent_replay=True,
                shadow=existing["would_be_action"] == "auto_create" and existing["action"] != "auto_create",
            )
            return JSONResponse(content=_build_response(fake_result, idempotent_replay=True))

    # Read and validate image
    image_bytes: bytes | None = None
    if image is not None:
        raw = await image.read()
        if raw:
            if len(raw) > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "code": "IMAGE_TOO_LARGE",
                        "message": f"Image exceeds {settings.MAX_UPLOAD_BYTES} bytes",
                    },
                )
            # Validate image format and decompression-bomb protection
            try:
                bio = io.BytesIO(raw)
                pil = Image.open(bio)
                pil.load()  # Force decode
                if pil.format not in ("JPEG", "PNG", "WEBP"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "INVALID_IMAGE_FORMAT",
                            "message": f"Unsupported format {pil.format}. Use JPEG, PNG, or WebP.",
                        },
                    )
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "INVALID_IMAGE", "message": f"Cannot decode image: {exc}"},
                ) from exc
            image_bytes = raw

    # At least one input required
    if image_bytes is None and not (description and description.strip()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INPUT_REQUIRED",
                "message": "At least one of 'image' or 'description' must be provided",
            },
        )

    # Run pipeline
    result = await engine.process_post(
        image_bytes=image_bytes,
        description=description,
        location=location,
        settings=settings,
        pool=state.db_pool,
        siglip_service=state.siglip_service,
        tei_client=state.tei_client,
        service_profiles=state.service_profiles,
        semaphore=state.semaphore,
        idempotency_key=idem_key,
    )

    # Prometheus metrics
    try:
        from app.metrics import DECISIONS_TOTAL, REQUEST_LATENCY
        action = result.action
        DECISIONS_TOTAL.labels(action=action).inc()
    except Exception:
        pass

    return JSONResponse(content=_build_response(result))

