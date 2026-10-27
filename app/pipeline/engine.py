"""
Main async pipeline engine.
Orchestrates: text ranking (TEI /embed + pgvector) → image ranking (SigLIP)
→ pool merge → reranking (TEI /rerank raw_scores=true + local softmax) → decision.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
from typing import Any

from PIL import Image
from psycopg_pool import AsyncConnectionPool

from app.config import Settings
from app.pipeline.decision import (
    CandidateOut,
    ProcessedPost,
    apply_reranker_softmax,
    make_decision,
)
from app.pipeline.image_rank import rank_image_against_profiles
from app.pipeline.pool import build_candidate_pool

logger = logging.getLogger("siglip.pipeline.engine")

# BGE asymmetric query instruction prefix (passages are embedded WITHOUT this)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


async def process_post(
    *,
    image_bytes: bytes | None,
    description: str | None,
    location: str | None,
    settings: Settings,
    pool: AsyncConnectionPool,
    siglip_service: Any,        # SigLIPService
    tei_client: Any,            # TEIClient
    service_profiles: dict[int, dict[str, Any]],
    semaphore: asyncio.Semaphore,
    idempotency_key: str | None = None,
) -> ProcessedPost:
    """
    Full async pipeline. Calls TEI over HTTP for embed/rerank;
    runs SigLIP via asyncio.to_thread to avoid blocking the event loop.
    """
    from app.db import repository as repo

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # ── Parse image ───────────────────────────────────────────────────────────
    pil_img: Image.Image | None = None
    photo_hash: str | None = None

    if image_bytes:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        photo_hash = hashlib.sha256(image_bytes).hexdigest()

    has_image = pil_img is not None
    has_desc = bool(description and description.strip())

    # ── Stage 1: Text ranking ──────────────────────────────────────────────
    t_text = time.perf_counter()
    text_top: list[dict[str, Any]] = []

    if has_desc and description:
        prefixed = BGE_QUERY_PREFIX + description.strip()
        try:
            vectors = await tei_client.embed([prefixed])
            query_vec = vectors[0]
            text_top = await repo.text_search(
                pool, query_vec,
                settings.PIPELINE_TEXT_DESC_WEIGHT,
                settings.PIPELINE_TEXT_TOP_N,
            )
        except Exception as exc:
            logger.warning("Text ranking failed: %s", exc)

    timings["text_ms"] = round((time.perf_counter() - t_text) * 1000, 2)

    # ── Stage 2: Image ranking ─────────────────────────────────────────────
    t_img = time.perf_counter()
    image_top: list[dict[str, Any]] | None = None

    if has_image and pil_img and service_profiles:
        try:
            async with semaphore:
                image_top = await asyncio.to_thread(
                    rank_image_against_profiles,
                    pil_img,
                    service_profiles,
                    siglip_service,
                    settings.PIPELINE_IMAGE_TOP_N,
                )
        except Exception as exc:
            logger.warning("Image ranking failed: %s", exc)

    timings["image_ms"] = round((time.perf_counter() - t_img) * 1000, 2)

    # ── Stage 3: Pool formation ────────────────────────────────────────────
    t_pool = time.perf_counter()
    candidate_pool = build_candidate_pool(
        text_top=text_top,
        image_top=image_top,
        pool_max=settings.PIPELINE_POOL_MAX,
    )
    timings["pool_ms"] = round((time.perf_counter() - t_pool) * 1000, 2)

    # ── Stage 4: Reranking ─────────────────────────────────────────────────
    t_rerank = time.perf_counter()
    reranked: list[dict[str, Any]] = []
    confidence: float | None = None
    gap: float | None = None

    if has_desc and description and candidate_pool:
        candidate_texts = [
            f"{item['service']}. {item.get('description', '')}".strip()
            for item in candidate_pool
        ]
        try:
            # Request raw scores so we can apply the same softmax as photoToWorkOrder
            raw_results = await tei_client.rerank(
                description.strip(),
                candidate_texts,
                raw_scores=True,
            )
            # raw_results: [{index, score}, ...] sorted by index
            # Re-order to match candidate_pool order
            index_to_score: dict[int, float] = {r["index"]: r["score"] for r in raw_results}
            raw_scores = [index_to_score.get(i, 0.0) for i in range(len(candidate_pool))]

            reranked = apply_reranker_softmax(
                candidate_pool, raw_scores, settings.PIPELINE_RERANKER_TEMP
            )

            if reranked:
                confidence = float(reranked[0]["confidence"])
                gap = float(confidence - reranked[1]["confidence"]) if len(reranked) > 1 else confidence

        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)

    timings["rerank_ms"] = round((time.perf_counter() - t_rerank) * 1000, 2)

    # ── Stage 5: Agreement signal ──────────────────────────────────────────
    agree: bool | None = None
    if has_image and image_top:
        img_top1 = image_top[0]["service"] if image_top else None
        if reranked:
            agree = bool(img_top1 == reranked[0]["service"])
        elif text_top:
            agree = bool(img_top1 == text_top[0]["service"])

    # ── Stage 6: Decision ──────────────────────────────────────────────────
    t_decide = time.perf_counter()
    result = make_decision(
        description=description,
        has_image=has_image,
        text_top=text_top,
        image_top=image_top,
        pool=candidate_pool,
        reranked=reranked,
        confidence=confidence,
        gap=gap,
        agree=agree,
        auto_create_enabled=settings.AUTO_CREATE_ENABLED,
        confidence_threshold=settings.DECISION_CONFIDENCE_THRESHOLD,
        gap_threshold=settings.DECISION_GAP_THRESHOLD,
        no_action_floor=settings.DECISION_NO_ACTION_FLOOR,
        require_agree=settings.DECISION_REQUIRE_AGREE,
        always_draft_services=settings.always_draft_services_list,
        min_description_words=settings.MIN_DESCRIPTION_WORDS,
    )
    timings["decide_ms"] = round((time.perf_counter() - t_decide) * 1000, 2)
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 7: Persist to DecisionLog ───────────────────────────────────
    top_service_id: int | None = None
    if result.top_candidates:
        top_service_id = result.top_candidates[0].service_id

    try:
        log_data: dict[str, Any] = {
            "description": description,
            "store_description": settings.STORE_DESCRIPTION,
            "photo_ref": None,
            "photo_hash": photo_hash,
            "location": location,
            "text_top": text_top,
            "image_top": image_top,
            "pool": candidate_pool,
            "rerank_scores": reranked,
            "top_service_id": top_service_id,
            "confidence": confidence,
            "gap": gap,
            "agree": agree,
            "action": result.action,
            "would_be_action": result.would_be_action,
            "reason": result.reason,
            "text_model": "BAAI/bge-base-en-v1.5",
            "image_model": settings.SIGLIP_MODEL_ID,
            "reranker_model": "BAAI/bge-reranker-base",
            "config_version": settings.config_version,
            "timings": timings,
            "idempotency_key": idempotency_key,
        }
        decision_id = await repo.insert_decision_log(pool, log_data)
        result.decision_id = decision_id
    except Exception as exc:
        logger.warning("Failed to persist decision log: %s", exc)

    return result

