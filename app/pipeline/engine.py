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
    softmax_with_temperature,
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
    dry_run: bool = False,
    return_diagnostics: bool = False,
) -> ProcessedPost:
    """
    Full async pipeline. Calls TEI over HTTP for embed/rerank;
    runs SigLIP via asyncio.to_thread to avoid blocking the event loop.
    When dry_run=True, skips writing to DecisionLog.
    When return_diagnostics=True, computes and attaches full-catalog rankings.
    """
    from app.db import repository as repo

    t0 = time.perf_counter()
    timings: dict[str, float] = {
        "image_decode_ms": 0.0,
        "siglip_encode_ms": 0.0,
        "text_embed_ms": 0.0,
        "pgvector_search_ms": 0.0,
        "text_ms": 0.0,
        "image_ms": 0.0,
        "pool_ms": 0.0,
        "rerank_ms": 0.0,
        "decision_ms": 0.0,
        "decide_ms": 0.0,
        "total_ms": 0.0,
    }

    # ── Parse image ───────────────────────────────────────────────────────────
    t_decode = time.perf_counter()
    pil_img: Image.Image | None = None
    photo_hash: str | None = None

    if image_bytes:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        photo_hash = hashlib.sha256(image_bytes).hexdigest()
    timings["image_decode_ms"] = round((time.perf_counter() - t_decode) * 1000, 2)

    has_image = pil_img is not None
    has_desc = bool(description and description.strip())

    # ── Stage 1: Text ranking ──────────────────────────────────────────────
    t_text = time.perf_counter()
    text_top: list[dict[str, Any]] = []
    all_text_results: list[dict[str, Any]] = []

    if has_desc and description:
        prefixed = BGE_QUERY_PREFIX + description.strip()
        try:
            t_embed = time.perf_counter()
            vectors = await tei_client.embed([prefixed])
            timings["text_embed_ms"] = round((time.perf_counter() - t_embed) * 1000, 2)
            query_vec = vectors[0]

            t_pg = time.perf_counter()
            if return_diagnostics:
                text_top, all_text_results = await repo.text_search(
                    pool, query_vec,
                    settings.PIPELINE_TEXT_DESC_WEIGHT,
                    settings.PIPELINE_TEXT_TOP_N,
                    return_all=True,
                )
            else:
                text_top = await repo.text_search(
                    pool, query_vec,
                    settings.PIPELINE_TEXT_DESC_WEIGHT,
                    settings.PIPELINE_TEXT_TOP_N,
                )
            timings["pgvector_search_ms"] = round((time.perf_counter() - t_pg) * 1000, 2)
        except Exception as exc:
            logger.warning("Text ranking failed: %s", exc)

    timings["text_ms"] = round((time.perf_counter() - t_text) * 1000, 2)

    # ── Stage 2: Image ranking ─────────────────────────────────────────────
    t_img = time.perf_counter()
    image_top: list[dict[str, Any]] | None = None
    all_image_results: list[dict[str, Any]] = []

    if has_image and pil_img and service_profiles:
        try:
            async with semaphore:
                if return_diagnostics:
                    image_top, all_image_results = await asyncio.to_thread(
                        rank_image_against_profiles,
                        pil_img,
                        service_profiles,
                        siglip_service,
                        settings.PIPELINE_IMAGE_TOP_N,
                        return_all_ranks=True,
                    )
                else:
                    image_top = await asyncio.to_thread(
                        rank_image_against_profiles,
                        pil_img,
                        service_profiles,
                        siglip_service,
                        settings.PIPELINE_IMAGE_TOP_N,
                    )
            timings["siglip_encode_ms"] = round((time.perf_counter() - t_img) * 1000, 2)
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

    enable_rerank = getattr(settings, "ENABLE_RERANKER", True)
    if enable_rerank and has_desc and description and candidate_pool:
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
    elif not enable_rerank and candidate_pool:
        # ── Bi-encoder path (no cross-encoder reranker) ──────────────────────
        # Problem: raw cosine scores from text (0.35–0.55) and image (0.04–0.10)
        # are on different scales, so blending them directly collapses the gap
        # between candidates, leaving gap ≈ 0 even when the top-1 is clearly correct.
        #
        # Fix: compute a per-candidate "adjusted score" based on the modal source
        # that actually ranked it (text if description present, image if not),
        # add a multimodal agreement boost when both modalities agree on rank-1,
        # then run softmax at a low temperature (PIPELINE_BIENCODER_TEMP, default
        # 0.06) to amplify the gap to a decisable range. This mirrors how a
        # cross-encoder produces discriminative confidences without needing TEI.
        #
        # Calibration sweep (65-row eval, 87.7% Top-1):
        #   temp=0.06, agree_boost=0.05 → 44 AUTO at 100% precision, 2 NO_ACTION, 19 DRAFT
        agree_boost = settings.PIPELINE_BIENCODER_AGREE_BOOST
        biencoder_temp = settings.PIPELINE_BIENCODER_TEMP
        has_desc = bool(description and description.strip())

        raw_adjusted: list[float] = []
        for item in candidate_pool:
            t_s = float(item.get("text_score") or 0.0)
            i_s = float(item.get("image_score") or 0.0)
            t_r = item.get("text_rank")
            i_r = item.get("image_rank")

            if has_desc:
                # Primary signal is text cosine similarity (same scale as threshold)
                if t_r is not None and i_r is not None:
                    # Both modalities present: use text score as base
                    score = t_s
                    # Full agreement boost if both ranked it first
                    if t_r == 1.0 and i_r == 1.0:
                        score += agree_boost
                    # Partial boost if image provides visual confirmation in top-3
                    elif i_r <= 3.0:
                        score += agree_boost * 0.5
                elif t_r is not None:
                    # Text only: use text score, down-weighted since no image confirmation
                    score = t_s * settings.PIPELINE_TEXT_DESC_WEIGHT
                else:
                    # Image only appeared via image search: weak signal without description match
                    score = i_s * (1.0 - settings.PIPELINE_TEXT_DESC_WEIGHT) * 2.0
            else:
                # No description: rely entirely on image score
                score = i_s

            raw_adjusted.append(score)

        # Apply softmax to amplify gap between candidates
        soft_scores = softmax_with_temperature(raw_adjusted, temperature=biencoder_temp)

        reranked = []
        for item, soft_conf, raw_s in zip(candidate_pool, soft_scores, raw_adjusted):
            updated = dict(item)
            updated["rerank_score"] = round(raw_s, 4)
            updated["confidence"] = round(soft_conf, 4)
            reranked.append(updated)

        # Sort by softmax probability (descending), then raw score as tiebreak
        reranked.sort(key=lambda x: (-x["confidence"], -x["rerank_score"]))
        for idx, item in enumerate(reranked, start=1):
            item["rerank_rank"] = idx

        if reranked:
            confidence = float(reranked[0]["confidence"])
            second_conf = float(reranked[1]["confidence"]) if len(reranked) > 1 else 0.0
            gap = max(0.0, round(confidence - second_conf, 4))

    timings["rerank_ms"] = round((time.perf_counter() - t_rerank) * 1000, 2)

    # ── Stage 5: Agreement signal ──────────────────────────────────────────
    agree: bool | None = None
    text_top1_name: str | None = text_top[0]["service"] if text_top else None
    image_top1_name: str | None = image_top[0]["service"] if image_top else None

    if has_image and image_top:
        if enable_rerank and reranked:
            agree = bool(image_top1_name == reranked[0]["service"])
        elif text_top:
            agree = bool(image_top1_name == text_top1_name)

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
    timings["decision_ms"] = round((time.perf_counter() - t_decide) * 1000, 2)
    timings["decide_ms"] = timings["decision_ms"]
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Attach diagnostic and timing metadata to result object
    result.timings = timings
    result.text_top1 = text_top1_name
    result.image_top1 = image_top1_name
    result.raw_rerank_scores = reranked

    if return_diagnostics:
        result.full_text_ranks = {
            item["service"]: int(item["rank"]) for item in all_text_results
        }
        result.full_image_ranks = {
            item["service"]: int(item["rank"]) for item in all_image_results
        }
        result.all_service_text_scores = {
            item["service"]: float(item.get("score", 0.0)) for item in all_text_results
        }
        result.all_service_image_scores = {
            item["service"]: float(item.get("score", 0.0)) for item in all_image_results
        }

        # Part 2: Additive diagnostics per row across all active services
        img_score_map: dict[str, float | None] = {}
        for item in all_image_results:
            svc_name = item.get("service")
            if svc_name:
                if item.get("has_visual_vote"):
                    img_score_map[svc_name] = float(item["score"])
                else:
                    img_score_map[svc_name] = None

        text_item_map: dict[str, dict[str, Any]] = {}
        for item in all_text_results:
            svc_name = item.get("service")
            if svc_name:
                text_item_map[svc_name] = item

        all_active_titles = set(text_item_map.keys())
        if service_profiles:
            for p in service_profiles.values():
                t = p.get("title")
                if t:
                    all_active_titles.add(t)

        svc_scores: dict[str, dict[str, float | None]] = {}
        for title in sorted(all_active_titles):
            t_item = text_item_map.get(title)
            if has_desc and t_item is not None:
                d_sim = float(t_item["desc_sim"]) if t_item.get("desc_sim") is not None else None
                t_sim = float(t_item["title_sim"]) if t_item.get("title_sim") is not None else None
                txt_score = float(t_item["score"]) if t_item.get("score") is not None else None
            else:
                d_sim = None
                t_sim = None
                txt_score = None

            i_sim = img_score_map.get(title)

            svc_scores[title] = {
                "desc_sim": d_sim,
                "title_sim": t_sim,
                "text_score": txt_score,
                "image_sim": i_sim,
            }

        result.service_scores = svc_scores

    # ── Stage 7: Persist to DecisionLog (skipped if dry_run=True) ──────────
    if not dry_run:
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

