"""
Decision engine and reranking logic — ported 1:1 from photoToWorkOrder.
Includes softmax_with_temperature (identical to ptwo/reranker.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Softmax — identical to photoToWorkOrder/ptwo/reranker.py ─────────────────

def softmax_with_temperature(scores: list[float], temperature: float = 1.0) -> list[float]:
    """
    Numerically stable softmax over raw scores with configurable temperature.
    Note: result is a score distribution, NOT a calibrated probability.
    """
    if not scores:
        return []
    if temperature <= 0.0:
        raise ValueError(f"Softmax temperature must be strictly positive, got {temperature}")
    arr = np.array(scores, dtype=np.float64) / temperature
    arr_shifted = arr - np.max(arr)
    exp_vals = np.exp(arr_shifted)
    return (exp_vals / np.sum(exp_vals)).tolist()


# ── Decision dataclasses ─────────────────────────────────────────────────────

@dataclass
class CandidateOut:
    service_id: int | None
    title: str
    confidence: float | None
    confidence_pct: str | None
    text_rank: float | None = None
    image_rank: float | None = None
    pool_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class ProcessedPost:
    decision_id: int | None
    action: str
    would_be_action: str
    confidence: float | None
    gap: float | None
    agree: bool | None
    reason: str
    top_candidates: list[CandidateOut] = field(default_factory=list)
    idempotent_replay: bool = False
    shadow: bool = False    # True when AUTO_CREATE_ENABLED=False and would_be=auto_create


# ── Reranker wrapper (applies softmax to TEI raw logits) ─────────────────────

def apply_reranker_softmax(
    pool: list[dict[str, Any]],
    raw_scores: list[float],
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Takes the pool and raw cross-encoder logits from TEI (requested with raw_scores=true),
    applies temperature softmax, and returns pool items annotated and sorted.
    Matches photoToWorkOrder's BGEReranker.rerank() output structure exactly.
    """
    probs = softmax_with_temperature(raw_scores, temperature=temperature)

    reranked: list[dict[str, Any]] = []
    for item, raw_s, prob in zip(pool, raw_scores, probs):
        updated = dict(item)
        updated["rerank_score"] = float(raw_s)
        updated["confidence"] = float(prob)
        reranked.append(updated)

    reranked.sort(key=lambda x: (-x["confidence"], -x["rerank_score"]))
    for idx, item in enumerate(reranked, start=1):
        item["rerank_rank"] = idx

    return reranked


# ── Decision engine — ported 1:1 from photoToWorkOrder/ptwo/decision.py ──────

def make_decision(
    *,
    description: str | None,
    has_image: bool,
    text_top: list[dict[str, Any]],
    image_top: list[dict[str, Any]] | None,
    pool: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    confidence: float | None,
    gap: float | None,
    agree: bool | None,
    # config
    auto_create_enabled: bool,
    confidence_threshold: float,
    gap_threshold: float,
    no_action_floor: float,
    require_agree: bool,
    always_draft_services: list[str],
    min_description_words: int,
    top_candidates_count: int = 3,
) -> ProcessedPost:
    """
    Rule-based decision engine — identical business logic to RuleBasedDecider.decide().
    Returns a ProcessedPost (not Decision, but same fields).
    """
    trimmed_desc = description.strip() if description else ""

    # Case 1: Description missing → draft, rank by image only
    if not trimmed_desc:
        candidates: list[CandidateOut] = []
        if image_top:
            for item in image_top[:top_candidates_count]:
                score = item.get("score")
                candidates.append(CandidateOut(
                    service_id=item.get("service_id"),
                    title=item["service"],
                    confidence=score,
                    confidence_pct=f"{score * 100:.1f}%" if score is not None else None,
                    image_rank=float(item.get("rank", 0)),
                ))
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            confidence=None,
            gap=None,
            agree=agree,
            reason="Description missing: manual triage required (ranked by image only)",
        )

    # Build candidate list from reranked (or pool fallback)
    candidate_source = reranked if reranked else pool
    candidates = []
    for item in candidate_source[:top_candidates_count]:
        conf = item.get("confidence")
        candidates.append(CandidateOut(
            service_id=item.get("service_id"),
            title=item["service"],
            confidence=conf,
            confidence_pct=f"{conf * 100:.1f}%" if conf is not None else None,
            text_rank=item.get("text_rank"),
            image_rank=item.get("image_rank"),
            pool_rank=item.get("pool_rank"),
            rerank_score=item.get("rerank_score"),
        ))

    top_service = candidates[0].title if candidates else ""

    # Case 2: Short description → draft
    words = trimmed_desc.split()
    if len(words) < min_description_words:
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Description has {len(words)} word(s) (< {min_description_words} words required)",
        )

    # Case 3: Always-draft services
    if top_service in always_draft_services:
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Service '{top_service}' requires mandatory human draft review for safety",
        )

    # Case 4: Below confidence floor → no_action
    if confidence is not None and confidence < no_action_floor:
        return ProcessedPost(
            decision_id=None,
            action="no_action",
            would_be_action="no_action",
            top_candidates=candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Confidence {confidence:.4f} is below floor threshold ({no_action_floor})",
        )

    # Case 5: Meets auto_create criteria
    meets_conf = (confidence is not None) and (confidence >= confidence_threshold)
    meets_gap = (gap is not None) and (gap >= gap_threshold)
    meets_agree = (agree is True) or (agree is None) or (not require_agree)

    if meets_conf and meets_gap and meets_agree:
        would_be = "auto_create"
        if auto_create_enabled:
            action = "auto_create"
            reason = (
                f"High confidence ({confidence:.4f} >= {confidence_threshold}) "
                f"and gap ({gap:.4f} >= {gap_threshold})"
            )
            shadow = False
        else:
            action = "draft"
            reason = (
                f"Shadow mode: would be auto_create "
                f"(conf={confidence:.4f}>={confidence_threshold}, gap={gap:.4f}>={gap_threshold}), "
                f"routed to draft"
            )
            shadow = True
        return ProcessedPost(
            decision_id=None,
            action=action,
            would_be_action=would_be,
            top_candidates=candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=reason,
            shadow=shadow,
        )

    # Case 6: Default → draft
    return ProcessedPost(
        decision_id=None,
        action="draft",
        would_be_action="draft",
        top_candidates=candidates,
        confidence=confidence,
        gap=gap,
        agree=agree,
        reason=f"Criteria for auto_create not met (conf={confidence}, gap={gap}, agree={agree})",
    )

