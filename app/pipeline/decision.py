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
    rerank_rank: int | None = None
    raw_score: float | None = None
    text_score: float | None = None


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
    all_candidates: list[CandidateOut] = field(default_factory=list)
    text_top1: str | None = None
    image_top1: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    raw_rerank_scores: list[dict[str, Any]] = field(default_factory=list)
    full_text_ranks: dict[str, int] = field(default_factory=dict)
    full_image_ranks: dict[str, int] = field(default_factory=dict)
    all_service_text_scores: dict[str, float] = field(default_factory=dict)
    all_service_image_scores: dict[str, float] = field(default_factory=dict)
    service_scores: dict[str, dict[str, float | None]] = field(default_factory=dict)
    text_guard: dict[str, Any] | None = None
    agree_mode: str = "strict"   # agreement strictness level used for the auto gate


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


AGREE_MODES = ("strict", "balanced", "relaxed")
_UNSET = object()  # make_decision(agree_ok=...) not provided -> use the strict `agree` signal


def agreement_ok(
    mode: str,
    text_top: list[dict[str, Any]],
    image_top: list[dict[str, Any]] | None,
    text_margin: float = 0.05,
    image_top_k: int = 3,
) -> tuple[bool | None, str]:
    """
    Text/photo agreement for the auto-create gate, by strictness level.
    Returns (ok, detail); ok is None when there is no photo or no text ranking
    (the agreement rule then does not apply, same as before).
      strict   : photo top-1 == text top-1
      balanced : strict, OR text top-1 in photo top-K and its text score leads text #2 by >= text_margin
      relaxed  : text top-1 in photo top-K
    """
    if not image_top or not text_top:
        return None, "no photo or no text ranking"
    mode = mode if mode in AGREE_MODES else "strict"
    top = text_top[0].get("service")
    photo_rank = {it.get("service"): int(it.get("rank", 99)) for it in image_top}
    rank = photo_rank.get(top, 99)
    if rank == 1:
        return True, f"photo top-1 matches text top-1 ({mode})"
    if mode == "strict":
        return False, f"photo top-1 differs from text top-1 (text top-1 is photo #{rank if rank < 99 else '>' + str(len(image_top))}; mode=strict)"
    in_top_k = rank <= image_top_k
    if mode == "relaxed":
        return in_top_k, f"text top-1 is photo #{rank if rank < 99 else '>' + str(len(image_top))} (relaxed needs <= {image_top_k})"
    lead = None
    if len(text_top) > 1 and text_top[0].get("score") is not None and text_top[1].get("score") is not None:
        lead = float(text_top[0]["score"]) - float(text_top[1]["score"])
    ok = in_top_k and lead is not None and lead >= text_margin
    lead_s = f"{lead:.4f}" if lead is not None else "n/a"
    return ok, (f"text top-1 is photo #{rank if rank < 99 else '>' + str(len(image_top))}, text lead {lead_s} "
                f"(balanced needs photo <= {image_top_k} and lead >= {text_margin})")


def _safety_near_tie(
    top_service: str,
    text_top: list[dict[str, Any]],
    image_top: list[dict[str, Any]] | None,
    safety_services: list[str],
    margin: float | None,
    image_top_k: int,
) -> tuple[str, float, int] | None:
    """
    Return (safety_service, text_margin, photo_rank) when a safety-critical service other
    than the chosen one is in the text top-3 within `margin` of the top text score and in
    the photo top-k; otherwise None.
    """
    if not margin or not safety_services or top_service in safety_services or not text_top:
        return None
    top_score = text_top[0].get("score")
    if top_score is None:
        return None
    photo_rank = {it.get("service"): int(it.get("rank", 99)) for it in (image_top or [])}
    for item in text_top[:3]:
        svc = item.get("service")
        if svc in safety_services and svc != top_service and item.get("score") is not None:
            diff = float(top_score) - float(item["score"])
            if diff <= margin and photo_rank.get(svc, 99) <= image_top_k:
                return svc, diff, photo_rank[svc]
    return None


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
    # text guard config
    enable_reranker: bool = False,
    text_guard_enabled: bool = True,
    text_no_action_floor: float = 0.40,
    text_auto_min: float = 0.45,
    # safety near-tie config
    safety_services: list[str] | None = None,
    safety_near_tie_margin: float | None = None,
    safety_image_top_k: int = 3,
    # agreement strictness: when agree_ok is given it is used for the auto gate
    # (agree stays the strict photo top-1 == text top-1 signal for logging/metrics)
    agree_ok: Any = _UNSET,
    agree_mode: str = "strict",
    agree_detail: str | None = None,
) -> ProcessedPost:
    """
    Rule-based decision engine — identical business logic to RuleBasedDecider.decide().
    Returns a ProcessedPost (not Decision, but same fields).
    """
    trimmed_desc = description.strip() if description else ""

    # Case 1: Description missing → draft, rank by image only
    if not trimmed_desc:
        candidates: list[CandidateOut] = []
        all_candidates: list[CandidateOut] = []
        if image_top:
            for idx, item in enumerate(image_top, start=1):
                score = item.get("score")
                c_out = CandidateOut(
                    service_id=item.get("service_id"),
                    title=item["service"],
                    confidence=score,
                    confidence_pct=f"{score * 100:.1f}%" if score is not None else None,
                    image_rank=float(item.get("rank", 0)),
                    pool_rank=idx,
                    rerank_rank=idx,
                )
                all_candidates.append(c_out)
                if len(candidates) < top_candidates_count:
                    candidates.append(c_out)
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=None,
            gap=None,
            agree=agree,
            reason="Description missing: manual triage required (ranked by image only)",
        )

    # Build candidate list from reranked (or pool fallback)
    candidate_source = reranked if reranked else pool
    candidates = []
    all_candidates = []
    for idx, item in enumerate(candidate_source, start=1):
        conf = item.get("confidence")
        t_rank = item.get("text_rank")
        t_score = item.get("text_score") if t_rank is not None else None
        c_out = CandidateOut(
            service_id=item.get("service_id"),
            title=item["service"],
            confidence=conf,
            confidence_pct=f"{conf * 100:.1f}%" if conf is not None else None,
            text_rank=t_rank,
            image_rank=item.get("image_rank"),
            pool_rank=item.get("pool_rank"),
            rerank_score=item.get("rerank_score"),
            rerank_rank=item.get("rerank_rank", idx),
            raw_score=item.get("rerank_score"),
            text_score=t_score,
        )
        all_candidates.append(c_out)
        if len(candidates) < top_candidates_count:
            candidates.append(c_out)

    top_service = candidates[0].title if candidates else ""

    # Case 2: Short description → draft
    words = trimmed_desc.split()
    if len(words) < min_description_words:
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            all_candidates=all_candidates,
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
            all_candidates=all_candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Service '{top_service}' requires mandatory human draft review for safety",
        )

    # ── Text Guard (bi-encoder path only, description present) ───────────────
    text_guard_info: dict[str, Any] | None = None
    text_auto_blocked = False

    is_biencoder = not enable_reranker
    if is_biencoder and text_guard_enabled and trimmed_desc and candidates:
        top1_cand = candidates[0]
        # s = raw text_score of final top-1 candidate (same as exported top1_text_score, before softmax)
        s = getattr(top1_cand, "text_score", None)
        if s is None and candidate_source:
            if candidate_source[0].get("text_rank") is not None:
                s = candidate_source[0].get("text_score")

        if s is not None:
            text_guard_info = {
                "score": round(float(s), 4),
                "no_action_floor": text_no_action_floor,
                "auto_min": text_auto_min,
            }
            if s < text_no_action_floor:
                reason = f"text_score_below_floor (score={s:.4f} < floor={text_no_action_floor:.2f}, auto_min={text_auto_min:.2f})"
                text_guard_info["triggered"] = True
                text_guard_info["action"] = "no_action"
                text_guard_info["reason"] = "text_score_below_floor"
                return ProcessedPost(
                    decision_id=None,
                    action="no_action",
                    would_be_action="no_action",
                    top_candidates=candidates,
                    all_candidates=all_candidates,
                    confidence=confidence,
                    gap=gap,
                    agree=agree,
                    reason=reason,
                    text_guard=text_guard_info,
                )
            elif s < text_auto_min:
                text_auto_blocked = True
                text_guard_info["triggered"] = True
                text_guard_info["action"] = "draft"
                text_guard_info["reason"] = "text_score_below_auto_min"

    # Case 4: Below confidence floor → no_action
    if confidence is not None and confidence < no_action_floor:
        return ProcessedPost(
            decision_id=None,
            action="no_action",
            would_be_action="no_action",
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Confidence {confidence:.4f} is below floor threshold ({no_action_floor})",
            text_guard=text_guard_info,
        )

    # Case 5: Meets auto_create criteria
    meets_conf = (confidence is not None) and (confidence >= confidence_threshold)
    meets_gap = (gap is not None) and (gap >= gap_threshold)
    # Agreement gate: the level-aware result when the caller computed one, else the strict signal
    gate_agree = agree if agree_ok is _UNSET else agree_ok
    meets_agree = (gate_agree is True) or (gate_agree is None) or (not require_agree)

    if meets_conf and meets_gap and meets_agree:
        if text_auto_blocked:
            s_val = text_guard_info.get("score") if text_guard_info else 0.0
            reason = f"text_score_below_auto_min (score={s_val:.4f} < auto_min={text_auto_min:.2f}, floor={text_no_action_floor:.2f})"
            return ProcessedPost(
                decision_id=None,
                action="draft",
                would_be_action="draft",
                top_candidates=candidates,
                all_candidates=all_candidates,
                confidence=confidence,
                gap=gap,
                agree=agree,
                reason=reason,
                text_guard=text_guard_info,
            )
        else:
            # Safety near-tie: never auto-route a request to a non-safety service when a
            # safety-critical service (e.g. Emergency Response) is almost tied on the text
            # AND also shows up in the photo ranking - a human must look at it.
            tie = _safety_near_tie(
                top_service, text_top, image_top,
                safety_services or [], safety_near_tie_margin, safety_image_top_k,
            )
            if tie is not None:
                return ProcessedPost(
                    decision_id=None,
                    action="draft",
                    would_be_action="draft",
                    top_candidates=candidates,
                    all_candidates=all_candidates,
                    confidence=confidence,
                    gap=gap,
                    agree=agree,
                    reason=(
                        f"safety_near_tie ('{tie[0]}' text score within {tie[1]:.4f} of top-1 "
                        f"'{top_service}' and photo rank {tie[2]}): human review required"
                    ),
                    text_guard=text_guard_info,
                )
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
                all_candidates=all_candidates,
                confidence=confidence,
                gap=gap,
                agree=agree,
                reason=reason,
                shadow=shadow,
                text_guard=text_guard_info,
            )

    # Case 6: Default → draft
    default_reason = f"Criteria for auto_create not met (conf={confidence}, gap={gap}, agree={agree})"
    if require_agree and agree_ok is not _UNSET and gate_agree is False:
        default_reason += f" | agreement not met (mode={agree_mode}: {agree_detail or ''})"
    if text_auto_blocked:
        s_val = text_guard_info.get("score") if text_guard_info else 0.0
        default_reason = f"text_score_below_auto_min (score={s_val:.4f} < auto_min={text_auto_min:.2f}, floor={text_no_action_floor:.2f})"
    return ProcessedPost(
        decision_id=None,
        action="draft",
        would_be_action="draft",
        top_candidates=candidates,
        all_candidates=all_candidates,
        confidence=confidence,
        gap=gap,
        agree=agree,
        reason=default_reason,
        text_guard=text_guard_info,
    )

