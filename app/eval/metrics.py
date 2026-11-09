"""
Evaluation metrics calculation, statistical intervals, confusion matrix,
threshold grids, and offline calibration sweeps.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.pipeline.decision import make_decision, softmax_with_temperature


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes Wilson score interval for binomial proportion.
    More reliable than normal approximation for small n or proportions near 0/1.
    """
    if n <= 0:
        return (0.0, 0.0)

    # z-value for 95% confidence is ~1.95996
    z = 1.95996 if confidence == 0.95 else 1.64485
    p_hat = k / n
    z2 = z * z

    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)

    ci_low = max(0.0, (center - spread) / denom)
    ci_high = min(1.0, (center + spread) / denom)
    return (round(ci_low * 100, 1), round(ci_high * 100, 1))


# ── Shared NONE Classification Helper (Requirement 4) ────────────────────────

def classify_none_action(action: str | None) -> dict[str, Any]:
    """
    Evaluates triage action for out-of-scope NONE ground truth rows in three levels:
    1. no_action: correct (suppressed out-of-scope post)
    2. draft: soft false match (routed to human draft review)
    3. auto_create: harmful false match (harmful false-positive auto-creation)
    """
    act = (action or "").strip().lower().replace(" ", "_")
    if act in ("no_action", "none", ""):
        return {
            "level": "no_action",
            "is_correct": True,
            "is_false_match": False,
            "is_wrong_auto": False,
            "is_harmful": False,
            "label": "no_action (correct)",
            "severity": "correct",
            "badge_class": "badge-no-action",
        }
    elif act == "auto_create":
        return {
            "level": "auto_create",
            "is_correct": False,
            "is_false_match": True,
            "is_wrong_auto": True,
            "is_harmful": True,
            "label": "auto_create (harmful false match)",
            "severity": "harmful",
            "badge_class": "badge-auto",
        }
    else:  # draft or other fallback
        return {
            "level": "draft",
            "is_correct": False,
            "is_false_match": True,
            "is_wrong_auto": False,
            "is_harmful": False,
            "label": "draft (soft false match)",
            "severity": "soft",
            "badge_class": "badge-draft",
        }


def is_row_correct(
    *,
    is_none: bool,
    action: str | None,
    predicted_service: str | None,
    expected_service: str | None,
) -> bool:
    """
    Single shared helper used across job_manager and metrics to determine row correctness.
    For NONE rows: correct ONLY if action is 'no_action' (level: no_action).
    For normal rows: correct if predicted top-1 service matches expected service.
    """
    if is_none:
        return classify_none_action(action)["is_correct"]
    return bool(predicted_service and expected_service and (predicted_service == expected_service))


# ── AUROC Computation Helper (Requirement 5) ──────────────────────────────────

def compute_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    """
    Computes Area Under the ROC Curve (AUROC) using Mann-Whitney U rank statistic with mid-ranks for ties.
    Returns value in [0.0, 1.0], or None if either positive or negative class is missing.
    """
    if len(y_true) != len(y_score) or not y_true:
        return None
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pass

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_sum_pos = sum(ranks[idx] for idx, (_, label) in enumerate(pairs) if label == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


@dataclass
class EvalRowResult:
    row_id: int
    image: str
    description: str
    expected_service: str
    predicted_service: str
    action: str
    confidence: float | None
    gap: float | None
    correct: bool
    is_warmup: bool
    is_none: bool
    is_empty_desc: bool

    # Ranks
    text_pool_rank: int | None          # rank within text top-10, or None if "not in pool"
    image_pool_rank: int | None         # rank within image top-5, or None if "not in pool"
    reranker_rank: int | None           # rank after rerank, or None if "not in pool"
    full_text_rank: int | None          # rank across all services
    full_image_rank: int | None         # rank across all services
    entered_pool: bool                  # whether expected service entered the candidate pool

    # Signals
    text_top1: str | None
    image_top1: str | None
    agree: bool | None

    # Scores and candidates
    raw_reranker_scores: list[dict[str, Any]]  # [{service, raw_score, confidence}, ...]
    candidates: list[dict[str, Any]]           # full pool candidate list with ranks/scores

    # Timings per stage in ms
    timings: dict[str, float]
    # Small base64 thumbnail for live preview without disk dependency
    thumbnail_b64: str | None = None

    # Additive fields (Requirement 2 & Part 2)
    all_service_text_scores: dict[str, float] = field(default_factory=dict)
    all_service_image_scores: dict[str, float] = field(default_factory=dict)
    service_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    top1_raw_score: float | None = None
    top1_image_similarity: float | None = None
    top1_text_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Signal Quality Table (Requirement 5) ──────────────────────────────────────

def compute_signal_quality(
    rows: list[EvalRowResult],
    enable_reranker: bool = True,
) -> dict[str, Any]:
    """
    Computes AUROC with sample size n for each signal as a predictor of:
    (a) 'top-1 is correct' on normal (non-NONE) rows
    (b) 'normal row vs NONE row' across all evaluated rows
    Signals:
    - softmax confidence
    - gap
    - raw top-1 reranker score (only when reranker was on)
    - top-1 text similarity
    - top-1 image similarity
    Warns when a class has fewer than 10 rows.
    """
    signal_defs = [
        ("softmax_confidence", "Softmax Confidence", lambda r: r.confidence),
        ("gap", "Confidence Gap", lambda r: r.gap),
        ("raw_reranker_score", "Raw Top-1 Reranker Score", lambda r: r.top1_raw_score),
        ("top1_text_similarity", "Top-1 Text Similarity", lambda r: r.top1_text_similarity),
        ("top1_image_similarity", "Top-1 Image Similarity", lambda r: r.top1_image_similarity),
    ]

    normal_rows = [r for r in rows if not r.is_none]
    all_warnings: list[str] = []
    signals_report: list[dict[str, Any]] = []

    for key, name, extractor in signal_defs:
        if key == "raw_reranker_score" and not enable_reranker:
            signals_report.append({
                "key": key,
                "name": name,
                "top1_correct": {
                    "auroc": None,
                    "n": 0,
                    "pos_count": 0,
                    "neg_count": 0,
                    "warning": "Reranker OFF (cross-encoder did not run)",
                },
                "normal_vs_none": {
                    "auroc": None,
                    "n": 0,
                    "pos_count": 0,
                    "neg_count": 0,
                    "warning": "Reranker OFF (cross-encoder did not run)",
                },
            })
            continue

        # (a) Predictor of 'top-1 is correct' on normal rows
        norm_valid: list[tuple[float, int]] = []
        for r in normal_rows:
            v = extractor(r)
            if v is not None and not math.isnan(v):
                norm_valid.append((float(v), 1 if r.correct else 0))

        n_a = len(norm_valid)
        pos_a = sum(1 for _, y in norm_valid if y == 1)
        neg_a = n_a - pos_a
        warn_a = None
        if n_a > 0 and (pos_a < 10 or neg_a < 10):
            warn_a = f"Fewer than 10 rows in a class (correct: {pos_a}, incorrect: {neg_a})"
            all_warnings.append(f"{name} (Top-1 Correct): {warn_a}")

        auroc_a = None
        if pos_a > 0 and neg_a > 0:
            scores_a = [s for s, _ in norm_valid]
            labels_a = [y for _, y in norm_valid]
            raw_auc = compute_auroc(labels_a, scores_a)
            if raw_auc is not None:
                auroc_a = round(raw_auc, 3)

        # (b) Predictor of 'normal row vs NONE row' across all rows (normal=1, NONE=0)
        all_valid: list[tuple[float, int]] = []
        for r in rows:
            v = extractor(r)
            if v is not None and not math.isnan(v):
                all_valid.append((float(v), 0 if r.is_none else 1))

        n_b = len(all_valid)
        pos_b = sum(1 for _, y in all_valid if y == 1)
        neg_b = n_b - pos_b
        warn_b = None
        if n_b > 0 and (pos_b < 10 or neg_b < 10):
            warn_b = f"Fewer than 10 rows in a class (normal: {pos_b}, NONE: {neg_b})"
            all_warnings.append(f"{name} (Normal vs NONE): {warn_b}")

        auroc_b = None
        if pos_b > 0 and neg_b > 0:
            scores_b = [s for s, _ in all_valid]
            labels_b = [y for _, y in all_valid]
            raw_auc = compute_auroc(labels_b, scores_b)
            if raw_auc is not None:
                auroc_b = round(raw_auc, 3)

        signals_report.append({
            "key": key,
            "name": name,
            "top1_correct": {
                "auroc": auroc_a,
                "n": n_a,
                "pos_count": pos_a,
                "neg_count": neg_a,
                "warning": warn_a,
            },
            "normal_vs_none": {
                "auroc": auroc_b,
                "n": n_b,
                "pos_count": pos_b,
                "neg_count": neg_b,
                "warning": warn_b,
            },
        })

    return {
        "signals": signals_report,
        "warnings": all_warnings,
    }


def compute_metrics(
    rows: list[EvalRowResult],
    all_service_titles: list[str],
    current_conf_threshold: float = 0.40,
    current_gap_threshold: float = 0.10,
    current_no_action_floor: float = 0.25,
    total_elapsed_sec: float = 0.0,
    require_agree: bool = True,
    always_draft_services: list[str] | None = None,
    min_description_words: int = 3,
    enable_reranker: bool = True,
    text_guard_enabled: bool = True,
    text_no_action_floor: float = 0.40,
    text_auto_min: float = 0.45,
) -> dict[str, Any]:
    """
    Computes complete statistical report across all completed rows.
    Excludes warm-up rows from timing and accuracy statistics.
    Properly handles NONE rows:
      - Excluded from top-1/top-3/top-5, pool recall, per-service accuracy, and calibration reliability bins.
      - Tracked in separate NONE PANEL across 3 levels (no_action, draft, auto_create) with histograms.
      - Reuses pipeline make_decision for threshold sensitivity recomputation.
    """
    # Exclude warm-up requests
    eval_rows = [r for r in rows if not r.is_warmup]
    total_n = len(eval_rows)

    if total_n == 0:
        return {
            "n": 0,
            "message": "No evaluated rows yet (warmup only or empty dataset).",
            "warnings": ["Evaluation dataset contains no valid completed rows."],
            "findings": [],
        }

    # ── 1. Subsets ────────────────────────────────────────────────────────────
    non_none_rows = [r for r in eval_rows if not r.is_none]
    none_rows = [r for r in eval_rows if r.is_none]
    with_desc_rows = [r for r in eval_rows if not r.is_empty_desc]
    image_only_rows = [r for r in eval_rows if r.is_empty_desc]

    # Correct counts
    total_correct = sum(1 for r in eval_rows if r.correct)
    non_none_correct = sum(1 for r in non_none_rows if r.correct)
    none_correct = sum(1 for r in none_rows if r.correct)
    with_desc_correct = sum(1 for r in with_desc_rows if r.correct)
    image_only_correct = sum(1 for r in image_only_rows if r.correct)

    non_none_n = len(non_none_rows)
    n_none = len(none_rows)

    # ── 2. Top-k accuracy & Pool recall (EXCLUDING NONE rows) ─────────────────
    top3_correct = 0
    top5_correct = 0
    entered_pool_count = 0

    for r in non_none_rows:
        if r.entered_pool:
            entered_pool_count += 1
        if r.reranker_rank is not None:
            if r.reranker_rank <= 3:
                top3_correct += 1
            if r.reranker_rank <= 5:
                top5_correct += 1

    # Top-1 accuracy is strictly computed on normal (non-NONE) rows having a true service
    top1_acc = (non_none_correct / non_none_n * 100.0) if non_none_n else 0.0
    top3_acc = (top3_correct / non_none_n * 100.0) if non_none_n else 0.0
    top5_acc = (top5_correct / non_none_n * 100.0) if non_none_n else 0.0
    pool_recall = (entered_pool_count / non_none_n * 100.0) if non_none_n else 0.0

    # ── 3. Component accuracies & Agreement (non-NONE) ────────────────────────
    text_eligible = [r for r in non_none_rows if r.text_top1 is not None]
    text_correct = sum(1 for r in text_eligible if r.text_top1 == r.expected_service)
    text_top1_acc = (text_correct / len(text_eligible) * 100.0) if text_eligible else 0.0

    image_eligible = [r for r in non_none_rows if r.image_top1 is not None]
    image_correct = sum(1 for r in image_eligible if r.image_top1 == r.expected_service)
    image_top1_acc = (image_correct / len(image_eligible) * 100.0) if image_eligible else 0.0

    agree_rows = [r for r in non_none_rows if r.agree is True]
    disagree_rows = [r for r in non_none_rows if r.agree is False]

    agree_rate = (len(agree_rows) / non_none_n * 100.0) if non_none_n else 0.0
    acc_agree = (sum(1 for r in agree_rows if r.correct) / len(agree_rows) * 100.0) if agree_rows else 0.0
    acc_disagree = (sum(1 for r in disagree_rows if r.correct) / len(disagree_rows) * 100.0) if disagree_rows else 0.0

    # ── 4. Action distribution & Auto precision (includes NONE rows) ─────────
    auto_rows = [r for r in eval_rows if r.action == "auto_create"]
    draft_rows = [r for r in eval_rows if r.action == "draft"]
    no_action_rows = [r for r in eval_rows if r.action in ("no_action", "no action")]

    # A NONE row returned as AUTO is NOT correct, so it automatically counts in wrong_auto_rows
    auto_correct = sum(1 for r in auto_rows if r.correct)
    wrong_auto_rows = [r for r in auto_rows if not r.correct]
    wrong_auto_count = len(wrong_auto_rows)

    auto_precision = (auto_correct / len(auto_rows) * 100.0) if auto_rows else 0.0
    coverage = (len(auto_rows) / total_n * 100.0) if total_n else 0.0

    # ── 5. Latency & Throughput ───────────────────────────────────────────────
    total_latencies = [r.timings.get("total_ms", 0.0) for r in eval_rows]
    mean_latency = float(np.mean(total_latencies)) if total_latencies else 0.0
    p50_latency = float(np.percentile(total_latencies, 50)) if total_latencies else 0.0
    p95_latency = float(np.percentile(total_latencies, 95)) if total_latencies else 0.0
    max_latency = float(np.max(total_latencies)) if total_latencies else 0.0

    stage_keys = [
        ("image_decode_ms", "Image Decode"),
        ("siglip_encode_ms", "SigLIP Encode"),
        ("text_embed_ms", "Text Embed (TEI)"),
        ("pgvector_search_ms", "pgvector Search"),
        ("rerank_ms", "Rerank (TEI)"),
        ("decision_ms", "Decision Engine"),
    ]
    stage_breakdown = {}
    for k, label in stage_keys:
        vals = [r.timings.get(k, 0.0) for r in eval_rows if k in r.timings]
        stage_breakdown[k] = {
            "label": label,
            "mean_ms": round(float(np.mean(vals)), 2) if vals else 0.0,
        }

    throughput = round(total_n / max(total_elapsed_sec, 0.001), 2) if total_elapsed_sec > 0 else 0.0

    # ── 6. Per-Service accuracy (EXCLUDES NONE) & Confusion Matrix (INCLUDES NONE) ──
    # NONE rows have no true service, so exclude from per-service accuracy table
    actual_services = sorted(list(set(s for s in all_service_titles if s and s.upper() != "NONE")))
    service_stats: dict[str, dict[str, Any]] = {}
    for svc in actual_services:
        svc_rows = [r for r in eval_rows if r.expected_service == svc and not r.is_none]
        n_svc = len(svc_rows)
        if n_svc > 0:
            k_svc = sum(1 for r in svc_rows if r.correct)
            acc = round(k_svc / n_svc * 100.0, 1)
            ci_low, ci_high = wilson_score_interval(k_svc, n_svc, confidence=0.95)
            service_stats[svc] = {
                "service": svc,
                "n": n_svc,
                "correct": k_svc,
                "accuracy": acc,
                "ci_95": [ci_low, ci_high],
                "small_sample": bool(n_svc < 5),
            }

    # Confusion Matrix: Includes actual services + NONE row/column
    matrix_labels = list(actual_services)
    if "NONE" not in matrix_labels:
        matrix_labels.append("NONE")

    label_to_idx = {name: idx for idx, name in enumerate(matrix_labels)}
    confusion_counts = np.zeros((len(matrix_labels), len(matrix_labels)), dtype=int)
    confusions_list: dict[tuple[str, str], int] = {}

    for r in eval_rows:
        exp_svc = "NONE" if r.is_none else r.expected_service
        # For prediction in confusion matrix:
        # If action was "no action" (or "no_action"), predicted outcome is NONE.
        # Otherwise, predicted service is what the pipeline chose (even for NONE rows).
        if r.action in ("no_action", "no action"):
            pred_svc = "NONE"
        else:
            pred_svc = r.predicted_service or "NONE"

        exp_idx = label_to_idx.get(exp_svc)
        pred_idx = label_to_idx.get(pred_svc)

        if exp_idx is not None and pred_idx is not None:
            confusion_counts[exp_idx][pred_idx] += 1

        if not r.correct:
            pair = (exp_svc, pred_svc)
            confusions_list[pair] = confusions_list.get(pair, 0) + 1

    top_confusions = [
        {"expected": k[0], "predicted": k[1], "count": v}
        for k, v in sorted(confusions_list.items(), key=lambda x: x[1], reverse=True)
    ]

    # ── 7. Dedicated NONE PANEL (3 Levels: no_action, draft, auto_create) ─────
    none_no_action_rows = []
    none_draft_rows = []
    none_auto_create_rows = []

    for r in none_rows:
        cls_r = classify_none_action(r.action)
        if cls_r["level"] == "no_action":
            none_no_action_rows.append(r)
        elif cls_r["level"] == "auto_create":
            none_auto_create_rows.append(r)
        else:
            none_draft_rows.append(r)

    none_correct_count = len(none_no_action_rows)
    none_correct_rate = round(none_correct_count / n_none * 100.0, 1) if n_none else 0.0

    none_soft_count = len(none_draft_rows)
    none_soft_rate = round(none_soft_count / n_none * 100.0, 1) if n_none else 0.0

    none_harmful_count = len(none_auto_create_rows)
    none_harmful_rate = round(none_harmful_count / n_none * 100.0, 1) if n_none else 0.0

    false_match_count = none_soft_count + none_harmful_count

    false_matches_list = []
    for r in (none_draft_rows + none_auto_create_rows):
        cls_r = classify_none_action(r.action)
        false_matches_list.append({
            "row_id": r.row_id,
            "image": r.image,
            "description": r.description,
            "predicted_service": r.predicted_service,
            "action": r.action,
            "level": cls_r["level"],
            "is_harmful": cls_r["is_harmful"],
            "label": cls_r["label"],
            "badge_class": cls_r["badge_class"],
            "confidence": round(r.confidence, 4) if r.confidence is not None else 0.0,
            "gap": round(r.gap, 4) if r.gap is not None else 0.0,
            "thumbnail_b64": r.thumbnail_b64,
        })

    # Distribution of top-1 confidence for NONE rows vs correctly-answered normal rows
    correct_normal_rows = [
        r for r in eval_rows
        if not r.is_none and r.correct and r.confidence is not None
    ]
    none_confidences = [r.confidence for r in none_rows if r.confidence is not None]
    correct_normal_confidences = [r.confidence for r in correct_normal_rows]

    bin_edges = np.linspace(0.0, 1.0, 11)
    histogram_bins = []
    for b_idx in range(len(bin_edges) - 1):
        low = round(float(bin_edges[b_idx]), 2)
        high = round(float(bin_edges[b_idx + 1]), 2)
        label = f"{int(low * 100)}-{int(high * 100)}%"

        n_cnt = sum(
            1 for c in none_confidences
            if (low <= c < high) or (b_idx == 9 and low <= c <= high)
        )
        norm_cnt = sum(
            1 for c in correct_normal_confidences
            if (low <= c < high) or (b_idx == 9 and low <= c <= high)
        )
        histogram_bins.append({
            "bin_low": low,
            "bin_high": high,
            "label": label,
            "none_count": n_cnt,
            "normal_correct_count": norm_cnt,
        })

    none_panel = {
        "n": n_none,
        "correct_count": none_correct_count,
        "correct_rate": none_correct_rate,
        "false_match_count": false_match_count,
        "wrong_auto_count": none_harmful_count,  # Keep wrong-auto prominent!
        "levels": {
            "no_action": {
                "count": none_correct_count,
                "rate": none_correct_rate,
                "label": "no_action (correct)",
                "description": "Correctly suppressed out-of-scope requests",
            },
            "draft": {
                "count": none_soft_count,
                "rate": none_soft_rate,
                "label": "draft (soft false match)",
                "description": "Soft false match routed to human draft review",
            },
            "auto_create": {
                "count": none_harmful_count,
                "rate": none_harmful_rate,
                "label": "auto_create (harmful false match)",
                "description": "Harmful false match auto-created without review (wrong-auto)",
            },
        },
        "false_matches": false_matches_list,
        "histogram": histogram_bins,
    }

    # ── 8. Threshold Sensitivity Grid (Reusing pipeline make_decision) ────────
    conf_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    gap_grid = [0.00, 0.05, 0.10, 0.20]
    threshold_grid: list[dict[str, Any]] = []
    always_draft_list = always_draft_services or []

    for conf_th in conf_grid:
        for gap_th in gap_grid:
            auto_cnt = 0
            correct_auto_cnt = 0
            none_false_matches_cnt = 0

            for r in eval_rows:
                # Reconstruct candidate dicts for pipeline make_decision
                cand_dicts = [
                    {
                        "service": c.get("title", ""),
                        "service_id": c.get("service_id"),
                        "confidence": c.get("confidence"),
                        "rerank_score": c.get("raw_score"),
                        "text_score": c.get("text_score") or (r.top1_text_similarity if c_idx == 0 else None),
                        "text_rank": c.get("text_rank"),
                        "image_rank": c.get("image_rank"),
                        "pool_rank": c.get("pool_rank"),
                        "rerank_rank": c.get("rerank_rank"),
                    }
                    for c_idx, c in enumerate(r.candidates or [])
                ]
                text_top_dicts = [{"service": r.text_top1}] if r.text_top1 else []
                image_top_dicts = [{"service": r.image_top1}] if r.image_top1 else []

                # Recompute action using pipeline's own make_decision
                dec = make_decision(
                    description=r.description if r.description else None,
                    has_image=bool(r.image),
                    text_top=text_top_dicts,
                    image_top=image_top_dicts,
                    pool=cand_dicts,
                    reranked=cand_dicts,
                    confidence=r.confidence,
                    gap=r.gap,
                    agree=r.agree,
                    auto_create_enabled=True,
                    confidence_threshold=conf_th,
                    gap_threshold=gap_th,
                    no_action_floor=current_no_action_floor,
                    require_agree=require_agree,
                    always_draft_services=always_draft_list,
                    min_description_words=min_description_words,
                    enable_reranker=enable_reranker,
                    text_guard_enabled=text_guard_enabled,
                    text_no_action_floor=text_no_action_floor,
                    text_auto_min=text_auto_min,
                )

                act = dec.action

                if r.is_none:
                    c_none = classify_none_action(act)
                    if c_none["is_false_match"]:
                        none_false_matches_cnt += 1
                    if c_none["is_wrong_auto"]:
                        auto_cnt += 1  # Counted as auto, but wrong-auto!
                else:
                    if act == "auto_create":
                        auto_cnt += 1
                        if is_row_correct(is_none=False, action=act, predicted_service=r.predicted_service, expected_service=r.expected_service):
                            correct_auto_cnt += 1

            cov = round(auto_cnt / total_n * 100.0, 1) if total_n else 0.0
            prec = round(correct_auto_cnt / auto_cnt * 100.0, 1) if auto_cnt else 0.0
            threshold_grid.append({
                "confidence_threshold": conf_th,
                "gap_threshold": gap_th,
                "auto_count": auto_cnt,
                "coverage_pct": cov,
                "precision_pct": prec,
                "none_false_matches": none_false_matches_cnt,
            })

    # ── 9. Offline Temperature Calibration Sweeps (EXCLUDES NONE) ─────────────
    temps = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    calibration_report: list[dict[str, Any]] = []

    # Exclude NONE rows from calibration reliability bins
    rows_with_raw = [
        r for r in eval_rows
        if r.raw_reranker_scores and not r.is_empty_desc and not r.is_none
    ]

    for temp in temps:
        confidences: list[float] = []
        is_corrects: list[bool] = []
        gaps: list[float] = []

        for r in rows_with_raw:
            raw_scores = [c.get("raw_score", 0.0) for c in r.raw_reranker_scores]
            if not raw_scores:
                continue
            probs = softmax_with_temperature(raw_scores, temperature=temp)
            sorted_indices = np.argsort(probs)[::-1]
            c1 = probs[sorted_indices[0]]
            c2 = probs[sorted_indices[1]] if len(probs) > 1 else 0.0
            gap_val = c1 - c2

            top_candidate_svc = r.raw_reranker_scores[sorted_indices[0]].get("service", "")
            correct_at_temp = (top_candidate_svc == r.expected_service)

            confidences.append(c1)
            gaps.append(gap_val)
            is_corrects.append(correct_at_temp)

        # Build 10 reliability bins: [0-0.1, ..., 0.9-1.0]
        bin_edges = np.linspace(0.0, 1.0, 11)
        bins_data: list[dict[str, Any]] = []
        ece = 0.0
        n_scored = len(confidences)

        for b_idx in range(len(bin_edges) - 1):
            low = bin_edges[b_idx]
            high = bin_edges[b_idx + 1]
            indices_in_bin = [
                i for i, c in enumerate(confidences)
                if (low <= c < high) or (b_idx == 9 and low <= c <= high)
            ]
            bin_n = len(indices_in_bin)
            if bin_n > 0:
                avg_conf = float(np.mean([confidences[i] for i in indices_in_bin]))
                avg_acc = float(np.mean([1.0 if is_corrects[i] else 0.0 for i in indices_in_bin]))
                ece += (bin_n / max(n_scored, 1)) * abs(avg_acc - avg_conf)
            else:
                avg_conf = round((low + high) / 2.0, 2)
                avg_acc = 0.0

            bins_data.append({
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "label": f"{int(low * 100)}-{int(high * 100)}%",
                "n": bin_n,
                "avg_confidence": round(avg_conf * 100, 1),
                "accuracy": round(avg_acc * 100, 1),
                "small_sample": bool(bin_n < 30),
            })

        # Coverage / Precision under this temperature with current thresholds using make_decision
        temp_auto_cnt = 0
        temp_auto_correct = 0
        for c, g, corr, r in zip(confidences, gaps, is_corrects, rows_with_raw):
            cand_dicts = [
                {
                    "service": cd.get("title", ""),
                    "service_id": cd.get("service_id"),
                    "confidence": cd.get("confidence"),
                    "rerank_score": cd.get("raw_score"),
                    "text_rank": cd.get("text_rank"),
                    "image_rank": cd.get("image_rank"),
                    "pool_rank": cd.get("pool_rank"),
                    "rerank_rank": cd.get("rerank_rank"),
                }
                for cd in (r.candidates or [])
            ]
            text_top_dicts = [{"service": r.text_top1}] if r.text_top1 else []
            image_top_dicts = [{"service": r.image_top1}] if r.image_top1 else []

            dec = make_decision(
                description=r.description if r.description else None,
                has_image=bool(r.image),
                text_top=text_top_dicts,
                image_top=image_top_dicts,
                pool=cand_dicts,
                reranked=cand_dicts,
                confidence=c,
                gap=g,
                agree=r.agree,
                auto_create_enabled=True,
                confidence_threshold=current_conf_threshold,
                gap_threshold=current_gap_threshold,
                no_action_floor=current_no_action_floor,
                require_agree=require_agree,
                always_draft_services=always_draft_list,
                min_description_words=min_description_words,
            )
            if dec.action == "auto_create":
                temp_auto_cnt += 1
                if corr:
                    temp_auto_correct += 1

        cov_temp = round(temp_auto_cnt / total_n * 100.0, 1) if total_n else 0.0
        prec_temp = round(temp_auto_correct / temp_auto_cnt * 100.0, 1) if temp_auto_cnt else 0.0

        calibration_report.append({
            "temperature": temp,
            "ece": round(ece * 100, 2),
            "coverage_pct": cov_temp,
            "precision_pct": prec_temp,
            "bins": bins_data,
        })

    # ── 10. Findings and Small Sample Warnings ────────────────────────────────
    warnings: list[str] = []
    findings: list[str] = []

    if total_n < 30:
        warnings.append(
            f"Small sample size (total evaluated images n={total_n} < 30). "
            f"Confidence intervals are wide."
        )
    small_svc_count = sum(1 for s in service_stats.values() if s.get("small_sample"))
    if small_svc_count > 0:
        warnings.append(
            f"{small_svc_count} service categories have fewer than 5 sample images."
        )
    if wrong_auto_count > 0:
        warnings.append(
            f"CRITICAL: {wrong_auto_count} false-positive AUTO-CREATE decision(s) detected!"
        )

    # NONE Row Findings (3 Levels)
    if n_none > 0:
        if false_match_count > 0:
            fm_pct = round(false_match_count / n_none * 100.0, 1)
            parts = []
            if none_soft_count > 0:
                parts.append(f"{none_soft_count} draft (soft false match)")
            if none_harmful_count > 0:
                parts.append(f"{none_harmful_count} auto_create (harmful false match / wrong-auto)")
            details_str = ", ".join(parts)
            finding_msg = (
                f"NONE Ground Truth Finding: {false_match_count} of {n_none} ({fm_pct}%) out-of-scope 'NONE' "
                f"rows produced false matches ({details_str}) where model confidence exceeded the "
                f"no-action floor ({current_no_action_floor:.2f})."
            )
            findings.append(finding_msg)
            warnings.append(
                f"FALSE MATCHES DETECTED: {false_match_count}/{n_none} 'NONE' rows returned as false matches ({details_str})."
            )
        else:
            findings.append(
                f"NONE Ground Truth Finding: All {n_none} 'NONE' rows were correctly returned as 'no action' "
                f"(100% suppression rate)."
            )

        if none_harmful_count > 0:
            warnings.append(
                f"CRITICAL: {none_harmful_count} 'NONE' row(s) returned as AUTO-CREATE (harmful false match / WRONG-AUTO)!"
            )

    # ── 11. Signal Quality Analysis (Requirement 5) ───────────────────────────
    signal_quality = compute_signal_quality(eval_rows, enable_reranker=enable_reranker)
    for sq_warn in signal_quality.get("warnings", []):
        warnings.append(f"Signal Quality Warning: {sq_warn}")

    # ── 12. Assemble Final Summary ────────────────────────────────────────────
    return {
        "n": total_n,
        "overall": {
            "top1_accuracy": round(top1_acc, 1),
            "top3_accuracy": round(top3_acc, 1),
            "top5_accuracy": round(top5_acc, 1),
            "pool_recall": round(pool_recall, 1),
            "text_top1_accuracy": round(text_top1_acc, 1),
            "image_top1_accuracy": round(image_top1_acc, 1),
            "agree_rate": round(agree_rate, 1),
            "accuracy_when_agree": round(acc_agree, 1),
            "accuracy_when_disagree": round(acc_disagree, 1),
        },
        "subsets": {
            "all": {"n": total_n, "correct": total_correct, "accuracy": round(total_correct / total_n * 100.0, 1) if total_n else 0.0},
            "with_description": {
                "n": len(with_desc_rows),
                "correct": with_desc_correct,
                "accuracy": round(with_desc_correct / len(with_desc_rows) * 100.0, 1) if with_desc_rows else 0.0,
            },
            "image_only": {
                "n": len(image_only_rows),
                "correct": image_only_correct,
                "accuracy": round(image_only_correct / len(image_only_rows) * 100.0, 1) if image_only_rows else 0.0,
            },
            "none_expected": {
                "n": len(none_rows),
                "correct": none_correct,
                "accuracy": round(none_correct / len(none_rows) * 100.0, 1) if none_rows else 0.0,
            },
        },
        "actions": {
            "auto_create_count": len(auto_rows),
            "draft_count": len(draft_rows),
            "no_action_count": len(no_action_rows),
            "auto_precision": round(auto_precision, 1),
            "wrong_auto_count": wrong_auto_count,
            "coverage_pct": round(coverage, 1),
        },
        "none_panel": none_panel,
        "signal_quality": signal_quality,
        "latency": {
            "mean_ms": round(mean_latency, 1),
            "p50_ms": round(p50_latency, 1),
            "p95_ms": round(p95_latency, 1),
            "max_ms": round(max_latency, 1),
            "stages": stage_breakdown,
            "throughput_fps": throughput,
        },
        "services": list(service_stats.values()),
        "top_confusions": top_confusions[:15],
        "confusion_matrix": {
            "labels": matrix_labels,
            "matrix": confusion_counts.tolist(),
        },
        "threshold_grid": threshold_grid,
        "calibration": calibration_report,
        "warnings": warnings,
        "findings": findings,
    }

