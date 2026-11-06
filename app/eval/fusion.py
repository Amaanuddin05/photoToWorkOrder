"""
Offline Multimodal Fusion Explorer (Part 3).
Evaluates fusion formula families over stored text and image diagnostics.
Zero external services, zero model calls, numpy and stdlib only (no scipy/sklearn).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.pipeline.decision import make_decision

logger = logging.getLogger("siglip.eval.fusion")


# ── Wilson Score Confidence Interval ──────────────────────────────────────────

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes Wilson 95% score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% two-sided normal quantile
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ── AUROC with Mann-Whitney U Mid-Ranks ────────────────────────────────────────

def compute_auroc_rank(y_true: list[int], y_score: list[float]) -> float | None:
    """Non-parametric AUROC using Mann-Whitney U with mid-ranks for ties."""
    if not y_true or len(y_true) != len(y_score):
        return None
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

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

    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


# ── Softmax helper ─────────────────────────────────────────────────────────────

def softmax_rowwise(arr: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable row-wise softmax with temperature."""
    t = max(1e-6, temperature)
    scaled = arr / t
    max_vals = np.max(scaled, axis=1, keepdims=True)
    exp_vals = np.exp(scaled - max_vals)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def log_softmax_rowwise(arr: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable row-wise log-softmax with temperature."""
    t = max(1e-6, temperature)
    scaled = arr / t
    max_vals = np.max(scaled, axis=1, keepdims=True)
    exp_vals = np.exp(scaled - max_vals)
    sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
    return (scaled - max_vals) - np.log(np.maximum(sum_exp, 1e-12))


# ── Offline Replay of Pipeline Live Bi-Encoder ─────────────────────────────────

def replay_live_formula_row(
    row: dict[str, Any],
    all_service_titles: list[str],
    desc_weight: float = 0.7,
    agree_boost: float = 0.05,
    temp: float = 0.06,
    text_top_n: int = 10,
    image_top_n: int = 5,
    pool_max: int = 10,
) -> tuple[str, float]:
    """
    Replays the live bi-encoder candidate pool formation + scoring + softmax
    for a single row offline.
    """
    svc_scores = row.get("service_scores") or {}
    has_desc = bool(row.get("description") and str(row.get("description")).strip())

    # Build text top list
    text_items = []
    for svc in all_service_titles:
        data = svc_scores.get(svc) or {}
        score = data.get("text_score")
        if score is not None:
            text_items.append({"service": svc, "score": float(score)})
    # Sort text items descending by score, tie-break by service title
    text_items.sort(key=lambda x: (-x["score"], x["service"]))
    text_top = text_items[:text_top_n]
    for idx, item in enumerate(text_top, start=1):
        item["rank"] = float(idx)

    # Build image top list (only services with valid image profile)
    image_items = []
    for svc in all_service_titles:
        data = svc_scores.get(svc) or {}
        img_sim = data.get("image_sim")
        if img_sim is not None:
            image_items.append({"service": svc, "score": float(img_sim)})
    image_items.sort(key=lambda x: (-x["score"], x["service"]))
    image_top = image_items[:image_top_n]
    for idx, item in enumerate(image_top, start=1):
        item["rank"] = float(idx)

    # Build candidate pool exactly as app.pipeline.pool.build_candidate_pool does
    candidates_by_name: dict[str, dict[str, Any]] = {}
    for item in text_top:
        svc = item["service"]
        candidates_by_name[svc] = {
            "service": svc,
            "text_rank": float(item["rank"]),
            "image_rank": float("inf"),
            "text_score": float(item["score"]),
            "image_score": 0.0,
        }
    for item in image_top:
        svc = item["service"]
        if svc in candidates_by_name:
            candidates_by_name[svc]["image_rank"] = float(item["rank"])
            candidates_by_name[svc]["image_score"] = float(item["score"])
        else:
            candidates_by_name[svc] = {
                "service": svc,
                "text_rank": float("inf"),
                "image_rank": float(item["rank"]),
                "text_score": 0.0,
                "image_score": float(item["score"]),
            }

    pool_list = []
    for svc, meta in candidates_by_name.items():
        meta["min_rank"] = min(meta["text_rank"], meta["image_rank"])
        pool_list.append(meta)

    pool_list.sort(key=lambda x: (x["min_rank"], x["text_rank"], x["image_rank"], x["service"]))
    candidate_pool = pool_list[:pool_max]
    for idx, item in enumerate(candidate_pool, start=1):
        item["pool_rank"] = idx
        if item["text_rank"] == float("inf"):
            item["text_rank"] = None
        if item["image_rank"] == float("inf"):
            item["image_rank"] = None

    if not candidate_pool:
        return "", 0.0

    raw_adjusted: list[float] = []
    for item in candidate_pool:
        t_s = float(item.get("text_score") or 0.0)
        i_s = float(item.get("image_score") or 0.0)
        t_r = item.get("text_rank")
        i_r = item.get("image_rank")

        if has_desc:
            if t_r is not None and i_r is not None:
                score = t_s
                if t_r == 1.0 and i_r == 1.0:
                    score += agree_boost
                elif i_r <= 3.0:
                    score += agree_boost * 0.5
            elif t_r is not None:
                score = t_s * desc_weight
            else:
                score = i_s * (1.0 - desc_weight) * 2.0
        else:
            score = i_s

        raw_adjusted.append(score)

    # Softmax amplification at temp
    logits = np.array(raw_adjusted, dtype=np.float64) / max(1e-6, temp)
    logits -= np.max(logits)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l)

    reranked = []
    for item, soft_conf, raw_s in zip(candidate_pool, probs, raw_adjusted):
        reranked.append({
            "service": item["service"],
            "confidence": round(float(soft_conf), 4),
            "raw_score": round(float(raw_s), 4),
        })

    reranked.sort(key=lambda x: (-x["confidence"], -x["raw_score"]))
    top1 = reranked[0]["service"]
    conf = reranked[0]["confidence"]
    return top1, conf


# ── Stratified K-Fold Splitter (Pure Python / Numpy) ───────────────────────────

def stratified_kfold_indices(
    labels: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generates K-fold train/test index splits stratified by label.
    Distributes samples across folds in a balanced round-robin manner so all test
    splits remain non-empty even when individual classes have low counts.
    """
    rng = np.random.default_rng(seed)
    label_to_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    fold_test_indices: list[list[int]] = [[] for _ in range(n_splits)]
    sorted_labels = sorted(label_to_indices.keys(), key=lambda l: len(label_to_indices[l]), reverse=True)
    current_fold = 0
    for label in sorted_labels:
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        for idx in indices:
            fold_test_indices[current_fold % n_splits].append(idx)
            current_fold += 1

    all_indices = np.arange(len(labels))
    splits = []
    for k in range(n_splits):
        test_idx = np.array(sorted(fold_test_indices[k]), dtype=int)
        test_set = set(test_idx)
        train_idx = np.array([i for i in all_indices if i not in test_set], dtype=int)
        splits.append((train_idx, test_idx))
    return splits


# ── Main Fusion Analysis Function ─────────────────────────────────────────────

def run_fusion_analysis(run_data: dict[str, Any]) -> dict[str, Any]:
    """
    Runs complete offline fusion exploration analysis on finished evaluation run data.
    """
    effective_settings = run_data.get("effective_settings") or {}
    enable_rerank = effective_settings.get("enable_reranker", True)

    # ── Check pre-conditions ───────────────────────────────────────────────────
    if enable_rerank:
        return {
            "status": "rerun_needed",
            "message": "re-run needed: run was executed with reranker ON. Fusion Explorer requires a run with reranker OFF (bi-encoder mode).",
        }

    raw_rows = [r for r in run_data.get("rows", []) if not r.get("is_warmup", False)]
    if not raw_rows:
        return {
            "status": "error",
            "message": "Run contains no evaluated rows.",
        }

    # Verify Part 2 diagnostics fields exist
    has_part2 = any(
        isinstance(r.get("service_scores"), dict) and len(r.get("service_scores", {})) > 0
        for r in raw_rows
    )
    if not has_part2:
        return {
            "status": "rerun_needed",
            "message": "re-run needed: run is missing Part 2 diagnostics fields (service_scores). Please run a new evaluation.",
        }

    # Catalog & Demo Services info
    catalog_hash = effective_settings.get("catalog_hash", "unknown")
    has_demo_services = bool(effective_settings.get("has_demo_services", False))
    demo_service_count = int(effective_settings.get("demo_services_count", 0))
    demo_service_titles = effective_settings.get("demo_service_titles", [])

    # Identify active services across all rows
    all_services = sorted(list(set(
        s for r in raw_rows for s in (r.get("service_scores") or {}).keys()
    )))
    if not all_services:
        return {
            "status": "error",
            "message": "No active service titles found in service_scores.",
        }
    service_to_idx = {s: i for i, s in enumerate(all_services)}
    n_services = len(all_services)

    # ── Row Subsets ────────────────────────────────────────────────────────────
    normal_with_desc = [
        r for r in raw_rows
        if not r.get("is_none", False) and not r.get("is_empty_desc", False) and str(r.get("description") or "").strip()
    ]
    image_only_rows = [
        r for r in raw_rows
        if not r.get("is_none", False) and (r.get("is_empty_desc", False) or not str(r.get("description") or "").strip())
    ]
    none_rows = [r for r in raw_rows if r.get("is_none", False)]

    # ── Null image_sim tracking & Row Min substitution ─────────────────────────
    # "A service with image_sim null gets the row's minimum image_sim (it can never win by image); state this in the report."
    rows_with_null_image_sim = 0
    total_null_image_sim_instances = 0

    for r in raw_rows:
        scores = r.get("service_scores") or {}
        valid_img_sims = [v["image_sim"] for v in scores.values() if v.get("image_sim") is not None]
        min_img = min(valid_img_sims) if valid_img_sims else 0.0

        r_has_null = False
        r_preprocessed_img: dict[str, float] = {}
        for s in all_services:
            entry = scores.get(s) or {}
            val = entry.get("image_sim")
            if val is not None:
                r_preprocessed_img[s] = float(val)
            else:
                r_preprocessed_img[s] = float(min_img)
                r_has_null = True
                total_null_image_sim_instances += 1
        if r_has_null:
            rows_with_null_image_sim += 1
        r["_preprocessed_image_sims"] = r_preprocessed_img

    # ── 0. SANITY CHECKS ───────────────────────────────────────────────────────
    sanity_results: dict[str, Any] = {
        "passed": True,
        "checks": {},
    }

    # 0a. REPLAY: Recompute current live formula offline
    live_desc_w = float(effective_settings.get("description_weight", 0.7))
    live_agree_boost = float(effective_settings.get("agree_boost", 0.05))
    live_temp = float(effective_settings.get("biencoder_temp", 0.06))
    live_pool_max = int(effective_settings.get("pool_max", 10))
    live_text_top_n = int(effective_settings.get("text_top_n", 10))
    live_image_top_n = int(effective_settings.get("image_top_n", 5))

    replay_mismatches = 0
    replay_details = []

    for r in raw_rows:
        rep_top1, rep_conf = replay_live_formula_row(
            row=r,
            all_service_titles=all_services,
            desc_weight=live_desc_w,
            agree_boost=live_agree_boost,
            temp=live_temp,
            text_top_n=live_text_top_n,
            image_top_n=live_image_top_n,
            pool_max=live_pool_max,
        )
        stored_top1 = r.get("predicted_service", "")
        stored_conf = float(r.get("confidence") or 0.0)

        # Check top-1 match and confidence tolerance
        conf_diff = abs(rep_conf - stored_conf)
        if rep_top1 != stored_top1 or conf_diff > 1e-4:
            replay_mismatches += 1
            replay_details.append({
                "row_id": r.get("row_id"),
                "image": r.get("image"),
                "stored_top1": stored_top1,
                "replayed_top1": rep_top1,
                "stored_conf": stored_conf,
                "replayed_conf": rep_conf,
                "conf_diff": conf_diff,
            })

    check_a_passed = (replay_mismatches == 0)
    sanity_results["checks"]["replay"] = {
        "passed": check_a_passed,
        "mismatches": replay_mismatches,
        "tolerance": 1e-6,
        "total_rows": len(raw_rows),
        "failures": replay_details[:5],
    }

    # 0b. text-only (weight 0) top-1 == run text_top1; image-only (weight 1) == run image_top1
    text_matches = 0
    img_matches = 0
    n_normal_with_desc = len(normal_with_desc)

    for r in normal_with_desc:
        stored_text_top1 = r.get("text_top1")
        stored_img_top1 = r.get("image_top1")
        svc_scores = r.get("service_scores") or {}

        # Recomputed text top-1 across all services
        txt_candidates = [
            (s, float(svc_scores[s]["text_score"]))
            for s in all_services
            if s in svc_scores and svc_scores[s].get("text_score") is not None
        ]
        txt_candidates.sort(key=lambda x: (-x[1], x[0]))
        recomp_text_top1 = txt_candidates[0][0] if txt_candidates else None

        # Recomputed image top-1 across services with visual profile
        img_candidates = [
            (s, float(svc_scores[s]["image_sim"]))
            for s in all_services
            if s in svc_scores and svc_scores[s].get("image_sim") is not None
        ]
        img_candidates.sort(key=lambda x: (-x[1], x[0]))
        recomp_img_top1 = img_candidates[0][0] if img_candidates else None

        if recomp_text_top1 == stored_text_top1:
            text_matches += 1
        if recomp_img_top1 == stored_img_top1:
            img_matches += 1

    check_b_text_passed = (text_matches == n_normal_with_desc)
    check_b_img_passed = (img_matches == n_normal_with_desc)
    check_b_passed = check_b_text_passed and check_b_img_passed

    rep_text_0b = round((text_matches / max(1, n_normal_with_desc)) * 100, 2)
    rep_img_0b = round((img_matches / max(1, n_normal_with_desc)) * 100, 2)
    run_text_acc = round(float(effective_settings.get("text_top1_accuracy") or rep_text_0b), 2)
    run_img_acc = round(float(effective_settings.get("image_top1_accuracy") or rep_img_0b), 2)

    modality_info = {
        "passed": check_b_passed,
        "text_top1_matches": text_matches,
        "image_top1_matches": img_matches,
        "text_only_w0_top1": rep_text_0b,
        "image_only_w1_top1": rep_img_0b,
        "run_text_top1_accuracy": run_text_acc,
        "run_image_top1_accuracy": run_img_acc,
        "expected_matches": n_normal_with_desc,
    }
    sanity_results["checks"]["modality_top1"] = modality_info
    sanity_results["0b_modality"] = modality_info

    # 0c. desc_sim / title_sim recombined with run description weight == stored text score
    recombine_mismatches = 0
    max_recombine_diff = 0.0

    for r in normal_with_desc:
        svc_scores = r.get("service_scores") or {}
        for s in all_services:
            entry = svc_scores.get(s)
            if not entry:
                continue
            d_s = entry.get("desc_sim")
            t_s = entry.get("title_sim")
            stored_txt = entry.get("text_score")
            if d_s is not None and t_s is not None and stored_txt is not None:
                recomb = round(live_desc_w * float(d_s) + (1.0 - live_desc_w) * float(t_s), 4)
                diff = abs(recomb - float(stored_txt))
                if diff > max_recombine_diff:
                    max_recombine_diff = diff
                if diff > 1e-4:
                    recombine_mismatches += 1

    check_c_passed = (recombine_mismatches == 0)
    recombine_info = {
        "passed": check_c_passed,
        "mismatches": recombine_mismatches,
        "max_abs_diff": max_recombine_diff,
        "tolerance": 1e-4,
        "total_rows": len(normal_with_desc),
    }
    sanity_results["checks"]["recombine"] = recombine_info
    sanity_results["0c_recombine"] = recombine_info
    sanity_results["0a_replay"] = sanity_results["checks"]["replay"]

    sanity_all_passed = check_a_passed and check_b_passed and check_c_passed
    sanity_results["passed"] = sanity_all_passed

    if not sanity_all_passed:
        return {
            "status": "sanity_failed",
            "message": "Sanity checks failed: offline replay or score consistency mismatch detected. Halting sweep.",
            "sanity_checks": sanity_results,
        }

    # ── 1. FORMULA FAMILIES GRID SWEEP ─────────────────────────────────────────
    # Normal rows with description are used for formula selection
    wd_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    w_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    c_grid = [10, 30, 60]
    tt_grid = [0.02, 0.05, 0.1]
    ti_grid = [0.01, 0.02, 0.05]

    N = len(normal_with_desc)
    M = n_services

    # Build numpy matrices for normal rows with description
    desc_matrix = np.zeros((N, M), dtype=np.float64)
    title_matrix = np.zeros((N, M), dtype=np.float64)
    img_matrix = np.zeros((N, M), dtype=np.float64)
    target_idx = np.zeros(N, dtype=int)

    for i, r in enumerate(normal_with_desc):
        scores = r.get("service_scores") or {}
        prep_img = r.get("_preprocessed_image_sims") or {}
        exp_svc = r.get("expected_service")
        target_idx[i] = service_to_idx.get(exp_svc, -1)

        for s_idx, s in enumerate(all_services):
            e = scores.get(s) or {}
            desc_matrix[i, s_idx] = float(e.get("desc_sim") or 0.0)
            title_matrix[i, s_idx] = float(e.get("title_sim") or 0.0)
            img_matrix[i, s_idx] = float(prep_img.get(s, 0.0))

    # Precompute Text matrices for each wd
    text_matrices = {}
    for wd in wd_grid:
        text_matrices[wd] = wd * desc_matrix + (1.0 - wd) * title_matrix

    # Precompute rank matrices for RRF
    # Rank 1-based: rank 1 is highest score
    img_ranks = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        row_order = np.argsort(-img_matrix[i, :])
        ranks = np.empty(M, dtype=np.float64)
        ranks[row_order] = np.arange(1, M + 1)
        img_ranks[i, :] = ranks

    text_ranks_dict = {}
    for wd in wd_grid:
        t_mat = text_matrices[wd]
        t_ranks = np.zeros((N, M), dtype=np.float64)
        for i in range(N):
            row_order = np.argsort(-t_mat[i, :])
            ranks = np.empty(M, dtype=np.float64)
            ranks[row_order] = np.arange(1, M + 1)
            t_ranks[i, :] = ranks
        text_ranks_dict[wd] = t_ranks

    # Precompute log-softmax matrices for Log-Linear
    log_img_dict = {}
    for ti in ti_grid:
        log_img_dict[ti] = log_softmax_rowwise(img_matrix, ti)

    log_text_dict = {}
    for wd in wd_grid:
        log_text_dict[wd] = {}
        for tt in tt_grid:
            log_text_dict[wd][tt] = log_softmax_rowwise(text_matrices[wd], tt)

    # Precompute z-score matrices for Linear-Z
    img_mean = np.mean(img_matrix, axis=1, keepdims=True)
    img_std = np.std(img_matrix, axis=1, keepdims=True) + 1e-9
    z_img = (img_matrix - img_mean) / img_std

    z_text_dict = {}
    for wd in wd_grid:
        t_mat = text_matrices[wd]
        t_mean = np.mean(t_mat, axis=1, keepdims=True)
        t_std = np.std(t_mat, axis=1, keepdims=True) + 1e-9
        z_text_dict[wd] = (t_mat - t_mean) / t_std

    # Baseline performance (replayed live formula on normal rows with description)
    baseline_correct = np.zeros(N, dtype=bool)
    baseline_mrr_arr = np.zeros(N, dtype=np.float64)
    baseline_top3 = np.zeros(N, dtype=bool)

    for i, r in enumerate(normal_with_desc):
        rep_top1, _ = replay_live_formula_row(
            row=r,
            all_service_titles=all_services,
            desc_weight=live_desc_w,
            agree_boost=live_agree_boost,
            temp=live_temp,
            text_top_n=live_text_top_n,
            image_top_n=live_image_top_n,
            pool_max=live_pool_max,
        )
        exp_svc = r.get("expected_service")
        is_corr = (rep_top1 == exp_svc)
        baseline_correct[i] = is_corr
        # Candidate pool rank for MRR / Top-3
        # If expected entered candidates:
        r_pool = [c["service"] for c in (r.get("raw_rerank_scores") or [])]
        if exp_svc in r_pool:
            rank = r_pool.index(exp_svc) + 1
            baseline_mrr_arr[i] = 1.0 / rank
            baseline_top3[i] = (rank <= 3)
        else:
            baseline_mrr_arr[i] = 0.0
            baseline_top3[i] = False

    baseline_top1_acc = float(np.mean(baseline_correct))
    baseline_top3_acc = float(np.mean(baseline_top3))
    baseline_mrr = float(np.mean(baseline_mrr_arr))

    # Evaluate all configurations
    # Structure of config: (family, wd, w, params_dict)
    configs_list: list[dict[str, Any]] = []

    def evaluate_score_matrix(scores: np.ndarray, row_sub_indices: np.ndarray | None = None) -> tuple[float, float, float, np.ndarray]:
        """Calculates Top-1, Top-3, MRR and boolean is_top1_correct array."""
        sub_scores = scores if row_sub_indices is None else scores[row_sub_indices, :]
        sub_targets = target_idx if row_sub_indices is None else target_idx[row_sub_indices]
        n_sub = len(sub_targets)

        # Sort scores descending per row
        sorted_indices = np.argsort(-sub_scores, axis=1)

        top1_arr = (sorted_indices[:, 0] == sub_targets)
        top1_acc = float(np.mean(top1_arr))

        top3_arr = np.any(sorted_indices[:, :3] == sub_targets[:, None], axis=1)
        top3_acc = float(np.mean(top3_arr))

        ranks = np.where(sorted_indices == sub_targets[:, None])[1] + 1
        mrr = float(np.mean(1.0 / ranks))
        return top1_acc, top3_acc, mrr, top1_arr

    # 1. BASELINE_LIVE
    configs_list.append({
        "id": "BASELINE_LIVE",
        "family": "BASELINE_LIVE",
        "name": "Current Live Bi-Encoder (Replayed)",
        "wd": live_desc_w,
        "w": live_agree_boost,
        "params": {
            "temperature": live_temp,
            "agree_boost": live_agree_boost,
            "pool_max": live_pool_max,
        },
        "is_live_config": True,
        "top1_acc": baseline_top1_acc,
        "top3_acc": baseline_top3_acc,
        "mrr": baseline_mrr,
        "_correct_arr": baseline_correct,
        "_score_generator": None,
    })

    # 2. LINEAR_Z: (1 - w)*z_text + w*z_image
    for wd in wd_grid:
        zt = z_text_dict[wd]
        for w in w_grid:
            score_mat = (1.0 - w) * zt + w * z_img
            t1, t3, mrr, corr = evaluate_score_matrix(score_mat)
            configs_list.append({
                "id": f"LINEAR_Z_wd{wd}_w{w}",
                "family": "LINEAR_Z",
                "name": f"Linear Z-Score (w={w}, wd={wd})",
                "wd": wd,
                "w": w,
                "params": {"w": w, "wd": wd},
                "is_live_config": False,
                "top1_acc": t1,
                "top3_acc": t3,
                "mrr": mrr,
                "_correct_arr": corr,
                "_score_generator": ("LINEAR_Z", wd, w, None),
            })

    # 3. LINEAR_RAW: score = text + k*image
    for wd in wd_grid:
        tm = text_matrices[wd]
        for k in k_grid:
            score_mat = tm + k * img_matrix
            t1, t3, mrr, corr = evaluate_score_matrix(score_mat)
            configs_list.append({
                "id": f"LINEAR_RAW_wd{wd}_k{k}",
                "family": "LINEAR_RAW",
                "name": f"Linear Raw (k={k}, wd={wd})",
                "wd": wd,
                "w": k,  # k represents the relative weight
                "params": {"k": k, "wd": wd},
                "is_live_config": False,
                "top1_acc": t1,
                "top3_acc": t3,
                "mrr": mrr,
                "_correct_arr": corr,
                "_score_generator": ("LINEAR_RAW", wd, k, None),
            })

    # 4. RRF: 1/(c+rank_text) + w/(c+rank_image)
    for wd in wd_grid:
        tr = text_ranks_dict[wd]
        for c in c_grid:
            text_recip = 1.0 / (c + tr)
            img_recip = 1.0 / (c + img_ranks)
            for w in w_grid:
                score_mat = text_recip + w * img_recip
                t1, t3, mrr, corr = evaluate_score_matrix(score_mat)
                configs_list.append({
                    "id": f"RRF_wd{wd}_c{c}_w{w}",
                    "family": "RRF",
                    "name": f"Reciprocal Rank Fusion (c={c}, w={w}, wd={wd})",
                    "wd": wd,
                    "w": w,
                    "params": {"c": c, "w": w, "wd": wd},
                    "is_live_config": False,
                    "top1_acc": t1,
                    "top3_acc": t3,
                    "mrr": mrr,
                    "_correct_arr": corr,
                    "_score_generator": ("RRF", wd, w, {"c": c}),
                })

    # 5. LOGLINEAR: (1-w)*log_softmax(text/Tt) + w*log_softmax(image/Ti)
    for wd in wd_grid:
        for tt in tt_grid:
            lt = log_text_dict[wd][tt]
            for ti in ti_grid:
                li = log_img_dict[ti]
                for w in w_grid:
                    score_mat = (1.0 - w) * lt + w * li
                    t1, t3, mrr, corr = evaluate_score_matrix(score_mat)
                    configs_list.append({
                        "id": f"LOGLINEAR_wd{wd}_tt{tt}_ti{ti}_w{w}",
                        "family": "LOGLINEAR",
                        "name": f"Log-Linear (Tt={tt}, Ti={ti}, w={w}, wd={wd})",
                        "wd": wd,
                        "w": w,
                        "params": {"Tt": tt, "Ti": ti, "w": w, "wd": wd},
                        "is_live_config": False,
                        "top1_acc": t1,
                        "top3_acc": t3,
                        "mrr": mrr,
                        "_correct_arr": corr,
                        "_score_generator": ("LOGLINEAR", wd, w, {"Tt": tt, "Ti": ti}),
                    })

    # ── Helper to re-generate score matrix for any config ─────────────────────
    def get_score_matrix_for_config(cfg: dict[str, Any]) -> np.ndarray:
        gen = cfg.get("_score_generator")
        if gen is None:
            # Baseline dummy score
            return np.zeros((N, M), dtype=np.float64)
        fam, wd, w, p = gen
        if fam == "LINEAR_Z":
            return (1.0 - w) * z_text_dict[wd] + w * z_img
        elif fam == "LINEAR_RAW":
            return text_matrices[wd] + w * img_matrix
        elif fam == "RRF":
            c = p["c"]
            return 1.0 / (c + text_ranks_dict[wd]) + w * (1.0 / (c + img_ranks))
        elif fam == "LOGLINEAR":
            tt = p["Tt"]
            ti = p["Ti"]
            return (1.0 - w) * log_text_dict[wd][tt] + w * log_img_dict[ti]
        return np.zeros((N, M), dtype=np.float64)

    # ── 2. REPEATED STRATIFIED K-FOLD CV (K=5, 20 repeats = 100 splits) ───────
    labels_list = [str(r.get("expected_service")) for r in normal_with_desc]
    n_repeats = 20
    k_folds = 5
    total_splits = n_repeats * k_folds

    candidate_search_configs = [c for c in configs_list if c["family"] != "BASELINE_LIVE"]

    cv_selected_counts: dict[str, int] = {}
    cv_selected_names: dict[str, str] = {}
    cv_held_out_accs: list[float] = []
    cv_baseline_held_out_accs: list[float] = []
    cv_diffs: list[float] = []

    for repeat in range(n_repeats):
        splits = stratified_kfold_indices(labels_list, n_splits=k_folds, seed=42 + repeat)
        for train_idx, test_idx in splits:
            # Find best config on train_idx
            best_cfg = None
            best_train_t1 = -1.0
            best_train_mrr = -1.0
            best_simplicity_score = 999.0

            for cfg in candidate_search_configs:
                scores = get_score_matrix_for_config(cfg)
                t1, _, mrr, _ = evaluate_score_matrix(scores, train_idx)

                # Tie breaker metric: distance to central parameter (simpler setting)
                # w -> 0.5, wd -> 0.7
                w_dist = abs(float(cfg["w"]) - 0.5)
                wd_dist = abs(float(cfg["wd"]) - 0.7)
                simplicity = w_dist + wd_dist

                is_better = False
                if t1 > best_train_t1 + 1e-9:
                    is_better = True
                elif abs(t1 - best_train_t1) <= 1e-9:
                    if mrr > best_train_mrr + 1e-9:
                        is_better = True
                    elif abs(mrr - best_train_mrr) <= 1e-9:
                        if simplicity < best_simplicity_score - 1e-9:
                            is_better = True

                if is_better:
                    best_cfg = cfg
                    best_train_t1 = t1
                    best_train_mrr = mrr
                    best_simplicity_score = simplicity

            # Evaluate chosen config on test_idx
            if best_cfg is not None:
                cfg_id = best_cfg["id"]
                cv_selected_counts[cfg_id] = cv_selected_counts.get(cfg_id, 0) + 1
                cv_selected_names[cfg_id] = best_cfg["name"]

                test_scores = get_score_matrix_for_config(best_cfg)
                test_t1, _, _, _ = evaluate_score_matrix(test_scores, test_idx)
                cv_held_out_accs.append(test_t1)

                # Baseline on exact same test split
                base_t1 = float(np.mean(baseline_correct[test_idx]))
                cv_baseline_held_out_accs.append(base_t1)
                cv_diffs.append(test_t1 - base_t1)

    cv_mean_held_out = float(np.mean(cv_held_out_accs)) if cv_held_out_accs else baseline_top1_acc
    cv_mean_baseline = float(np.mean(cv_baseline_held_out_accs)) if cv_baseline_held_out_accs else baseline_top1_acc
    cv_mean_diff = float(np.mean(cv_diffs)) if cv_diffs else 0.0

    # Sort stability distribution descending
    cv_stability = [
        {"id": cid, "name": cv_selected_names[cid], "count": cnt, "pct": round((cnt / total_splits) * 100, 1)}
        for cid, cnt in sorted(cv_selected_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # ── Plateau View & Center Recommendation ──────────────────────────────────
    # For each family, map out top-1 accuracy grid over (w, wd)
    plateau_views: dict[str, Any] = {}
    family_best_configs: dict[str, dict[str, Any]] = {}

    for fam in ["LINEAR_Z", "LINEAR_RAW", "RRF", "LOGLINEAR"]:
        fam_cfgs = [c for c in candidate_search_configs if c["family"] == fam]
        if not fam_cfgs:
            continue

        # Extract grid axes
        wd_vals = sorted(list(set(c["wd"] for c in fam_cfgs)))
        w_vals = sorted(list(set(c["w"] for c in fam_cfgs)))

        grid_2d = []
        for wd in wd_vals:
            row_data = []
            for w in w_vals:
                matches = [c for c in fam_cfgs if abs(c["wd"] - wd) < 1e-5 and abs(c["w"] - w) < 1e-5]
                if matches:
                    best_m = max(matches, key=lambda x: (x["top1_acc"], x["mrr"]))
                    row_data.append({
                        "top1_acc": round(best_m["top1_acc"] * 100, 2),
                        "mrr": round(best_m["mrr"], 4),
                        "config_id": best_m["id"],
                    })
                else:
                    row_data.append({"top1_acc": 0.0, "mrr": 0.0, "config_id": ""})
            grid_2d.append(row_data)

        # Find plateau cells (within 0.0001 of max top1_acc in this family)
        max_fam_t1 = max(c["top1_acc"] for c in fam_cfgs)
        plateau_cells = [c for c in fam_cfgs if c["top1_acc"] >= max_fam_t1 - 1e-6]

        # Find center of plateau: closest to mean (w, wd) among plateau cells
        mean_w = np.mean([float(c["w"]) for c in plateau_cells])
        mean_wd = np.mean([float(c["wd"]) for c in plateau_cells])
        center_cell = min(
            plateau_cells,
            key=lambda c: (
                (float(c["w"]) - mean_w) ** 2 + (float(c["wd"]) - mean_wd) ** 2,
                -c["mrr"],
            ),
        )

        plateau_views[fam] = {
            "family": fam,
            "wd_values": wd_vals,
            "w_values": w_vals,
            "heatmap": grid_2d,
            "max_top1_acc": round(max_fam_t1 * 100, 2),
            "plateau_size": len(plateau_cells),
            "center_cell": center_cell["id"],
            "center_w": center_cell["w"],
            "center_wd": center_cell["wd"],
        }
        family_best_configs[fam] = center_cell

    # Recommend overall best configuration:
    # Pick among family centers based on CV selection frequency + highest validated accuracy
    most_stable_id = cv_stability[0]["id"] if cv_stability else None
    recommended_config = None

    if most_stable_id:
        for c in candidate_search_configs:
            if c["id"] == most_stable_id:
                recommended_config = c
                break

    if recommended_config is None:
        # Fallback to center of widest plateau across families
        widest_fam = max(plateau_views.values(), key=lambda p: (p["max_top1_acc"], p["plateau_size"]))
        recommended_config = family_best_configs[widest_fam["family"]]

    # ── Paired Bootstrap 95% Confidence Interval (2000 resamples) ─────────────
    rec_correct_arr = recommended_config["_correct_arr"]
    paired_deltas = rec_correct_arr.astype(int) - baseline_correct.astype(int)

    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_diffs = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sample_idx = rng.integers(0, N, size=N)
        boot_diffs[b] = np.mean(paired_deltas[sample_idx])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    mean_bootstrap_diff = float(np.mean(boot_diffs))

    wrong_to_right_count = int(np.sum((~baseline_correct) & rec_correct_arr))
    right_to_wrong_count = int(np.sum(baseline_correct & (~rec_correct_arr)))

    # CI includes 0 rule:
    # "If the CI of the difference includes 0, the recommendation must say 'no clear improvement: keep the current formula'. Do not hide this."
    ci_includes_zero = (ci_lower <= 0.0 <= ci_upper)

    if ci_includes_zero:
        verdict = "no clear improvement: keep the current formula"
        verdict_type = "keep_current"
        verdict_reason = (
            f"The 95% paired bootstrap confidence interval of the held-out difference includes 0 "
            f"([{ci_lower * 100:+.2f}%, {ci_upper * 100:+.2f}%]). "
            f"Observed flips: {wrong_to_right_count} gained, {right_to_wrong_count} lost. "
            f"Retaining the current live formula is recommended."
        )
    else:
        verdict = "clear improvement"
        verdict_type = "clear_improvement"
        verdict_reason = (
            f"The 95% paired bootstrap confidence interval of the held-out difference excludes 0 "
            f"([{ci_lower * 100:+.2f}%, {ci_upper * 100:+.2f}%]). "
            f"Observed flips: {wrong_to_right_count} gained, {right_to_wrong_count} lost."
        )

    # ── 3. DETAILED EVALUATION AT RECOMMENDED CONFIGURATION ───────────────────
    rec_scores = get_score_matrix_for_config(recommended_config)

    # 3a. Metrics with Wilson 95% intervals
    t1_rec, t3_rec, mrr_rec, _ = evaluate_score_matrix(rec_scores)
    t1_rec_low, t1_rec_high = wilson_interval(int(round(t1_rec * N)), N)
    t3_rec_low, t3_rec_high = wilson_interval(int(round(t3_rec * N)), N)

    t1_base_low, t1_base_high = wilson_interval(int(round(baseline_top1_acc * N)), N)
    t3_base_low, t3_base_high = wilson_interval(int(round(baseline_top3_acc * N)), N)

    # Per-service accuracy table
    per_service_stats = []
    service_acc_list_rec = []
    service_acc_list_base = []

    for s_idx, svc in enumerate(all_services):
        mask = (target_idx == s_idx)
        n_svc = int(np.sum(mask))
        if n_svc > 0:
            rec_corr_s = int(np.sum(rec_correct_arr[mask]))
            base_corr_s = int(np.sum(baseline_correct[mask]))
            acc_rec_s = rec_corr_s / n_svc
            acc_base_s = base_corr_s / n_svc
            service_acc_list_rec.append(acc_rec_s)
            service_acc_list_base.append(acc_base_s)

            low_s, high_s = wilson_interval(rec_corr_s, n_svc)
            per_service_stats.append({
                "service": svc,
                "n": n_svc,
                "warn_small_sample": (n_svc < 5),
                "recommended_correct": rec_corr_s,
                "recommended_acc_pct": round(acc_rec_s * 100, 1),
                "recommended_ci_95": [round(low_s * 100, 1), round(high_s * 100, 1)],
                "baseline_correct": base_corr_s,
                "baseline_acc_pct": round(acc_base_s * 100, 1),
            })
        else:
            per_service_stats.append({
                "service": svc,
                "n": 0,
                "warn_small_sample": True,
                "recommended_correct": 0,
                "recommended_acc_pct": 0.0,
                "recommended_ci_95": [0.0, 0.0],
                "baseline_correct": 0,
                "baseline_acc_pct": 0.0,
            })

    micro_acc_rec = t1_rec
    macro_acc_rec = float(np.mean(service_acc_list_rec)) if service_acc_list_rec else 0.0
    micro_acc_base = baseline_top1_acc
    macro_acc_base = float(np.mean(service_acc_list_base)) if service_acc_list_base else 0.0

    # Subset comparison: With description vs Image-only vs NONE
    # Image-only uses image score alone
    img_only_corr = 0
    for r in image_only_rows:
        exp_svc = r.get("expected_service")
        img_top = r.get("image_top1")
        if exp_svc and img_top and exp_svc == img_top:
            img_only_corr += 1
    n_img_only = len(image_only_rows)
    img_only_acc = (img_only_corr / n_img_only) if n_img_only > 0 else 0.0

    # Agreement splits: when text_top1 and image_top1 agree vs disagree
    agree_mask = np.array([
        bool(r.get("text_top1") and r.get("image_top1") and r.get("text_top1") == r.get("image_top1"))
        for r in normal_with_desc
    ])
    disagree_mask = ~agree_mask

    n_agree = int(np.sum(agree_mask))
    n_disagree = int(np.sum(disagree_mask))

    acc_when_agree_rec = float(np.mean(rec_correct_arr[agree_mask])) if n_agree > 0 else 0.0
    acc_when_disagree_rec = float(np.mean(rec_correct_arr[disagree_mask])) if n_disagree > 0 else 0.0
    acc_when_agree_base = float(np.mean(baseline_correct[agree_mask])) if n_agree > 0 else 0.0
    acc_when_disagree_base = float(np.mean(baseline_correct[disagree_mask])) if n_disagree > 0 else 0.0

    # 3b. Confidence Calibration Sweep over T
    # confidence = softmax(fused score / T) over ALL services
    t_sweep_grid = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
    # For fused score scales with large spread, scale T proportionally
    rec_score_std = float(np.std(rec_scores))
    if rec_score_std > 2.0 or rec_score_std < 0.05:
        scale_factor = rec_score_std / 0.1
        t_sweep_grid = sorted(list(set(round(t * scale_factor, 4) for t in t_sweep_grid)))

    calibration_results = []
    best_ece = 999.0
    best_t = t_sweep_grid[0]
    best_reliability_table = []

    for t_val in t_sweep_grid:
        probs = softmax_rowwise(rec_scores, t_val)
        confs = np.max(probs, axis=1)

        # 10 bins: [0, 0.1), [0.1, 0.2), ...
        bin_edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        rel_table = []

        for b in range(10):
            low, high = bin_edges[b], bin_edges[b + 1]
            if b == 9:
                in_bin = (confs >= low) & (confs <= high)
            else:
                in_bin = (confs >= low) & (confs < high)
            bin_n = int(np.sum(in_bin))
            if bin_n > 0:
                bin_acc = float(np.mean(rec_correct_arr[in_bin]))
                bin_conf = float(np.mean(confs[in_bin]))
                ece += (bin_n / N) * abs(bin_acc - bin_conf)
            else:
                bin_acc = 0.0
                bin_conf = (low + high) / 2.0

            rel_table.append({
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "label": f"{int(low * 100)}-{int(high * 100)}%",
                "n": bin_n,
                "warn_small_sample": (bin_n < 30),
                "avg_confidence": round(bin_conf * 100, 1),
                "accuracy": round(bin_acc * 100, 1),
            })

        calibration_results.append({
            "temperature": t_val,
            "ece": round(ece, 4),
            "ece_pct": round(ece * 100, 2),
        })

        if ece < best_ece:
            best_ece = ece
            best_t = t_val
            best_reliability_table = rel_table

    # 3c. Decision Table Recomputation
    # Reuses pipeline's make_decision() across confidence & gap threshold grid at best_t
    conf_thresh_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gap_thresh_grid = [0.0, 0.05, 0.1, 0.2]

    # Precompute fused probabilities and gaps at best_t
    best_probs = softmax_rowwise(rec_scores, best_t)
    sorted_probs = np.sort(best_probs, axis=1)[:, ::-1]
    rec_confs = sorted_probs[:, 0]
    rec_gaps = np.maximum(0.0, sorted_probs[:, 0] - sorted_probs[:, 1])

    # Rank-1 candidate for each row
    rec_top1_idx = np.argmax(rec_scores, axis=1)
    rec_top1_services = [all_services[idx] for idx in rec_top1_idx]

    # Compute fused scores and confidences for NONE rows as well
    N_none = len(none_rows)
    none_confs = np.zeros(N_none, dtype=np.float64)
    none_gaps = np.zeros(N_none, dtype=np.float64)

    if N_none > 0:
        none_desc_matrix = np.zeros((N_none, M), dtype=np.float64)
        none_title_matrix = np.zeros((N_none, M), dtype=np.float64)
        none_img_matrix = np.zeros((N_none, M), dtype=np.float64)

        for i, r in enumerate(none_rows):
            scores = r.get("service_scores") or {}
            prep_img = r.get("_preprocessed_image_sims") or {}
            for s_idx, s in enumerate(all_services):
                e = scores.get(s) or {}
                none_desc_matrix[i, s_idx] = float(e.get("desc_sim") or 0.0)
                none_title_matrix[i, s_idx] = float(e.get("title_sim") or 0.0)
                none_img_matrix[i, s_idx] = float(prep_img.get(s, 0.0))

        # Generate scores for none rows
        fam = recommended_config["family"]
        wd = recommended_config["wd"]
        w = recommended_config["w"]
        p = recommended_config["params"]

        if fam == "LINEAR_Z":
            none_tm = wd * none_desc_matrix + (1.0 - wd) * none_title_matrix
            zt_none = (none_tm - np.mean(none_tm, axis=1, keepdims=True)) / (np.std(none_tm, axis=1, keepdims=True) + 1e-9)
            zi_none = (none_img_matrix - np.mean(none_img_matrix, axis=1, keepdims=True)) / (np.std(none_img_matrix, axis=1, keepdims=True) + 1e-9)
            none_rec_scores = (1.0 - w) * zt_none + w * zi_none
        elif fam == "LINEAR_RAW":
            none_tm = wd * none_desc_matrix + (1.0 - wd) * none_title_matrix
            none_rec_scores = none_tm + w * none_img_matrix
        elif fam == "RRF":
            c_val = p["c"]
            none_tm = wd * none_desc_matrix + (1.0 - wd) * none_title_matrix
            tr_none = np.zeros((N_none, M), dtype=np.float64)
            ir_none = np.zeros((N_none, M), dtype=np.float64)
            for i in range(N_none):
                ro = np.argsort(-none_tm[i, :])
                ranks = np.empty(M, dtype=np.float64)
                ranks[ro] = np.arange(1, M + 1)
                tr_none[i, :] = ranks

                ro_i = np.argsort(-none_img_matrix[i, :])
                ranks_i = np.empty(M, dtype=np.float64)
                ranks_i[ro_i] = np.arange(1, M + 1)
                ir_none[i, :] = ranks_i
            none_rec_scores = 1.0 / (c_val + tr_none) + w * (1.0 / (c_val + ir_none))
        elif fam == "LOGLINEAR":
            none_tm = wd * none_desc_matrix + (1.0 - wd) * none_title_matrix
            lt_none = log_softmax_rowwise(none_tm, p["Tt"])
            li_none = log_softmax_rowwise(none_img_matrix, p["Ti"])
            none_rec_scores = (1.0 - w) * lt_none + w * li_none
        else:
            none_rec_scores = np.zeros((N_none, M), dtype=np.float64)

        none_probs = softmax_rowwise(none_rec_scores, best_t)
        sorted_none = np.sort(none_probs, axis=1)[:, ::-1]
        none_confs = sorted_none[:, 0]
        none_gaps = np.maximum(0.0, sorted_none[:, 0] - sorted_none[:, 1])

    decision_table = []
    floor_val = float(effective_settings.get("no_action_floor", 0.25))
    always_draft = effective_settings.get("always_draft_services") or []

    for c_th in conf_thresh_grid:
        for g_th in gap_thresh_grid:
            auto_count = 0
            draft_count = 0
            no_action_count = 0
            wrong_auto_count = 0

            # Evaluate on normal rows with description
            for i in range(N):
                c_val = float(rec_confs[i])
                g_val = float(rec_gaps[i])
                pred_svc = rec_top1_services[i]
                exp_svc = normal_with_desc[i].get("expected_service")

                # Reconstruct pool-like top structure for make_decision
                fake_top = [{"service": pred_svc, "confidence": c_val}]
                dec = make_decision(
                    description=normal_with_desc[i].get("description"),
                    has_image=True,
                    text_top=fake_top,
                    image_top=fake_top,
                    pool=fake_top,
                    reranked=fake_top,
                    confidence=c_val,
                    gap=g_val,
                    agree=True,  # Agreement not forced unless require_agree=true
                    auto_create_enabled=True,
                    confidence_threshold=c_th,
                    gap_threshold=g_th,
                    no_action_floor=floor_val,
                    require_agree=False,
                    always_draft_services=always_draft,
                    min_description_words=3,
                )
                if dec.action == "auto_create":
                    auto_count += 1
                    if pred_svc != exp_svc:
                        wrong_auto_count += 1
                elif dec.action == "draft":
                    draft_count += 1
                else:
                    no_action_count += 1

            # Evaluate on NONE rows
            none_l1_no_action = 0
            none_l2_draft = 0
            none_l3_auto = 0

            for i in range(N_none):
                c_val = float(none_confs[i])
                g_val = float(none_gaps[i])
                fake_top = [{"service": all_services[0], "confidence": c_val}]
                dec = make_decision(
                    description=none_rows[i].get("description"),
                    has_image=True,
                    text_top=fake_top,
                    image_top=fake_top,
                    pool=fake_top,
                    reranked=fake_top,
                    confidence=c_val,
                    gap=g_val,
                    agree=True,
                    auto_create_enabled=True,
                    confidence_threshold=c_th,
                    gap_threshold=g_th,
                    no_action_floor=floor_val,
                    require_agree=False,
                    always_draft_services=always_draft,
                    min_description_words=3,
                )
                if dec.action == "no_action":
                    none_l1_no_action += 1
                elif dec.action == "draft":
                    none_l2_draft += 1
                elif dec.action == "auto_create":
                    none_l3_auto += 1

            coverage = (auto_count / N) * 100 if N > 0 else 0.0
            auto_precision = ((auto_count - wrong_auto_count) / auto_count) * 100 if auto_count > 0 else 100.0

            decision_table.append({
                "confidence_threshold": c_th,
                "gap_threshold": g_th,
                "coverage_pct": round(coverage, 1),
                "auto_precision_pct": round(auto_precision, 1),
                "auto_count": auto_count,
                "wrong_auto_count": wrong_auto_count,
                "draft_count": draft_count,
                "no_action_count": no_action_count,
                "none_l1_correct_no_action": none_l1_no_action,
                "none_l2_soft_draft": none_l2_draft,
                "none_l3_harmful_auto": none_l3_auto,
            })

    # 3d. Signal Quality Table (AUROC with sample warnings)
    # Signals: fused top score, softmax confidence, gap, top text sim, top image sim
    fused_top_scores = np.max(rec_scores, axis=1)
    text_top_scores = np.max(text_matrices[recommended_config["wd"]], axis=1)
    image_top_scores = np.max(img_matrix, axis=1)

    y_correct = rec_correct_arr.astype(int).tolist()
    signal_top1_correct = [
        {"name": "Fused Top Score", "auroc": compute_auroc_rank(y_correct, fused_top_scores.tolist())},
        {"name": "Softmax Confidence", "auroc": compute_auroc_rank(y_correct, rec_confs.tolist())},
        {"name": "Score Gap", "auroc": compute_auroc_rank(y_correct, rec_gaps.tolist())},
        {"name": "Top Text Similarity", "auroc": compute_auroc_rank(y_correct, text_top_scores.tolist())},
        {"name": "Top Image Similarity", "auroc": compute_auroc_rank(y_correct, image_top_scores.tolist())},
    ]

    # Task B: Normal vs NONE (all evaluated rows)
    all_is_normal: list[int] = [1] * N + [0] * N_none
    all_fused_scores: list[float] = fused_top_scores.tolist() + (none_confs.tolist() if N_none > 0 else [])
    all_confs: list[float] = rec_confs.tolist() + (none_confs.tolist() if N_none > 0 else [])
    all_gaps: list[float] = rec_gaps.tolist() + (none_gaps.tolist() if N_none > 0 else [])

    signal_none_distinction = [
        {"name": "Fused Top Score", "auroc": compute_auroc_rank(all_is_normal, all_fused_scores)},
        {"name": "Softmax Confidence", "auroc": compute_auroc_rank(all_is_normal, all_confs)},
        {"name": "Score Gap", "auroc": compute_auroc_rank(all_is_normal, all_gaps)},
    ]

    signal_quality_summary = {
        "n_top1_correct_eval": N,
        "n_correct": int(np.sum(rec_correct_arr)),
        "n_incorrect": int(np.sum(~rec_correct_arr)),
        "warn_top1_correct_small": (int(np.sum(rec_correct_arr)) < 10 or int(np.sum(~rec_correct_arr)) < 10),
        "n_none_distinction_eval": N + N_none,
        "n_normal": N,
        "n_none": N_none,
        "warn_none_distinction_small": (N < 10 or N_none < 10),
        "signals_top1_correct": signal_top1_correct,
        "signals_none_distinction": signal_none_distinction,
    }

    # 3e. Changed Rows Table (before/after vs baseline)
    changed_rows_table = []
    for i, r in enumerate(normal_with_desc):
        b_corr = bool(baseline_correct[i])
        r_corr = bool(rec_correct_arr[i])
        b_pred = str(r.get("predicted_service"))
        r_pred = rec_top1_services[i]

        if b_pred != r_pred:
            if not b_corr and r_corr:
                change_type = "wrong_to_right"
            elif b_corr and not r_corr:
                change_type = "right_to_wrong"
            else:
                change_type = "other_changed"

            changed_rows_table.append({
                "row_id": r.get("row_id"),
                "image": r.get("image"),
                "description": r.get("description"),
                "expected_service": r.get("expected_service"),
                "baseline_pred": b_pred,
                "baseline_conf": float(r.get("confidence") or 0.0),
                "baseline_correct": b_corr,
                "recommended_pred": r_pred,
                "recommended_conf": round(float(rec_confs[i]), 4),
                "recommended_correct": r_corr,
                "fused_score": round(float(fused_top_scores[i]), 4),
                "text_score": float(r.get("top1_text_similarity") or 0.0),
                "image_sim": float(r.get("top1_image_similarity") or 0.0),
                "change_type": change_type,
                "thumbnail_b64": r.get("thumbnail_b64"),
            })

    # Warnings collection
    warnings_list = []
    if N < 50:
        warnings_list.append(f"Small sample size warning: Only {N} normal rows with description evaluated.")
    if has_demo_services:
        warnings_list.append(
            f"Catalog Warning: {demo_service_count} demo service(s) are active in catalog ({', '.join(demo_service_titles)}). Catalog hash: {catalog_hash}."
        )
    if rows_with_null_image_sim > 0:
        warnings_list.append(
            f"Image profile warning: {rows_with_null_image_sim} rows contain active services with no image profile (null image_sim was substituted with row minimum image_sim: {total_null_image_sim_instances} total service instances)."
        )
    if any(s.get("warn_small_sample") for s in per_service_stats if s.get("n", 0) > 0):
        low_svc_count = sum(1 for s in per_service_stats if 0 < s.get("n", 0) < 5)
        warnings_list.append(f"Per-service accuracy warning: {low_svc_count} service categories have fewer than 5 sample instances (n < 5).")

    # ── Formula in words and equation ──────────────────────────────────────────
    fam = recommended_config["family"]
    rec_w = recommended_config["w"]
    rec_wd = recommended_config["wd"]

    if fam == "LINEAR_Z":
        eq_text = f"score(s) = (1 - {rec_w}) * z_text(s) + {rec_w} * z_image(s)"
        words_text = (
            f"Z-score normalization of text score (desc_weight={rec_wd}) and SigLIP image similarity "
            f"across active services, linearly fused with image weight w={rec_w} and text weight 1-w={round(1.0 - rec_w, 2)}."
        )
    elif fam == "LINEAR_RAW":
        eq_text = f"score(s) = text(s) + {rec_w} * image(s)"
        words_text = (
            f"Linear raw combination of text score (desc_weight={rec_wd}) and SigLIP image similarity "
            f"with scaling factor k={rec_w}."
        )
    elif fam == "RRF":
        c_val = recommended_config["params"]["c"]
        eq_text = f"score(s) = 1 / ({c_val} + rank_text(s)) + {rec_w} / ({c_val} + rank_image(s))"
        words_text = (
            f"Reciprocal Rank Fusion with smoothing constant c={c_val}, description weight wd={rec_wd}, "
            f"and image rank weight w={rec_w}."
        )
    elif fam == "LOGLINEAR":
        tt = recommended_config["params"]["Tt"]
        ti = recommended_config["params"]["Ti"]
        eq_text = f"score(s) = (1 - {rec_w}) * log_softmax(text / {tt})(s) + {rec_w} * log_softmax(image / {ti})(s)"
        words_text = (
            f"Log-linear probability mixture of text temperature softmax (Tt={tt}, wd={rec_wd}) "
            f"and image temperature softmax (Ti={ti}) with fusion weight w={rec_w}."
        )
    else:
        eq_text = "Current live candidate pool + agreement boost + softmax(temp=0.06)"
        words_text = "Baseline live bi-encoder pipeline formula."

    # ── Assemble Full Report Object ───────────────────────────────────────────
    report_data: dict[str, Any] = {
        "status": "success",
        "job_id": run_data.get("job_id"),
        "catalog_hash": catalog_hash,
        "has_demo_services": has_demo_services,
        "demo_service_count": demo_service_count,
        "demo_service_titles": demo_service_titles,
        "rows_with_null_image_sim": rows_with_null_image_sim,
        "total_null_image_sim_instances": total_null_image_sim_instances,
        "subsets_count": {
            "all_rows": len(raw_rows),
            "normal_with_desc": N,
            "image_only": len(image_only_rows),
            "none_rows": N_none,
        },
        "sanity_checks": sanity_results,
        "recommendation": {
            "family": fam,
            "config_id": recommended_config["id"],
            "config_name": recommended_config["name"],
            "formula_words": words_text,
            "formula_equation": eq_text,
            "parameters": {
                "w": rec_w,
                "wd": rec_wd,
                "temperature": best_t,
                "family_params": recommended_config["params"],
            },
            "held_out_accuracy": {
                "recommended_mean_held_out_pct": round(cv_mean_held_out * 100, 2),
                "baseline_mean_held_out_pct": round(cv_mean_baseline * 100, 2),
                "mean_diff_pct": round(cv_mean_diff * 100, 2),
                "ci_95_bootstrap": [round(ci_lower * 100, 2), round(ci_upper * 100, 2)],
                "ci_includes_zero": ci_includes_zero,
            },
            "flips": {
                "wrong_to_right": wrong_to_right_count,
                "right_to_wrong": right_to_wrong_count,
                "net_change": wrong_to_right_count - right_to_wrong_count,
            },
            "verdict": verdict,
            "verdict_type": verdict_type,
            "verdict_reason": verdict_reason,
            "warnings": warnings_list,
        },
        "metrics_comparison": {
            "recommended": {
                "top1_pct": round(t1_rec * 100, 2),
                "top1_ci_95": [round(t1_rec_low * 100, 2), round(t1_rec_high * 100, 2)],
                "top3_pct": round(t3_rec * 100, 2),
                "top3_ci_95": [round(t3_rec_low * 100, 2), round(t3_rec_high * 100, 2)],
                "mrr": round(mrr_rec, 4),
                "micro_accuracy_pct": round(micro_acc_rec * 100, 2),
                "macro_accuracy_pct": round(macro_acc_rec * 100, 2),
                "accuracy_when_agree_pct": round(acc_when_agree_rec * 100, 2),
                "accuracy_when_disagree_pct": round(acc_when_disagree_rec * 100, 2),
            },
            "baseline": {
                "top1_pct": round(baseline_top1_acc * 100, 2),
                "top1_ci_95": [round(t1_base_low * 100, 2), round(t1_base_high * 100, 2)],
                "top3_pct": round(baseline_top3_acc * 100, 2),
                "top3_ci_95": [round(t3_base_low * 100, 2), round(t3_base_high * 100, 2)],
                "mrr": round(baseline_mrr, 4),
                "micro_accuracy_pct": round(micro_acc_base * 100, 2),
                "macro_accuracy_pct": round(macro_acc_base * 100, 2),
                "accuracy_when_agree_pct": round(acc_when_agree_base * 100, 2),
                "accuracy_when_disagree_pct": round(acc_when_disagree_base * 100, 2),
            },
            "image_only_accuracy_pct": round(img_only_acc * 100, 2),
            "n_image_only": n_img_only,
            "n_normal_with_desc": N,
        },
        "per_service_accuracy": per_service_stats,
        "cv_stability": cv_stability,
        "plateau_views": plateau_views,
        "confidence_calibration": {
            "best_temperature": best_t,
            "best_ece": round(best_ece, 4),
            "best_ece_pct": round(best_ece * 100, 2),
            "sweep": calibration_results,
            "reliability_table": best_reliability_table,
        },
        "decision_table": decision_table,
        "signal_quality": signal_quality_summary,
        "changed_rows": changed_rows_table,
    }

    # Generate Markdown export
    report_data["markdown_report"] = generate_markdown_report(report_data)

    return report_data


# ── Markdown Report Formatter ──────────────────────────────────────────────────

def generate_markdown_report(rep: dict[str, Any]) -> str:
    """Formats full fusion exploration report into comprehensive Markdown."""
    rec = rep.get("recommendation", {})
    mc = rep.get("metrics_comparison", {})
    calib = rep.get("confidence_calibration", {})
    sig = rep.get("signal_quality", {})
    changed = rep.get("changed_rows", [])

    lines = [
        "# Multimodal Fusion Explorer Report",
        f"**Run ID:** `{rep.get('job_id')}` | **Catalog Hash:** `{rep.get('catalog_hash')}`",
        "",
        "---",
        "",
        "## 1. Recommendation",
        f"- **Verdict:** **`{rec.get('verdict', '').upper()}`**",
        f"- **Formula:** {rec.get('formula_words')}",
        f"- **Equation:**",
        f"  $$\\text{{{rec.get('formula_equation')}}}$$",
        f"- **Recommended Parameters:** `w = {rec.get('parameters', {}).get('w')}`, `wd = {rec.get('parameters', {}).get('wd')}`, `Temperature = {rec.get('parameters', {}).get('temperature')}`",
        f"- **Held-Out Accuracy (5-Fold Repeated CV):** {rec.get('held_out_accuracy', {}).get('recommended_mean_held_out_pct')}% vs Baseline {rec.get('held_out_accuracy', {}).get('baseline_mean_held_out_pct')}% (Δ {rec.get('held_out_accuracy', {}).get('mean_diff_pct'):+}%)",
        f"- **95% Paired Bootstrap CI:** [{rec.get('held_out_accuracy', {}).get('ci_95_bootstrap', [0, 0])[0]:+}%, {rec.get('held_out_accuracy', {}).get('ci_95_bootstrap', [0, 0])[1]:+}%]",
        f"- **Row Flips:** {rec.get('flips', {}).get('wrong_to_right')} wrong->right, {rec.get('flips', {}).get('right_to_wrong')} right->wrong (net {rec.get('flips', {}).get('net_change'):+})",
        f"- **Reasoning:** {rec.get('verdict_reason')}",
        "",
    ]

    warnings = rec.get("warnings") or []
    if warnings:
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"> ⚠️ **Warning:** {w}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Sanity Checks Verification",
        "| Check | Description | Tolerance | Status |",
        "|---|---|---|---|",
    ])
    sc = rep.get("sanity_checks", {}).get("checks", {})
    rep_c = sc.get("replay", {})
    lines.append(f"| **0a. Replay** | Replay live bi-encoder candidate pool formula offline | tolerance 1e-6 | {'✅ PASSED (0 mismatches)' if rep_c.get('passed') else '❌ FAILED'} |")
    mod_c = sc.get("modality_top1", {})
    lines.append(f"| **0b. Modality Top-1** | Text-only & image-only accuracy equality vs stored run | exact | {'✅ PASSED' if mod_c.get('passed') else '❌ FAILED'} |")
    rec_c = sc.get("recombine", {})
    lines.append(f"| **0c. Recombine** | wd*desc_sim + (1-wd)*title_sim == stored text_score | max diff {rec_c.get('max_diff', 0):.6f} | {'✅ PASSED (0 mismatches)' if rec_c.get('passed') else '❌ FAILED'} |")
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Metrics Comparison: Recommended vs Baseline",
        "| Metric | Recommended Fusion | Baseline Live | Difference |",
        "|---|---|---|---|",
        f"| **Top-1 Accuracy** | {mc.get('recommended', {}).get('top1_pct')}% [{mc.get('recommended', {}).get('top1_ci_95', [0,0])[0]}%, {mc.get('recommended', {}).get('top1_ci_95', [0,0])[1]}%] | {mc.get('baseline', {}).get('top1_pct')}% [{mc.get('baseline', {}).get('top1_ci_95', [0,0])[0]}%, {mc.get('baseline', {}).get('top1_ci_95', [0,0])[1]}%] | {round(mc.get('recommended', {}).get('top1_pct', 0) - mc.get('baseline', {}).get('top1_pct', 0), 2):+}% |",
        f"| **Top-3 Accuracy** | {mc.get('recommended', {}).get('top3_pct')}% [{mc.get('recommended', {}).get('top3_ci_95', [0,0])[0]}%, {mc.get('recommended', {}).get('top3_ci_95', [0,0])[1]}%] | {mc.get('baseline', {}).get('top3_pct')}% [{mc.get('baseline', {}).get('top3_ci_95', [0,0])[0]}%, {mc.get('baseline', {}).get('top3_ci_95', [0,0])[1]}%] | {round(mc.get('recommended', {}).get('top3_pct', 0) - mc.get('baseline', {}).get('top3_pct', 0), 2):+}% |",
        f"| **MRR** | {mc.get('recommended', {}).get('mrr')} | {mc.get('baseline', {}).get('mrr')} | {round(mc.get('recommended', {}).get('mrr', 0) - mc.get('baseline', {}).get('mrr', 0), 4):+} |",
        f"| **Micro Accuracy** | {mc.get('recommended', {}).get('micro_accuracy_pct')}% | {mc.get('baseline', {}).get('micro_accuracy_pct')}% | {round(mc.get('recommended', {}).get('micro_accuracy_pct', 0) - mc.get('baseline', {}).get('micro_accuracy_pct', 0), 2):+}% |",
        f"| **Macro Accuracy** | {mc.get('recommended', {}).get('macro_accuracy_pct')}% | {mc.get('baseline', {}).get('macro_accuracy_pct')}% | {round(mc.get('recommended', {}).get('macro_accuracy_pct', 0) - mc.get('baseline', {}).get('macro_accuracy_pct', 0), 2):+}% |",
        f"| **Accuracy when Agree** | {mc.get('recommended', {}).get('accuracy_when_agree_pct')}% | {mc.get('baseline', {}).get('accuracy_when_agree_pct')}% | {round(mc.get('recommended', {}).get('accuracy_when_agree_pct', 0) - mc.get('baseline', {}).get('accuracy_when_agree_pct', 0), 2):+}% |",
        f"| **Accuracy when Disagree** | {mc.get('recommended', {}).get('accuracy_when_disagree_pct')}% | {mc.get('baseline', {}).get('accuracy_when_disagree_pct')}% | {round(mc.get('recommended', {}).get('accuracy_when_disagree_pct', 0) - mc.get('baseline', {}).get('accuracy_when_disagree_pct', 0), 2):+}% |",
        "",
        "---",
        "",
        "## 4. Confidence Calibration Sweep",
        f"**Best Softmax Temperature:** `T = {calib.get('best_temperature')}` | **Minimum ECE:** `{calib.get('best_ece_pct')}%`",
        "",
        "| Temperature T | Expected Calibration Error (ECE) |",
        "|---|---|",
    ])
    for item in calib.get("sweep", []):
        marker = " 🎯 (best)" if item.get("temperature") == calib.get("best_temperature") else ""
        lines.append(f"| {item.get('temperature')} | {item.get('ece_pct')}%{marker} |")
    lines.append("")

    lines.extend([
        "### Reliability Table (at best T)",
        "| Confidence Bin | n | Avg Confidence | Accuracy | Status |",
        "|---|---|---|---|---|",
    ])
    for b in calib.get("reliability_table", []):
        w_flag = "⚠️ n < 30" if b.get("warn_small_sample") else "OK"
        lines.append(f"| {b.get('label')} | {b.get('n')} | {b.get('avg_confidence')}% | {b.get('accuracy')}% | {w_flag} |")
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Signal Quality (AUROC)",
        "| Signal | Task A: Top-1 Correct (n=" + str(sig.get("n_top1_correct_eval")) + ") | Task B: Normal vs NONE (n=" + str(sig.get("n_none_distinction_eval")) + ") |",
        "|---|---|---|",
    ])
    sigs_a = {s["name"]: s.get("auroc") for s in sig.get("signals_top1_correct", [])}
    sigs_b = {s["name"]: s.get("auroc") for s in sig.get("signals_none_distinction", [])}
    all_sig_names = list(sigs_a.keys())
    for name in all_sig_names:
        a_val = f"{sigs_a.get(name):.3f}" if sigs_a.get(name) is not None else "N/A"
        b_val = f"{sigs_b.get(name):.3f}" if sigs_b.get(name) is not None else "N/A"
        lines.append(f"| {name} | {a_val} | {b_val} |")
    lines.append("")

    lines.extend([
        "---",
        "",
        f"## 6. Changed Predictions vs Baseline ({len(changed)} rows)",
        "| Row | Image | Expected | Baseline Pred (conf) | Recommended Pred (conf) | Status |",
        "|---|---|---|---|---|---|",
    ])
    for r in changed:
        st = "🟢 Wrong -> Right" if r.get("change_type") == "wrong_to_right" else ("🔴 Right -> Wrong" if r.get("change_type") == "right_to_wrong" else "⚪ Changed")
        lines.append(
            f"| #{r.get('row_id')} | `{r.get('image')}` | **{r.get('expected_service')}** | {r.get('baseline_pred')} ({round(r.get('baseline_conf', 0)*100, 1)}%) | {r.get('recommended_pred')} ({round(r.get('recommended_conf', 0)*100, 1)}%) | {st} |"
        )
    lines.append("")

    return "\n".join(lines)
