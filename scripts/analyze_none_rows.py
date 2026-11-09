"""
Read-only analysis script for ground-truth evaluation and NONE row separation analysis.
Strictly read-only: does not modify pipeline logic, scoring, thresholds, or configuration.

Features:
1. Runs the eval pipeline on the ground-truth set with reranker OFF (via API or offline run file).
2. For every row, exports a CSV with:
   - image
   - label (service or NONE)
   - predicted top-1 service
   - correct (bool)
   - top1_text_score (raw, before softmax)
   - top1_image_sim
   - max_image_sim_over_all_services
   - softmax confidence
   - gap
   - description word count
3. Computes min / median / max for NONE and non-NONE rows across:
   - top1_text_score
   - max_image_sim_over_all_services
   - confidence
   - gap
4. Computes AUROC for separating NONE from non-NONE using:
   - top1_text_score
   - max_image_sim_over_all_services
5. Saves CSV to app/eval/runs/ and prints summary and NONE rows.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_none_rows")


def compute_auroc_rank(y_true: list[int], y_score: list[float]) -> float | None:
    """
    Computes AUROC using Mann-Whitney U / Wilcoxon rank-sum statistic with mid-ranks for ties.
    y_true: binary labels (0 or 1).
    y_score: continuous scores.
    Pure NumPy/Python implementation - no scipy/sklearn dependency.
    """
    if len(y_true) != len(y_score) or not y_true:
        return None
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Sort pairs by score ascending
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


def run_eval_pipeline_via_api(
    base_url: str,
    zip_path: Path,
    enable_reranker: bool = False,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """
    Triggers a fresh evaluation run on the running service with reranker OFF.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset zip not found at: {zip_path}")

    logger.info("Uploading %s to %s/v1/eval/upload...", zip_path.name, base_url)
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    boundary = "----WebKitFormBoundaryAnalysis7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{zip_path.name}"\r\n'
        f"Content-Type: application/zip\r\n\r\n"
    ).encode("utf-8") + zip_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    upload_req = urllib.request.Request(
        f"{base_url}/v1/eval/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(upload_req) as resp:
        upload_res = json.loads(resp.read().decode("utf-8"))

    job_id = upload_res["job_id"]
    logger.info("Created eval job: %s (total images in zip: %s)", job_id, upload_res.get("total_images_in_zip"))

    logger.info("Starting eval job %s with enable_reranker=%s...", job_id, enable_reranker)
    start_payload = json.dumps({
        "enable_reranker": enable_reranker,
        "text_guard_enabled": True,
        "text_no_action_floor": 0.40,
        "text_auto_min": 0.45,
    }).encode("utf-8")
    start_req = urllib.request.Request(
        f"{base_url}/v1/eval/jobs/{job_id}/start",
        data=start_payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(start_req) as resp:
        start_res = json.loads(resp.read().decode("utf-8"))
    logger.info("Job %s started successfully.", job_id)

    # Poll status until completion
    t_start = time.perf_counter()
    last_status = ""
    while True:
        time.sleep(1.0)
        status_req = urllib.request.Request(f"{base_url}/v1/eval/jobs/{job_id}")
        with urllib.request.urlopen(status_req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        status = data.get("status")
        curr = data.get("current_index", 0)
        tot = data.get("total_rows", 0)
        if status != last_status or curr % 10 == 0:
            logger.info("Job %s status: %s (%d/%d rows)", job_id, status, curr, tot)
            last_status = status

        if status in ("completed", "stopped", "error"):
            elapsed = time.perf_counter() - t_start
            logger.info("Job %s finished in %.2fs with status '%s'.", job_id, elapsed, status)
            if status == "error":
                raise RuntimeError(f"Eval job failed with error: {data.get('error_message')}")
            return data

        if time.perf_counter() - t_start > timeout_sec:
            raise TimeoutError(f"Eval job {job_id} exceeded timeout of {timeout_sec}s")


def load_run_from_disk(run_file_or_id: str | Path, runs_dir: Path) -> dict[str, Any]:
    """Loads an existing eval run JSON file from app/eval/runs."""
    target_path = Path(run_file_or_id)
    if not target_path.is_file():
        # Try appending .json or searching in runs_dir
        candidate = runs_dir / f"{target_path.stem}.json"
        if candidate.is_file():
            target_path = candidate
        else:
            raise FileNotFoundError(f"Run file not found: {run_file_or_id} (checked {candidate})")

    logger.info("Loading run data from: %s", target_path)
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_reranker_off_run(runs_dir: Path) -> Path | None:
    """Finds the most recent completed eval run JSON with reranker OFF and full rows."""
    json_files = sorted(runs_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    for p in json_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            eff = data.get("effective_settings", {})
            if eff.get("enable_reranker") is False and len(data.get("rows", [])) > 0:
                return p
        except Exception:
            continue
    return None


def extract_row_data(row: dict[str, Any]) -> dict[str, Any]:
    """
    Extracts the required fields from an evaluated row:
    - image
    - label (service or NONE)
    - predicted top-1 service
    - correct (bool)
    - top1_text_score (raw, before softmax)
    - top1_image_sim
    - max_image_sim_over_all_services
    - softmax confidence
    - gap
    - description word count
    """
    image = row.get("image", "")
    label = row.get("expected_service", "")
    pred = row.get("predicted_service", "")
    correct = bool(row.get("correct", False))
    conf = float(row.get("confidence", 0.0)) if row.get("confidence") is not None else 0.0
    gap = float(row.get("gap", 0.0)) if row.get("gap") is not None else 0.0

    desc = row.get("description") or ""
    word_count = len(desc.strip().split()) if desc.strip() else 0

    svc_scores = row.get("service_scores", {})
    pred_svc_score = svc_scores.get(pred, {})

    # 1. top1_text_score (raw, before softmax)
    top1_text_score = pred_svc_score.get("text_score")
    if top1_text_score is None:
        top1_text_score = row.get("top1_text_similarity")

    # 2. top1_image_sim
    top1_image_sim = pred_svc_score.get("image_sim")
    if top1_image_sim is None:
        top1_image_sim = row.get("top1_image_similarity")
    if top1_image_sim is None and row.get("all_service_image_scores"):
        top1_image_sim = row["all_service_image_scores"].get(pred)

    # 3. max_image_sim_over_all_services
    all_imgs = [
        s.get("image_sim")
        for s in svc_scores.values()
        if s.get("image_sim") is not None
    ]
    if not all_imgs and row.get("all_service_image_scores"):
        all_imgs = [
            v for v in row["all_service_image_scores"].values()
            if v is not None
        ]
    max_image_sim = max(all_imgs) if all_imgs else None

    return {
        "image": image,
        "label": label,
        "predicted_service": pred,
        "correct": correct,
        "top1_text_score": float(top1_text_score) if top1_text_score is not None else None,
        "top1_image_sim": float(top1_image_sim) if top1_image_sim is not None else None,
        "max_image_sim_over_all_services": float(max_image_sim) if max_image_sim is not None else None,
        "confidence": conf,
        "gap": gap,
        "description_word_count": word_count,
        "is_none": bool(row.get("is_none", label.strip().upper() == "NONE")),
        "description": desc,
        "action": row.get("action", ""),
        "reason": row.get("reason", ""),
    }


def compute_distribution_stats(values: list[float | None]) -> dict[str, float | None]:
    valid = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not valid:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "max": float(np.max(valid)),
    }


def analyze_dataset(
    run_data: dict[str, Any],
    runs_dir: Path,
    output_csv: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    rows_raw = run_data.get("rows", [])
    if not rows_raw:
        raise ValueError("Run data contains no evaluated rows.")

    extracted_rows = [extract_row_data(r) for r in rows_raw]

    # Save to CSV
    runs_dir.mkdir(parents=True, exist_ok=True)
    if output_csv is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = run_data.get("job_id", "run")
        output_csv = runs_dir / f"none_analysis_{job_id}_{timestamp_str}.csv"

    fieldnames = [
        "image",
        "label",
        "predicted_top1_service",
        "correct",
        "top1_text_score",
        "top1_image_sim",
        "max_image_sim_over_all_services",
        "softmax_confidence",
        "gap",
        "description_word_count",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in extracted_rows:
            writer.writerow({
                "image": r["image"],
                "label": r["label"],
                "predicted_top1_service": r["predicted_service"],
                "correct": r["correct"],
                "top1_text_score": f"{r['top1_text_score']:.6f}" if r["top1_text_score"] is not None else "",
                "top1_image_sim": f"{r['top1_image_sim']:.6f}" if r["top1_image_sim"] is not None else "",
                "max_image_sim_over_all_services": f"{r['max_image_sim_over_all_services']:.6f}" if r["max_image_sim_over_all_services"] is not None else "",
                "softmax_confidence": f"{r['confidence']:.4f}",
                "gap": f"{r['gap']:.4f}",
                "description_word_count": r["description_word_count"],
            })

    logger.info("Saved CSV with %d rows to: %s", len(extracted_rows), output_csv)

    # Split into NONE and non-NONE rows
    none_rows = [r for r in extracted_rows if r["is_none"]]
    non_none_rows = [r for r in extracted_rows if not r["is_none"]]

    metrics_keys = [
        ("top1_text_score", "top1_text_score (raw, before softmax)"),
        ("max_image_sim_over_all_services", "max_image_sim_over_all_services"),
        ("confidence", "softmax confidence"),
        ("gap", "gap"),
    ]

    stats_none = {}
    stats_non_none = {}

    for key, name in metrics_keys:
        stats_none[key] = compute_distribution_stats([r[key] for r in none_rows])
        stats_non_none[key] = compute_distribution_stats([r[key] for r in non_none_rows])

    # AUROC: separating NONE from non-NONE
    # Standard convention: normal = 1 (positive), NONE = 0 (negative)
    # Since normal rows exhibit higher text and image similarity than unrepresented/NONE requests,
    # AUROC(normal=1, NONE=0) measures P(score(normal) > score(NONE)).
    y_normal = [0 if r["is_none"] else 1 for r in extracted_rows]
    y_none = [1 if r["is_none"] else 0 for r in extracted_rows]

    txt_scores = [r["top1_text_score"] if r["top1_text_score"] is not None else 0.0 for r in extracted_rows]
    img_max_scores = [r["max_image_sim_over_all_services"] if r["max_image_sim_over_all_services"] is not None else 0.0 for r in extracted_rows]

    auroc_txt_normal = compute_auroc_rank(y_normal, txt_scores)
    auroc_img_normal = compute_auroc_rank(y_normal, img_max_scores)

    # Inverted score AUROC (where positive class is NONE=1, inverted score so lower score -> NONE):
    # This equals exactly the discrimination power of the score.
    auroc_txt_none_inverted = compute_auroc_rank(y_none, [-s for s in txt_scores])
    auroc_img_none_inverted = compute_auroc_rank(y_none, [-s for s in img_max_scores])

    results = {
        "output_csv": str(output_csv),
        "total_rows": len(extracted_rows),
        "none_count": len(none_rows),
        "non_none_count": len(non_none_rows),
        "stats_none": stats_none,
        "stats_non_none": stats_non_none,
        "auroc": {
            "top1_text_score": auroc_txt_normal,
            "max_image_sim_over_all_services": auroc_img_normal,
            "top1_text_score_none_inverted": auroc_txt_none_inverted,
            "max_image_sim_none_inverted": auroc_img_none_inverted,
        },
        "none_rows": none_rows,
    }

    return output_csv, results


def print_formatted_summary(res: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("                     EVALUATION PIPELINE ANALYSIS (RERANKER OFF)")
    print("=" * 90)
    print(f"Total Rows Evaluated : {res['total_rows']}")
    print(f"Non-NONE Rows        : {res['non_none_count']}")
    print(f"NONE Rows            : {res['none_count']}")
    print(f"Exported CSV Path    : {res['output_csv']}")
    print("-" * 90)

    print("\n" + "=" * 90)
    print(" 1. DISTRIBUTION METRICS: MIN / MEDIAN / MAX")
    print("=" * 90)

    header = f"{'Group':<14} | {'Metric':<34} | {'Min':>10} | {'Median':>10} | {'Max':>10}"
    print(header)
    print("-" * len(header))

    # NONE rows
    for key, label in [
        ("top1_text_score", "top1_text_score (raw)"),
        ("max_image_sim_over_all_services", "max_image_sim (all svcs)"),
        ("confidence", "softmax confidence"),
        ("gap", "confidence gap"),
    ]:
        s = res["stats_none"][key]
        min_s = f"{s['min']:.4f}" if s["min"] is not None else "N/A"
        med_s = f"{s['median']:.4f}" if s["median"] is not None else "N/A"
        max_s = f"{s['max']:.4f}" if s["max"] is not None else "N/A"
        print(f"{'NONE (n=5)':<14} | {label:<34} | {min_s:>10} | {med_s:>10} | {max_s:>10}")

    print("-" * len(header))

    # Non-NONE rows
    for key, label in [
        ("top1_text_score", "top1_text_score (raw)"),
        ("max_image_sim_over_all_services", "max_image_sim (all svcs)"),
        ("confidence", "softmax confidence"),
        ("gap", "confidence gap"),
    ]:
        s = res["stats_non_none"][key]
        min_s = f"{s['min']:.4f}" if s["min"] is not None else "N/A"
        med_s = f"{s['median']:.4f}" if s["median"] is not None else "N/A"
        max_s = f"{s['max']:.4f}" if s["max"] is not None else "N/A"
        print(f"{'Non-NONE (n=62)':<14} | {label:<34} | {min_s:>10} | {med_s:>10} | {max_s:>10}")

    print("=" * 90)

    print("\n" + "=" * 90)
    print(" 2. AUROC: SEPARATING NONE FROM NON-NONE ROWS")
    print("=" * 90)
    auroc = res["auroc"]
    print(f"AUROC (top1_text_score)                : {auroc['top1_text_score']:.4f}  (P[score(normal) > score(NONE)])")
    print(f"AUROC (max_image_sim_over_all_services): {auroc['max_image_sim_over_all_services']:.4f}  (P[score(normal) > score(NONE)])")
    print("Note: When predicting NONE directly via inverted score (lower score -> NONE), AUROC is identical:")
    print(f"  - Inverted top1_text_score AUROC    : {auroc['top1_text_score_none_inverted']:.4f}")
    print(f"  - Inverted max_image_sim AUROC      : {auroc['max_image_sim_none_inverted']:.4f}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print(" 3. ALL NONE ROWS DETAILS")
    print("=" * 90)
    none_header = (
        f"{'Image':<12} | {'Label':<6} | {'Predicted Top-1 Service':<36} | "
        f"{'Action':<11} | {'TxtScore':>8} | {'Top1Img':>8} | {'MaxImg':>8} | {'Conf':>6} | {'Gap':>6} | {'Words':>5} | {'Correct'}"
    )
    print(none_header)
    print("-" * len(none_header))

    for r in res["none_rows"]:
        txt_s = f"{r['top1_text_score']:.4f}" if r["top1_text_score"] is not None else "N/A"
        top1_img_s = f"{r['top1_image_sim']:.4f}" if r["top1_image_sim"] is not None else "N/A"
        max_img_s = f"{r['max_image_sim_over_all_services']:.4f}" if r["max_image_sim_over_all_services"] is not None else "N/A"
        print(
            f"{r['image']:<12} | {r['label']:<6} | {r['predicted_service'][:36]:<36} | "
            f"{r['action']:<11} | {txt_s:>8} | {top1_img_s:>8} | {max_img_s:>8} | "
            f"{r['confidence']:>6.4f} | {r['gap']:>6.4f} | {r['description_word_count']:>5d} | {r['correct']}"
        )
    print("-" * len(none_header))
    for r in res["none_rows"]:
        print(f"[{r['image']}] Action: {r['action']:<10} | Correct: {str(r['correct']):<5} | Reason: {r.get('reason')}")
        print(f"             Description: \"{r['description']}\"")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ground-truth evaluation and NONE row separation.")
    parser.add_argument("--base-url", default="http://localhost:8083", help="Base URL of running siglip service")
    parser.add_argument("--zip-path", default="eval_dataset.zip", help="Path to ground truth eval_dataset.zip")
    parser.add_argument("--run-file", default=None, help="Path or Job ID of existing run JSON in app/eval/runs")
    parser.add_argument("--output-csv", default=None, help="Custom output CSV path")
    parser.add_argument("--force-fresh", action="store_true", help="Force triggering a fresh run via API even if run file exists")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    runs_dir = base_dir / "app" / "eval" / "runs"
    zip_path = Path(args.zip_path)
    if not zip_path.is_absolute():
        zip_path = base_dir / zip_path

    run_data: dict[str, Any] | None = None

    # Step 1: Check if specific run file requested
    if args.run_file and not args.force_fresh:
        run_data = load_run_from_disk(args.run_file, runs_dir)

    # Step 2: Try running via live API
    if run_data is None:
        try:
            # Check service health first
            with urllib.request.urlopen(f"{args.base_url}/health/ready", timeout=3) as resp:
                if resp.status == 200:
                    logger.info("Live service detected at %s. Triggering eval pipeline with reranker OFF...", args.base_url)
                    run_data = run_eval_pipeline_via_api(args.base_url, zip_path, enable_reranker=False)
        except Exception as e:
            logger.warning("Could not run via live API (%s): %s", args.base_url, e)

    # Step 3: Fallback to latest saved reranker-off run if API unreachable
    if run_data is None:
        latest_run = find_latest_reranker_off_run(runs_dir)
        if latest_run:
            logger.info("Falling back to latest completed reranker-off run: %s", latest_run.name)
            run_data = load_run_from_disk(latest_run, runs_dir)
        else:
            logger.error("No live service available and no existing reranker-off run found in %s", runs_dir)
            sys.exit(1)

    # Step 4: Extract fields, generate CSV, calculate stats & AUROC
    out_csv = Path(args.output_csv) if args.output_csv else None
    csv_path, res = analyze_dataset(run_data, runs_dir, out_csv)

    # Step 5: Print formatted summary and NONE rows
    print_formatted_summary(res)


if __name__ == "__main__":
    main()

