"""
A/B Evaluation Script for BGE Query Prefix (scripts/ab_prefix.py).
Read-only analysis; no pipeline changes.

Compares:
  A = current prefix ("Represent this sentence for searching relevant passages: ")
  B = empty prefix ("")

Evaluates:
  a. Text-only ranking (62 service rows): top-1, top-3 accuracy, and mean rank of the correct service.
  b. Final pipeline result at current settings (62 rows): top-1 accuracy, auto precision, wrong-auto count, and NONE outcomes.
  c. Raw top-1 text_score distribution (min / median / max) for 62 real rows and 5 NONE rows, plus AUROC.
  d. check_text_floor.py groups (20 junk, 10 completion): min / median / max score and action breakdown.
  e. Any service rows whose text-only top-1 changed between A and B.

Saves per-row CSVs in app/eval/runs/ for both runs with the prefix recorded in the file name.
"""

from __future__ import annotations

import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# Use SelectorEventLoop on Windows for psycopg async compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings
from app.db import repository as repo
from app.db.connection import create_pool
from app.pipeline.decision import make_decision, softmax_with_temperature
from app.pipeline.pool import build_candidate_pool
from app.tei_client import TEIClient
from scripts.check_text_floor import UNRELATED_MESSAGES, COMPLETION_UPDATES, evaluate_descriptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ab_prefix")

CURRENT_PREFIX = "Represent this sentence for searching relevant passages: "
EMPTY_PREFIX = ""


def compute_auroc_rank(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) != len(y_score) or not y_true:
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

    rank_sum_pos = sum(ranks[idx] for idx, (_, label) in enumerate(pairs) if label == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def run_pipeline_eval(base_url: str, zip_path: Path, prefix_override: str) -> dict[str, Any]:
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    boundary = f"----WebKitFormBoundary{int(time.time()*1000)}"
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
    logger.info("Created eval job %s for prefix='%s'", job_id, prefix_override)

    start_payload = json.dumps({
        "enable_reranker": False,
        "text_guard_enabled": True,
        "text_no_action_floor": 0.40,
        "text_auto_min": 0.45,
        "text_query_prefix": prefix_override,
    }).encode("utf-8")

    start_req = urllib.request.Request(
        f"{base_url}/v1/eval/jobs/{job_id}/start",
        data=start_payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(start_req) as resp:
        start_res = json.loads(resp.read().decode("utf-8"))

    t_start = time.perf_counter()
    while True:
        time.sleep(1.0)
        status_req = urllib.request.Request(f"{base_url}/v1/eval/jobs/{job_id}")
        with urllib.request.urlopen(status_req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        status = data.get("status")
        if status in ("completed", "stopped", "error"):
            if status == "error":
                raise RuntimeError(f"Eval job failed: {data.get('error_message')}")
            return data
        if time.perf_counter() - t_start > 180:
            raise TimeoutError("Eval job exceeded 180s")


def save_run_csv(run_data: dict[str, Any], tag: str, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    job_id = run_data.get("job_id", "run")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_dir / f"eval_ab_{tag}_{job_id}_{timestamp}.csv"

    fieldnames = [
        "image", "label", "predicted_top1_service", "correct",
        "top1_text_score", "top1_image_sim", "max_image_sim_over_all_services",
        "softmax_confidence", "gap", "description_word_count", "text_top1",
        "text_pool_rank", "full_text_rank", "action", "reason"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run_data.get("rows", []):
            pred = r.get("predicted_service")
            svc_scores = r.get("service_scores", {})
            pred_scores = svc_scores.get(pred, {})
            t_score = pred_scores.get("text_score")
            if t_score is None:
                t_score = r.get("top1_text_similarity")

            img_sim = pred_scores.get("image_sim")
            if img_sim is None:
                img_sim = r.get("top1_image_similarity")

            all_imgs = [s.get("image_sim") for s in svc_scores.values() if s.get("image_sim") is not None]
            max_img = max(all_imgs) if all_imgs else None

            desc = r.get("description") or ""
            words = len(desc.strip().split()) if desc.strip() else 0

            writer.writerow({
                "image": r.get("image"),
                "label": r.get("expected_service"),
                "predicted_top1_service": pred,
                "correct": r.get("correct"),
                "top1_text_score": f"{t_score:.6f}" if t_score is not None else "",
                "top1_image_sim": f"{img_sim:.6f}" if img_sim is not None else "",
                "max_image_sim_over_all_services": f"{max_img:.6f}" if max_img is not None else "",
                "softmax_confidence": f"{r.get('confidence', 0.0):.4f}",
                "gap": f"{r.get('gap', 0.0):.4f}",
                "description_word_count": words,
                "text_top1": r.get("text_top1"),
                "text_pool_rank": r.get("text_pool_rank"),
                "full_text_rank": r.get("full_text_rank"),
                "action": r.get("action"),
                "reason": r.get("reason", ""),
            })

    logger.info("Saved %s CSV to %s", tag, csv_path)
    return csv_path


def analyze_run(run_data: dict[str, Any]) -> dict[str, Any]:
    rows = run_data.get("rows", [])
    real_rows = [r for r in rows if not r.get("is_none")]
    none_rows = [r for r in rows if r.get("is_none")]

    # a. Text-only ranking on 62 real rows
    text_top1_correct = sum(1 for r in real_rows if r.get("text_top1") == r.get("expected_service"))
    text_top1_acc = text_top1_correct / len(real_rows) * 100.0 if real_rows else 0.0

    # Top-3 in text pool: text_pool_rank in [1, 2, 3] or full_text_rank in [1, 2, 3]
    text_top3_correct = 0
    ranks = []
    for r in real_rows:
        t_rank = r.get("text_pool_rank")
        f_rank = r.get("full_text_rank")
        effective_rank = t_rank if t_rank is not None else f_rank
        if effective_rank is not None:
            ranks.append(effective_rank)
            if effective_rank <= 3:
                text_top3_correct += 1
        else:
            # If not ranked at all, treat as 22 (beyond catalog of 21)
            ranks.append(22.0)

    text_top3_acc = text_top3_correct / len(real_rows) * 100.0 if real_rows else 0.0
    mean_rank = float(np.mean(ranks)) if ranks else 0.0

    # b. Final pipeline result
    pipeline_top1_correct = sum(1 for r in real_rows if r.get("correct"))
    pipeline_top1_acc = pipeline_top1_correct / len(real_rows) * 100.0 if real_rows else 0.0

    auto_rows = [r for r in rows if r.get("action") == "auto_create"]
    auto_correct = sum(1 for r in auto_rows if r.get("correct"))
    auto_prec = (auto_correct / len(auto_rows) * 100.0) if auto_rows else 0.0
    wrong_auto_count = len(auto_rows) - auto_correct

    none_outcomes = {
        "no_action": sum(1 for r in none_rows if r.get("action") == "no_action"),
        "draft": sum(1 for r in none_rows if r.get("action") == "draft"),
        "auto_create": sum(1 for r in none_rows if r.get("action") == "auto_create"),
    }

    # c. Raw top-1 text_score distribution
    def extract_t_score(r):
        pred = r.get("predicted_service")
        s = r.get("service_scores", {}).get(pred, {}).get("text_score")
        if s is None:
            s = r.get("top1_text_similarity")
        return float(s) if s is not None else 0.0

    real_scores = [extract_t_score(r) for r in real_rows]
    none_scores = [extract_t_score(r) for r in none_rows]

    real_dist = {
        "min": float(np.min(real_scores)) if real_scores else 0.0,
        "median": float(np.median(real_scores)) if real_scores else 0.0,
        "max": float(np.max(real_scores)) if real_scores else 0.0,
    }
    none_dist = {
        "min": float(np.min(none_scores)) if none_scores else 0.0,
        "median": float(np.median(none_scores)) if none_scores else 0.0,
        "max": float(np.max(none_scores)) if none_scores else 0.0,
    }

    # AUROC for separating NONE from real
    all_scores = [extract_t_score(r) for r in rows]
    y_normal = [0 if r.get("is_none") else 1 for r in rows]
    auroc = compute_auroc_rank(y_normal, all_scores)

    return {
        "text_top1_acc": text_top1_acc,
        "text_top3_acc": text_top3_acc,
        "mean_rank": mean_rank,
        "pipeline_top1_acc": pipeline_top1_acc,
        "auto_precision": auto_prec,
        "wrong_auto_count": wrong_auto_count,
        "none_outcomes": none_outcomes,
        "none_per_row": [{ "image": r.get("image"), "action": r.get("action"), "score": extract_t_score(r) } for r in none_rows],
        "real_dist": real_dist,
        "none_dist": none_dist,
        "auroc": auroc,
        "rows": rows,
    }


async def run_floor_checks(prefix: str, settings: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    # Clone settings with the prefix override
    test_settings = settings.model_copy(update={"TEXT_QUERY_PREFIX": prefix})
    pool = create_pool(test_settings)
    await pool.open()
    tei_client = TEIClient(
        test_settings.TEI_EMBED_URL,
        test_settings.TEI_RERANK_URL,
        api_key=test_settings.TEI_API_KEY,
        timeout=10,
    )
    try:
        res_unrelated = await evaluate_descriptions(UNRELATED_MESSAGES, pool, tei_client, test_settings)
        res_completion = await evaluate_descriptions(COMPLETION_UPDATES, pool, tei_client, test_settings)

        def summarize_group(items):
            scores = [it["top1_text_score"] for it in items]
            actions = {
                "no_action": sum(1 for it in items if it["action"] == "no_action"),
                "draft": sum(1 for it in items if it["action"] == "draft"),
                "auto_create": sum(1 for it in items if it["action"] == "auto_create"),
            }
            return {
                "min": float(np.min(scores)),
                "median": float(np.median(scores)),
                "max": float(np.max(scores)),
                "actions": actions,
            }

        return summarize_group(res_unrelated), summarize_group(res_completion)
    finally:
        await pool.close()
        await tei_client.close()


def main() -> None:
    settings = get_settings()
    base_url = "http://localhost:8083"
    zip_path = base_dir / "eval_dataset.zip"
    runs_dir = base_dir / "app" / "eval" / "runs"

    logger.info("Starting A/B Query Prefix Evaluation...")
    logger.info("A = Current Prefix: '%s'", CURRENT_PREFIX)
    logger.info("B = Empty Prefix:   '%s'", EMPTY_PREFIX)

    # 1. Run Pipeline Eval A
    logger.info("--- Executing Run A (Current Prefix) ---")
    data_a = run_pipeline_eval(base_url, zip_path, CURRENT_PREFIX)
    csv_a = save_run_csv(data_a, "prefix_current", runs_dir)
    res_a = analyze_run(data_a)

    # 2. Run Pipeline Eval B
    logger.info("--- Executing Run B (Empty Prefix) ---")
    data_b = run_pipeline_eval(base_url, zip_path, EMPTY_PREFIX)
    csv_b = save_run_csv(data_b, "prefix_empty", runs_dir)
    res_b = analyze_run(data_b)

    # 3. Run Text Floor Checks
    logger.info("--- Executing Text Floor Checks for A and B ---")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    junk_a, comp_a = loop.run_until_complete(run_floor_checks(CURRENT_PREFIX, settings))
    junk_b, comp_b = loop.run_until_complete(run_floor_checks(EMPTY_PREFIX, settings))

    # 4. Find service rows whose text-only top-1 changed
    rows_a_map = {r.get("image"): r for r in res_a["rows"] if not r.get("is_none")}
    rows_b_map = {r.get("image"): r for r in res_b["rows"] if not r.get("is_none")}

    changed_text_top1 = []
    for img, r_a in rows_a_map.items():
        r_b = rows_b_map.get(img)
        if r_b and r_a.get("text_top1") != r_b.get("text_top1"):
            changed_text_top1.append({
                "image": img,
                "description": r_a.get("description"),
                "expected": r_a.get("expected_service"),
                "text_top1_a": r_a.get("text_top1"),
                "text_top1_b": r_b.get("text_top1"),
                "correct_a": r_a.get("text_top1") == r_a.get("expected_service"),
                "correct_b": r_b.get("text_top1") == r_b.get("expected_service"),
            })

    # 5. Print Comparison Table
    print("\n" + "=" * 92)
    print("                      A/B TEST: BGE QUERY PREFIX COMPARISON")
    print("=" * 92)
    print(f"{'Metric':<48} | {'Run A (Current Prefix)':<19} | {'Run B (Empty Prefix)':<18}")
    print("-" * 92)

    # Section a: Text-only ranking
    print("--- [a] Text-Only Ranking (62 service rows) ---")
    print(f"{'Text-Only Top-1 Accuracy':<48} | {res_a['text_top1_acc']:>17.2f}% | {res_b['text_top1_acc']:>16.2f}%")
    print(f"{'Text-Only Top-3 Accuracy':<48} | {res_a['text_top3_acc']:>17.2f}% | {res_b['text_top3_acc']:>16.2f}%")
    print(f"{'Mean Rank of Correct Service':<48} | {res_a['mean_rank']:>18.2f}  | {res_b['mean_rank']:>17.2f} ")

    # Section b: Final pipeline result
    print("\n--- [b] Final Pipeline Result (Current settings, reranker OFF) ---")
    print(f"{'Pipeline Top-1 Accuracy (62 real rows)':<48} | {res_a['pipeline_top1_acc']:>17.2f}% | {res_b['pipeline_top1_acc']:>16.2f}%")
    print(f"{'Auto Precision':<48} | {res_a['auto_precision']:>17.2f}% | {res_b['auto_precision']:>16.2f}%")
    print(f"{'Wrong-Auto Count':<48} | {res_a['wrong_auto_count']:>18d}  | {res_b['wrong_auto_count']:>17d} ")
    print(f"{'NONE: no_action (correct)':<48} | {res_a['none_outcomes']['no_action']:>18d}  | {res_b['none_outcomes']['no_action']:>17d} ")
    print(f"{'NONE: draft (soft false match)':<48} | {res_a['none_outcomes']['draft']:>18d}  | {res_b['none_outcomes']['draft']:>17d} ")
    print(f"{'NONE: auto_create (harmful false match)':<48} | {res_a['none_outcomes']['auto_create']:>18d}  | {res_b['none_outcomes']['auto_create']:>17d} ")

    # Section c: Top-1 text_score distribution & AUROC
    print("\n--- [c] Raw Top-1 Text Score Distributions & Discrimination ---")
    ra_d = res_a['real_dist']
    rb_d = res_b['real_dist']
    print(f"{'Real Rows text_score Min':<48} | {ra_d['min']:>18.4f}  | {rb_d['min']:>17.4f} ")
    print(f"{'Real Rows text_score Median':<48} | {ra_d['median']:>18.4f}  | {rb_d['median']:>17.4f} ")
    print(f"{'Real Rows text_score Max':<48} | {ra_d['max']:>18.4f}  | {rb_d['max']:>17.4f} ")

    na_d = res_a['none_dist']
    nb_d = res_b['none_dist']
    print(f"{'NONE Rows text_score Min':<48} | {na_d['min']:>18.4f}  | {nb_d['min']:>17.4f} ")
    print(f"{'NONE Rows text_score Median':<48} | {na_d['median']:>18.4f}  | {nb_d['median']:>17.4f} ")
    print(f"{'NONE Rows text_score Max':<48} | {na_d['max']:>18.4f}  | {nb_d['max']:>17.4f} ")
    print(f"{'AUROC (top1_text_score: Normal vs NONE)':<48} | {res_a['auroc']:>18.4f}  | {res_b['auroc']:>17.4f} ")

    # Section d: Floor checks
    print("\n--- [d] Text-Only Junk & Completion Floor Checks ---")
    print(f"{'Junk (20 msgs) Min / Median / Max':<48} | {junk_a['min']:.4f}/{junk_a['median']:.4f}/{junk_a['max']:.4f}  | {junk_b['min']:.4f}/{junk_b['median']:.4f}/{junk_b['max']:.4f} ")
    print(f"{'Junk Actions (no_action / draft / auto)':<48} | {junk_a['actions']['no_action']}/{junk_a['actions']['draft']}/{junk_a['actions']['auto_create']:>15}  | {junk_b['actions']['no_action']}/{junk_b['actions']['draft']}/{junk_b['actions']['auto_create']:>14} ")
    print(f"{'Completion (10 msgs) Min / Median / Max':<48} | {comp_a['min']:.4f}/{comp_a['median']:.4f}/{comp_a['max']:.4f}  | {comp_b['min']:.4f}/{comp_b['median']:.4f}/{comp_b['max']:.4f} ")
    print(f"{'Completion Actions (no_action / draft / auto)':<48} | {comp_a['actions']['no_action']}/{comp_a['actions']['draft']}/{comp_a['actions']['auto_create']:>15}  | {comp_b['actions']['no_action']}/{comp_b['actions']['draft']}/{comp_b['actions']['auto_create']:>14} ")
    print("=" * 92)

    # NONE rows individual breakdown
    print("\n--- NONE Rows Detailed Comparison ---")
    for ra_n, rb_n in zip(res_a['none_per_row'], res_b['none_per_row']):
        print(f"[{ra_n['image']}] A: action={ra_n['action']:<10} (score={ra_n['score']:.4f})  |  B: action={rb_n['action']:<10} (score={rb_n['score']:.4f})")

    # Section e: Changed text-only top-1 rows
    print("\n" + "=" * 92)
    print(f"--- [e] Service Rows with Changed Text-Only Top-1 (Total: {len(changed_text_top1)}) ---")
    print("=" * 92)
    if not changed_text_top1:
        print("No service rows changed their text-only top-1 prediction between Run A and Run B.")
    else:
        for item in changed_text_top1:
            print(f"Image       : {item['image']}")
            print(f"Description : \"{item['description']}\"")
            print(f"Expected    : {item['expected']}")
            print(f"A Top-1     : {item['text_top1_a']} (Correct: {item['correct_a']})")
            print(f"B Top-1     : {item['text_top1_b']} (Correct: {item['correct_b']})")
            print("-" * 92)

    print(f"\nArtifact CSV Run A: {csv_a}")
    print(f"Artifact CSV Run B: {csv_b}\n")


if __name__ == "__main__":
    main()

