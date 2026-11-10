"""
Model comparison harness for the text embedder (scripts/eval_model_compare.py).
Read-only with respect to the catalog: runs the ground-truth eval as a dry run.

Usage (from the siglip folder, server running on :8083):

  ..\\venv\\Scripts\\python.exe scripts\\eval_model_compare.py run --tag baseline_bge_base
  ..\\venv\\Scripts\\python.exe scripts\\eval_model_compare.py run --tag bge_m3
  ..\\venv\\Scripts\\python.exe scripts\\eval_model_compare.py compare \\
        app/eval/runs/eval_summary_baseline_bge_base_<ts>.json \\
        app/eval/runs/eval_summary_bge_m3_<ts>.json

`run` uploads eval_dataset.zip, starts the eval with the reranker OFF and no
other overrides, waits for completion, then runs the check_text_floor groups
(20 junk, 10 completion) and writes into app/eval/runs/:
  - eval_rows_<tag>_<job_id>_<ts>.csv        per-row results
  - eval_summary_<tag>_<ts>.json             metrics + rows needed for compare
The full job JSON is also saved by the server itself as eval_<job_id>.json.

`compare` prints a side-by-side table of two summaries plus every real row
whose text-only top-1 changed.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings  # noqa: E402
from app.db.connection import create_pool  # noqa: E402
from app.tei_client import TEIClient  # noqa: E402
from scripts.ab_prefix import analyze_run  # noqa: E402
from scripts.check_text_floor import (  # noqa: E402
    COMPLETION_UPDATES,
    UNRELATED_MESSAGES,
    evaluate_descriptions,
)

# Subfolder: the eval job manager prunes *.json directly in app/eval/runs (keeps 20).
SERVER_RUNS_DIR = base_dir / "app" / "eval" / "runs"
RUNS_DIR = SERVER_RUNS_DIR / "model_compare"


def _http_json(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_pipeline_eval(base_url: str, zip_path: Path, timeout_s: float,
                      extra_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    zip_bytes = zip_path.read_bytes()
    boundary = f"----EvalBoundary{int(time.time() * 1000)}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{zip_path.name}"\r\n'
        f"Content-Type: application/zip\r\n\r\n"
    ).encode("utf-8") + zip_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    upload = _http_json(
        f"{base_url}/v1/eval/upload", body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    job_id = upload["job_id"]
    print(f"Created eval job {job_id}")

    _http_json(
        f"{base_url}/v1/eval/jobs/{job_id}/start",
        json.dumps({"enable_reranker": False, **(extra_overrides or {})}).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    while True:
        time.sleep(2.0)
        data = _http_json(f"{base_url}/v1/eval/jobs/{job_id}")
        st = data.get("status")
        if st in ("completed", "stopped", "error"):
            if st != "completed":
                raise RuntimeError(f"Eval job {job_id} ended with status={st}: {data.get('error_message')}")
            print(f"Eval job {job_id} completed in {time.perf_counter() - t0:.1f}s")
            return data
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"Eval job {job_id} exceeded {timeout_s}s")


def _top1_text_score(r: dict[str, Any]) -> float | None:
    pred = r.get("predicted_service")
    s = (r.get("service_scores") or {}).get(pred, {}).get("text_score")
    if s is None:
        s = r.get("top1_text_similarity")
    return float(s) if s is not None else None


def save_rows_csv(run: dict[str, Any], tag: str, ts: str) -> Path:
    path = RUNS_DIR / f"eval_rows_{tag}_{run.get('job_id')}_{ts}.csv"
    fields = ["image", "description", "expected_service", "is_none", "text_top1",
              "full_text_rank", "predicted_service", "correct", "action",
              "top1_text_score", "confidence", "gap", "reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in run.get("rows", []):
            if r.get("is_warmup"):
                continue
            s = _top1_text_score(r)
            w.writerow({
                "image": r.get("image"),
                "description": r.get("description"),
                "expected_service": r.get("expected_service"),
                "is_none": r.get("is_none"),
                "text_top1": r.get("text_top1"),
                "full_text_rank": r.get("full_text_rank"),
                "predicted_service": r.get("predicted_service"),
                "correct": r.get("correct"),
                "action": r.get("action"),
                "top1_text_score": f"{s:.6f}" if s is not None else "",
                "confidence": r.get("confidence"),
                "gap": r.get("gap"),
                "reason": r.get("reason", ""),
            })
    return path


async def run_floor_groups(settings: Any) -> dict[str, Any]:
    pool = create_pool(settings)
    await pool.open()
    tei = TEIClient(settings.TEI_EMBED_URL, settings.TEI_RERANK_URL,
                    api_key=settings.TEI_API_KEY, timeout=30)
    try:
        out: dict[str, Any] = {}
        for name, msgs in (("junk", UNRELATED_MESSAGES), ("completion", COMPLETION_UPDATES)):
            items = await evaluate_descriptions(msgs, pool, tei, settings)
            scores = [it["top1_text_score"] for it in items]
            out[name] = {
                "n": len(items),
                "min": float(np.min(scores)),
                "median": float(np.median(scores)),
                "max": float(np.max(scores)),
                "actions": {a: sum(1 for it in items if it["action"] == a)
                            for a in ("no_action", "draft", "auto_create")},
                "items": [{k: it[k] for k in ("description", "top1_service", "top1_text_score", "action")}
                          for it in items],
            }
        try:
            out["tei_embed_info"] = await tei.get_info("embed")
        except Exception as exc:  # informational only
            out["tei_embed_info"] = {"error": str(exc)}
        return out
    finally:
        await pool.close()
        await tei.close()


def cmd_run(args: argparse.Namespace) -> None:
    settings = get_settings()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    overrides: dict[str, Any] = {}
    if args.text_floor is not None:
        overrides["text_no_action_floor"] = args.text_floor
    if args.text_auto_min is not None:
        overrides["text_auto_min"] = args.text_auto_min
    if args.agree_mode is not None:
        overrides["agree_mode"] = args.agree_mode
    run = run_pipeline_eval(args.base_url, Path(args.zip), args.timeout, overrides)
    run["rows"] = [r for r in run.get("rows", []) if not r.get("is_warmup")]
    csv_path = save_rows_csv(run, args.tag, ts)
    m = analyze_run(run)
    floor = asyncio.run(run_floor_groups(settings))

    rows_min = [{
        "image": r.get("image"),
        "description": r.get("description"),
        "expected_service": r.get("expected_service"),
        "is_none": r.get("is_none"),
        "text_top1": r.get("text_top1"),
        "predicted_service": r.get("predicted_service"),
        "correct": r.get("correct"),
        "action": r.get("action"),
        "top1_text_score": _top1_text_score(r),
    } for r in run["rows"]]

    summary = {
        "tag": args.tag,
        "created_at": ts,
        "job_id": run.get("job_id"),
        "model_ids": (run.get("effective_settings") or {}).get("model_ids"),
        "tei_embed_info": floor.pop("tei_embed_info", None),
        "metrics": {k: v for k, v in m.items() if k != "rows"},
        "floor": floor,
        "rows": rows_min,
    }
    sum_path = RUNS_DIR / f"eval_summary_{args.tag}_{ts}.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_single(summary)
    job_json = SERVER_RUNS_DIR / f"{run.get('job_id')}.json"
    tagged_job_json = RUNS_DIR / f"eval_job_{args.tag}_{run.get('job_id')}.json"
    if job_json.exists():
        tagged_job_json.write_bytes(job_json.read_bytes())
    print(f"\nRows CSV : {csv_path}\nSummary  : {sum_path}\nJob JSON : {tagged_job_json}")


def _metric_lines(s: dict[str, Any]) -> list[tuple[str, str]]:
    m, fl = s["metrics"], s["floor"]
    rd, nd = m["real_dist"], m["none_dist"]
    no = m["none_outcomes"]
    j, c = fl["junk"], fl["completion"]
    auroc = m.get("auroc")
    return [
        ("Text model", str((s.get("model_ids") or {}).get("text"))),
        ("Text-only top-1 (real rows)", f"{m['text_top1_acc']:.2f}%"),
        ("Text-only top-3 (real rows)", f"{m['text_top3_acc']:.2f}%"),
        ("Mean rank of correct service", f"{m['mean_rank']:.2f}"),
        ("Pipeline top-1 (real rows)", f"{m['pipeline_top1_acc']:.2f}%"),
        ("Auto precision", f"{m['auto_precision']:.2f}%"),
        ("Wrong-auto count", str(m["wrong_auto_count"])),
        ("Real rows text_score min/med/max", f"{rd['min']:.4f}/{rd['median']:.4f}/{rd['max']:.4f}"),
        ("NONE rows text_score min/med/max", f"{nd['min']:.4f}/{nd['median']:.4f}/{nd['max']:.4f}"),
        ("NONE actions no_action/draft/auto", f"{no['no_action']}/{no['draft']}/{no['auto_create']}"),
        ("AUROC top1_text_score (real vs NONE)", f"{auroc:.4f}" if auroc is not None else "n/a"),
        ("Junk (20) text_score min/med/max", f"{j['min']:.4f}/{j['median']:.4f}/{j['max']:.4f}"),
        ("Junk actions no_action/draft/auto", f"{j['actions']['no_action']}/{j['actions']['draft']}/{j['actions']['auto_create']}"),
        ("Completion (10) min/med/max", f"{c['min']:.4f}/{c['median']:.4f}/{c['max']:.4f}"),
        ("Completion actions no_action/draft/auto", f"{c['actions']['no_action']}/{c['actions']['draft']}/{c['actions']['auto_create']}"),
    ]


def _print_single(s: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f" EVAL SUMMARY: {s['tag']}  (job {s['job_id']})")
    print("=" * 80)
    for k, v in _metric_lines(s):
        print(f"{k:<42} | {v}")
    print("\nNONE rows:")
    for r in s["metrics"]["none_per_row"]:
        print(f"  {r['image']}: action={r['action']} score={r['score']:.4f}")


def cmd_compare(args: argparse.Namespace) -> None:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    la, lb = _metric_lines(a), _metric_lines(b)
    w = 42
    print("\n" + "=" * 110)
    print(f"{'Metric':<{w}} | {a['tag']:<30} | {b['tag']:<30}")
    print("-" * 110)
    for (k, va), (_, vb) in zip(la, lb):
        print(f"{k:<{w}} | {va:<30} | {vb:<30}")
    print("=" * 110)

    print("\nNONE rows (A vs B):")
    nb = {r["image"]: r for r in b["metrics"]["none_per_row"]}
    for r in a["metrics"]["none_per_row"]:
        o = nb.get(r["image"], {})
        print(f"  {r['image']}: A action={r['action']:<11} score={r['score']:.4f} | "
              f"B action={o.get('action', '-'):<11} score={o.get('score', float('nan')):.4f}")

    rb = {r["image"]: r for r in b["rows"] if not r.get("is_none")}
    changed = [(r, rb[r["image"]]) for r in a["rows"]
               if not r.get("is_none") and r["image"] in rb and r["text_top1"] != rb[r["image"]]["text_top1"]]
    print(f"\nService rows whose text-only top-1 changed: {len(changed)}")
    for ra, rbb in changed:
        exp = ra["expected_service"]
        print("-" * 110)
        print(f"Image       : {ra['image']}")
        print(f"Description : {ra['description']}")
        print(f"Old (A)     : {ra['text_top1']}  [{'correct' if ra['text_top1'] == exp else 'wrong'}]")
        print(f"New (B)     : {rbb['text_top1']}  [{'correct' if rbb['text_top1'] == exp else 'wrong'}]")
        print(f"Correct     : {exp}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--base-url", default="http://localhost:8083")
    r.add_argument("--zip", default=str(base_dir / "eval_dataset.zip"))
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--text-floor", type=float, default=None, help="per-run override of the text no_action floor")
    r.add_argument("--text-auto-min", type=float, default=None, help="per-run override of the text auto_min")
    r.add_argument("--agree-mode", choices=["strict", "balanced", "relaxed"], default=None,
                   help="per-run override of the text/photo agreement strictness")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
