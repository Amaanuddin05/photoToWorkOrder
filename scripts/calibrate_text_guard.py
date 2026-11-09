"""
Text-guard calibration for the model TEI serves (scripts/calibrate_text_guard.py). READ-ONLY.

The bi-encoder text guard uses the top-1 TEXT score (w*cos(desc) + (1-w)*cos(title)):
  score < no_action_floor  -> no_action (post ignored)
  score < auto_min         -> capped at draft
Those are absolute cosine values, so they depend on the embedding model. This script
measures the top-1 score distribution of the current TEI model, in memory, on:

  real requests (must NOT be ignored):
    gt_en / gt_ar          62 ground-truth rows (groundTruth.csv / groundTruth_ar.csv)
    holdout_en / _ar       63 held-out rows (eval/holdout_text_queries.csv / eval/calibration_ar.json)
  non-requests (should be ignored or at least not auto-created):
    none_en / none_ar      5 NONE rows of the ground truth
    junk_en / junk_ar      20 unrelated chat messages (check_text_floor.py / calibration_ar.json)
    completion_en / _ar    10 "work done" updates

and prints, for a grid of floors and auto_min values, what would be ignored / auto-eligible.
Recommendation rule (safety first):
  floor    = highest 0.01 step that still keeps EVERY real request (all 4 sets) at least
             0.01 above it (no real request ignored)
  auto_min = lowest 0.01 step at or above the floor where no wrong top-1 real request and no
             non-request reaches it in BOTH languages is not always possible; the script
             prints the trade-off table, the full-pipeline check is done with eval runs.
Nothing is written to the DB or config.

Usage (from the siglip folder, TEI serving the model to calibrate):
  ..\\venv\\Scripts\\python.exe scripts\\calibrate_text_guard.py
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.catalog_text import catalog_texts  # noqa: E402
from app.config import get_settings  # noqa: E402
from scripts.check_text_floor import COMPLETION_UPDATES, UNRELATED_MESSAGES  # noqa: E402
from scripts.eval_text_catalog import TEI, Out, load_catalog, load_holdout  # noqa: E402

OUT_DIR = base_dir / "app" / "eval" / "runs" / "model_compare"


def read_gt(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    real = [{"description": r["description"].strip(), "service": r["service"].strip(), "image": r["image"]}
            for r in rows if r["service"].strip().upper() != "NONE"]
    none = [r["description"].strip() for r in rows if r["service"].strip().upper() == "NONE"]
    return real, none


def main() -> None:
    s = get_settings()
    w = float(s.PIPELINE_TEXT_DESC_WEIGHT)
    tei = TEI(s)
    model = tei.info.get("model_id")
    out = Out()
    out("=" * 100)
    out(f" TEXT-GUARD CALIBRATION (read-only)  model={model}  w={w}")
    out(f" current settings: DECISION_TEXT_NO_ACTION_FLOOR={s.DECISION_TEXT_NO_ACTION_FLOOR}  "
        f"DECISION_TEXT_AUTO_MIN={s.DECISION_TEXT_AUTO_MIN}")
    out("=" * 100)

    cat = load_catalog(s)
    titles = [c["title"] for c in cat]
    texts = [catalog_texts(c["title"], c["description"], c["keywords"]) for c in cat]
    T = tei.embed([t for t, _ in texts])
    D = tei.embed([d for _, d in texts])
    lookup = {t.lower(): i for i, t in enumerate(titles)}

    ar = json.loads((base_dir / "eval" / "calibration_ar.json").read_text(encoding="utf-8"))
    gt_en, none_en = read_gt(base_dir / "groundTruth.csv")
    gt_ar, none_ar = read_gt(base_dir / "groundTruth_ar.csv")
    hold_en = load_holdout()
    hold_ar = [{"description": d, "service": svc} for d, svc in ar["holdout"]]

    real_sets = {"gt_en": gt_en, "gt_ar": gt_ar, "holdout_en": hold_en, "holdout_ar": hold_ar}
    neg_sets = {"none_en": none_en, "none_ar": none_ar, "junk_en": UNRELATED_MESSAGES, "junk_ar": ar["junk"],
                "completion_en": COMPLETION_UPDATES, "completion_ar": ar["completion"]}

    top_n = int(s.PIPELINE_TEXT_TOP_N)
    temp = float(s.PIPELINE_BIENCODER_TEMP)
    w_desc = float(s.PIPELINE_TEXT_DESC_WEIGHT)
    conf_cache: dict[str, np.ndarray] = {}

    def top1(descs: list[str], key: str) -> tuple[np.ndarray, np.ndarray]:
        Q = tei.embed(descs)
        S = w * (Q @ D.T) + (1 - w) * (Q @ T.T)
        # Text-only post confidence, exactly as the engine's bi-encoder path without an
        # image: pool = text top-N, adjusted score = text_score * PIPELINE_TEXT_DESC_WEIGHT,
        # softmax at PIPELINE_BIENCODER_TEMP, confidence = top probability.
        top = -np.sort(-S, axis=1)[:, :top_n] * w_desc / temp
        p = np.exp(top - top.max(axis=1, keepdims=True))
        conf_cache[key] = (p / p.sum(axis=1, keepdims=True))[:, 0]
        return S.max(axis=1), S.argmax(axis=1)

    real = {}
    for name, rows in real_sets.items():
        sc, idx = top1([r["description"] for r in rows], name)
        ok = np.array([idx[i] == lookup[r["service"].lower()] for i, r in enumerate(rows)])
        real[name] = {"scores": sc, "correct": ok, "rows": rows}
    neg = {name: top1(list(msgs), name)[0] for name, msgs in neg_sets.items()}

    def pct(a, q):
        return float(np.percentile(a, q))

    out(f"\n{'group':<15} | {'n':>3} | {'min':>6} | {'p5':>6} | {'median':>6} | {'max':>6} | text top-1 correct")
    out("-" * 80)
    for name, r in real.items():
        a = r["scores"]
        out(f"{name:<15} | {len(a):>3} | {a.min():.4f} | {pct(a, 5):.4f} | {np.median(a):.4f} | {a.max():.4f} | "
            f"{100 * r['correct'].mean():.1f}%")
    for name, a in neg.items():
        out(f"{name:<15} | {len(a):>3} | {a.min():.4f} | {pct(a, 5):.4f} | {np.median(a):.4f} | {a.max():.4f} | (non-request)")

    all_real = np.concatenate([r["scores"] for r in real.values()])
    lowest = sorted(((float(r["scores"][i]), name, r["rows"][i]) for name, r in real.items()
                     for i in range(len(r["rows"]))), key=lambda x: x[0])[:8]
    out("\nLowest-scoring REAL requests (these decide the floor):")
    for sc, name, row in lowest:
        out(f"  {sc:.4f}  {name:<11} {row['service'][:40]:<40} {row['description'][:60]!r}")

    floors = [round(x / 100, 2) for x in range(28, 46)]
    out(f"\nFLOOR sweep: real requests ignored (score < floor) vs non-requests ignored")
    out(f"{'floor':>6} | {'real ignored (of ' + str(len(all_real)) + ')':>22} | " + " | ".join(f"{n:>13}" for n in neg))
    for f in floors:
        ri = int((all_real < f).sum())
        out(f"{f:>6.2f} | {ri:>22} | " + " | ".join(f"{int((a < f).sum()):>6}/{len(a):<6}" for a in neg.values()))

    out(f"\nAUTO_MIN sweep (text-only proxy): real rows auto-eligible (score >= auto_min), of which WRONG top-1;"
        f" non-requests reaching auto_min")
    out(f"{'auto_min':>8} | {'eligible real':>13} | {'wrong top-1 eligible':>20} | {'non-requests >= auto_min':>24}")
    all_ok = np.concatenate([r["correct"] for r in real.values()])
    all_neg = np.concatenate(list(neg.values()))
    for a_min in [round(x / 100, 2) for x in range(36, 56)]:
        elig = all_real >= a_min
        out(f"{a_min:>8.2f} | {int(elig.sum()):>13} | {int((elig & ~all_ok).sum()):>20} | {int((all_neg >= a_min).sum()):>24}")

    # ── Text-only confidence floor (DECISION_NO_ACTION_FLOOR) ──────────────────
    out(f"\nTEXT-ONLY POST CONFIDENCE (no image; pool top-{top_n}, score*{w_desc}, softmax T={temp}); "
        f"current DECISION_NO_ACTION_FLOOR={s.DECISION_NO_ACTION_FLOOR}")
    out(f"{'group':<15} | {'min':>6} | {'p5':>6} | {'median':>6} | {'below current floor':>19}")
    for name in list(real) + list(neg):
        c = conf_cache[name]
        out(f"{name:<15} | {c.min():.4f} | {pct(c, 5):.4f} | {np.median(c):.4f} | "
            f"{int((c < s.DECISION_NO_ACTION_FLOOR).sum()):>8}/{len(c):<10}")
    all_conf_real = np.concatenate([conf_cache[n] for n in real])
    all_conf_neg = np.concatenate([conf_cache[n] for n in neg])
    out(f"\nCONFIDENCE FLOOR sweep (text-only posts): real ignored (of {len(all_conf_real)}) vs non-requests ignored (of {len(all_conf_neg)})")
    for cf in [round(x / 100, 2) for x in range(5, 31)]:
        out(f"  conf_floor {cf:.2f}: real ignored {int((all_conf_real < cf).sum()):>3} | non-requests ignored {int((all_conf_neg < cf).sum()):>3}")
    conf_rec = math.floor((all_conf_real.min() - 0.005) * 100) / 100
    out(f"Lowest real text-only confidence: {all_conf_real.min():.4f} -> a floor of {conf_rec:.2f} keeps every real request.")

    floor_rec = math.floor((all_real.min() - 0.01) * 100) / 100
    out(f"\nRECOMMENDED floor for {model}: {floor_rec:.2f}  (lowest real request {all_real.min():.4f}; "
        f"keeps every real request >= 0.01 above the floor)")
    out("auto_min: choose with full-pipeline eval runs (EN + AR) at that floor; see the sweep above for the trade-off.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", model or "model").strip("_")
    path = OUT_DIR / f"calibrate_text_guard_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
