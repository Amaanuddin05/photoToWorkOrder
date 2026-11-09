"""
Text-only catalog evaluation (scripts/eval_text_catalog.py). READ-ONLY.

Scores the catalog text against labelled requests with the model the TEI embedder serves,
entirely in memory: catalog text is read from the DB (SELECT only) and embedded with the
catalog recipe (app/catalog_text.py: title, "{description}. {keywords}"); nothing is written.

Test sets:
  gt       62 real rows of ground_truth.csv in eval_dataset.zip (NONE rows skipped)
  holdout  eval/holdout_text_queries.csv (63 requests, 3 per service, written separately)

Metrics per set: text-only top-1 / top-3 / mean rank of the correct service with
text_score = w * cos(q, desc) + (1 - w) * cos(q, title), w = PIPELINE_TEXT_DESC_WEIGHT.

--override <json> scores a draft catalog text next to the current one (no DB change).
The JSON is a list of {"title": ..., "proposed_description": ..., "proposed_keywords": ...}
(proposed_* missing or null = keep current). Prints both, the delta, and every row whose
prediction changed.

Usage (from the siglip folder):
  ..\\venv\\Scripts\\python.exe scripts\\eval_text_catalog.py
  ..\\venv\\Scripts\\python.exe scripts\\eval_text_catalog.py --override catalog\\catalog_text_draft.json
Output is also saved to app/eval/runs/model_compare/eval_text_catalog_<model>_<ts>.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import psycopg

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.catalog_text import catalog_texts  # noqa: E402
from app.config import get_settings  # noqa: E402

OUT_DIR = base_dir / "app" / "eval" / "runs" / "model_compare"


class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        self.lines.append(line)
        enc = sys.stdout.encoding or "utf-8"
        print(line.encode(enc, "replace").decode(enc, "replace"))


def load_gt() -> list[dict]:
    with zipfile.ZipFile(base_dir / "eval_dataset.zip") as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("utf-8-sig")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        svc, desc = (r.get("service") or "").strip(), (r.get("description") or "").strip()
        if svc and svc.upper() != "NONE" and desc:
            rows.append({"description": desc, "service": svc})
    return rows


def load_holdout() -> list[dict]:
    with open(base_dir / "eval" / "holdout_text_queries.csv", encoding="utf-8") as f:
        return [{"description": r["description"].strip(), "service": r["service"].strip()} for r in csv.DictReader(f)]


def load_catalog(s) -> list[dict]:
    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        conn.read_only = True
        rows = conn.execute(
            'SELECT id, title, description, keywords FROM "WoService" WHERE "isActive" ORDER BY id;'
        ).fetchall()
        conn.rollback()
    return [{"id": r[0], "title": r[1], "description": r[2] or "", "keywords": r[3]} for r in rows]


def apply_override(cat: list[dict], path: Path) -> tuple[list[dict], list[str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    by_title = {c["title"]: dict(c) for c in cat}
    changed = []
    for it in items:
        t = it["title"]
        if t not in by_title:
            raise SystemExit(f"Override title not in active catalog: {t!r}")
        c = by_title[t]
        if it.get("proposed_description") is not None and it["proposed_description"] != c["description"]:
            c["description"] = it["proposed_description"]
            changed.append(t)
        if it.get("proposed_keywords") is not None and it["proposed_keywords"] != c["keywords"]:
            c["keywords"] = it["proposed_keywords"]
            if t not in changed:
                changed.append(t)
    return [by_title[c["title"]] for c in cat], changed


class TEI:
    def __init__(self, s):
        h = {"Authorization": f"Bearer {s.TEI_API_KEY}"} if s.TEI_API_KEY else {}
        self.c = httpx.Client(base_url=s.TEI_EMBED_URL.rstrip("/"), headers=h, timeout=120.0)
        self.info = self.c.get("/info").raise_for_status().json()
        self.batch = int(self.info.get("max_client_batch_size") or 32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            out.extend(self.c.post("/embed", json={"inputs": texts[i:i + self.batch]}).raise_for_status().json())
        m = np.array(out, dtype=np.float64)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def evaluate(tei: TEI, cat: list[dict], rows: list[dict], w: float, qvecs: np.ndarray) -> dict:
    titles = [c["title"] for c in cat]
    texts = [catalog_texts(c["title"], c["description"], c["keywords"]) for c in cat]
    T = tei.embed([t for t, _ in texts])
    D = tei.embed([d for _, d in texts])
    S = w * (qvecs @ D.T) + (1 - w) * (qvecs @ T.T)
    lookup = {t.lower(): i for i, t in enumerate(titles)}
    ranks, preds = [], []
    for i, r in enumerate(rows):
        g = lookup[r["service"].lower()]
        order = np.argsort(-S[i], kind="stable")
        ranks.append(int(np.where(order == g)[0][0]) + 1)
        preds.append(titles[int(order[0])])
    n = len(rows)
    return {
        "top1": 100.0 * sum(x == 1 for x in ranks) / n,
        "top3": 100.0 * sum(x <= 3 for x in ranks) / n,
        "mean_rank": float(np.mean(ranks)),
        "preds": preds,
        "hubs": Counter(p for p, r in zip(preds, rows) if p != r["service"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--override", type=Path, help="draft catalog text JSON to score next to the current text")
    args = ap.parse_args()

    s = get_settings()
    w = float(s.PIPELINE_TEXT_DESC_WEIGHT)
    tei = TEI(s)
    model = tei.info.get("model_id")
    out = Out()
    out("=" * 96)
    out(f" TEXT-ONLY CATALOG EVAL (read-only)  model={model}  w={w}")
    out("=" * 96)

    cat = load_catalog(s)
    variants = [("current", cat)]
    if args.override:
        new_cat, changed = apply_override(cat, args.override)
        variants.append(("draft", new_cat))
        out(f"Override: {args.override}  ({len(changed)} services changed: {', '.join(changed)})")

    sets = [("gt (62 test rows)", load_gt()), ("holdout (63 new rows)", load_holdout())]
    results: dict = {}
    for set_name, rows in sets:
        q = tei.embed([r["description"] for r in rows])
        for vname, vcat in variants:
            results[(set_name, vname)] = evaluate(tei, vcat, rows, w, q)

    out(f"\n{'set':<24} | {'variant':<8} | {'top-1':>7} | {'top-3':>7} | {'mean rank':>9}")
    out("-" * 68)
    for set_name, _rows in sets:
        for vname, _ in variants:
            r = results[(set_name, vname)]
            out(f"{set_name:<24} | {vname:<8} | {r['top1']:>6.2f}% | {r['top3']:>6.2f}% | {r['mean_rank']:>9.2f}")
        if len(variants) == 2:
            a, b = results[(set_name, "current")], results[(set_name, "draft")]
            out(f"{'':<24} | {'delta':<8} | {b['top1'] - a['top1']:>+6.2f}  | {b['top3'] - a['top3']:>+6.2f}  | {b['mean_rank'] - a['mean_rank']:>+9.2f}")

    for set_name, rows in sets:
        for vname, _ in variants:
            hubs = results[(set_name, vname)]["hubs"].most_common(4)
            out(f"\nWrong top-1 landed on ({set_name}, {vname}): " + (", ".join(f"{h}={n}" for h, n in hubs) or "none"))
        if len(variants) == 2:
            a, b = results[(set_name, "current")]["preds"], results[(set_name, "draft")]["preds"]
            changed_rows = [(r, x, y) for r, x, y in zip(rows, a, b) if x != y]
            out(f"Rows whose prediction changed ({set_name}): {len(changed_rows)}")
            for r, x, y in changed_rows:
                mark = "FIXED " if y == r["service"] else ("BROKE " if x == r["service"] else "other ")
                out(f"  {mark} {r['description'][:70]!r}: {x} -> {y}  (correct: {r['service']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", model or "model").strip("_")
    path = OUT_DIR / f"eval_text_catalog_{tag}{'_draft' if args.override else ''}_{ts}.txt"
    path.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
