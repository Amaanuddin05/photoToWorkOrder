"""
Photo-side (SigLIP visual profile) evaluation (scripts/eval_visual_profiles.py). READ-ONLY.

Ranks the labelled eval photos (eval_dataset.zip) against the service visual profiles
exactly like the pipeline (app.pipeline.image_rank.rank_image_against_profiles), in memory:
  current = the stored ServiceImageProfile vectors
  draft   = --override JSON: [{"title": ..., "proposed_visual_line": ...}], encoded like
            app/routers/services.py (f"{title}. {visual_line}", SigLIP text, 64-token limit)

Prints photo top-1 / top-3 accuracy, per-service accuracy, the most common confusions and
SigLIP truncation of draft lines. With saved eval runs it also replays the STRICT agreement
rule (photo top-1 == text top-1) to estimate auto-creates and wrong autos per run; that
estimate reuses each row's saved confidence/gap, the full eval after applying is the real check.

Usage (from the siglip folder):
  ..\\venv\\Scripts\\python.exe scripts\\eval_visual_profiles.py --override catalog\\visual_profiles_draft.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psycopg
import torch
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings  # noqa: E402
from app.model import SigLIPService  # noqa: E402
from app.pipeline.image_rank import rank_image_against_profiles  # noqa: E402
from scripts.eval_text_catalog import Out  # noqa: E402

RUNS = base_dir / "app" / "eval" / "runs" / "model_compare"
REPLAY_RUNS = {"EN bge-base": "main_safety_en", "EN bge-m3": "m3_safety_en", "AR bge-m3": "m3_safety_ar"}
SAFETY = {"Emergency Response & Area Securing", "Emergency Reactive Cleaning (Spills, Vomit, Incidents)"}


def load_profiles(s) -> dict[int, dict]:
    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        conn.read_only = True
        rows = conn.execute(
            """SELECT s.id, s.title, s.description, p."visualLine", p."textUsed", p."textEmb"::text
               FROM "ServiceImageProfile" p JOIN "WoService" s ON s.id = p."serviceId"
               WHERE p."imageModel" = %s AND s."isActive" ORDER BY s.id;""",
            (s.SIGLIP_MODEL_ID,),
        ).fetchall()
        conn.rollback()
    out = {}
    for sid, title, desc, vline, used, emb in rows:
        v = torch.tensor(json.loads(emb), dtype=torch.float32)
        out[sid] = {"id": sid, "title": title, "description": desc or "", "visual_line": vline,
                    "text_used": used, "vector": v / torch.norm(v)}
    return out


def load_photos() -> list[dict]:
    rows = []
    with zipfile.ZipFile(base_dir / "eval_dataset.zip") as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for r in csv.DictReader(io.StringIO(zf.read(name).decode("utf-8-sig"))):
            path = next(n for n in zf.namelist() if n.endswith("/" + r["image"]) or n == r["image"])
            rows.append({"image": r["image"], "service": r["service"].strip(),
                         "pil": Image.open(io.BytesIO(zf.read(path))).convert("RGB")})
    return rows


def rank_all(photos, profiles, siglip) -> dict[str, list[str]]:
    out = {}
    for p in photos:
        top, _ = rank_image_against_profiles(p["pil"], profiles, siglip, top_n=5, return_all_ranks=True)
        out[p["image"]] = [t["service"] for t in top]
    return out


def summarize(out, name, photos, ranks) -> dict:
    real = [p for p in photos if p["service"].upper() != "NONE"]
    t1 = sum(ranks[p["image"]][0] == p["service"] for p in real)
    t3 = sum(p["service"] in ranks[p["image"]][:3] for p in real)
    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    for p in real:
        per[p["service"]][1] += 1
        if ranks[p["image"]][0] == p["service"]:
            per[p["service"]][0] += 1
        else:
            conf[(p["service"], ranks[p["image"]][0])] += 1
    out(f"{name:<8} photo top-1 {100 * t1 / len(real):6.2f}% ({t1}/{len(real)})  top-3 {100 * t3 / len(real):6.2f}% ({t3}/{len(real)})")
    return {"per": per, "conf": conf, "t1": t1}


def replay(out, ranks_by_variant):
    """Strict agreement replay on saved eval runs (estimate)."""
    for run_name, tag in REPLAY_RUNS.items():
        files = glob.glob(str(RUNS / f"eval_job_{tag}_*.json"))
        if not files:
            out(f"  (no saved run {tag})")
            continue
        d = json.load(open(max(files, key=os.path.getmtime), encoding="utf-8"))
        e = d["effective_settings"]
        cells = []
        for vname, ranks in ranks_by_variant.items():
            autos = wrong = 0
            for r in d["rows"]:
                if r.get("is_warmup"):
                    continue
                ss = r["service_scores"]
                t = sorted(((v["text_score"], k) for k, v in ss.items() if v.get("text_score") is not None), reverse=True)
                if not t or r["action"] == "no_action":
                    continue
                top_s, top = t[0]
                if top_s < e["text_auto_min"] or (r["confidence"] or 0) < e["confidence_threshold"] or (r["gap"] or 0) < e["gap_threshold"]:
                    continue
                photo = ranks[r["image"]]
                if photo[0] != top:
                    continue
                if top not in SAFETY and any(k in SAFETY and top_s - s <= 0.02 and k in photo[:3] for s, k in t[1:3]):
                    continue
                autos += 1
                wrong += int(top != r["expected_service"])
            cells.append(f"{vname}: autos {autos:>2}, wrong {wrong}")
        out(f"  {run_name:<12} | " + " | ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--override", type=Path)
    args = ap.parse_args()
    s = get_settings()
    out = Out()
    siglip = SigLIPService(model_id=s.SIGLIP_MODEL_ID, torch_threads=s.SIGLIP_TORCH_THREADS, load_now=True)
    profiles = load_profiles(s)
    photos = load_photos()
    out("=" * 96)
    out(f" PHOTO PROFILE EVAL (read-only)  model={s.SIGLIP_MODEL_ID}  services={len(profiles)}  photos={len(photos)}")
    out("=" * 96)

    variants = {"current": profiles}
    if args.override:
        items = json.loads(args.override.read_text(encoding="utf-8"))
        by_title = {p["title"]: sid for sid, p in profiles.items()}
        draft = {sid: dict(p) for sid, p in profiles.items()}
        texts, sids = [], []
        for it in items:
            sid = by_title.get(it["title"])
            if sid is None:
                raise SystemExit(f"Title not in active profiles: {it['title']!r}")
            if it.get("proposed_visual_line"):
                texts.append(f"{it['title']}. {it['proposed_visual_line'].strip()}")
                sids.append(sid)
        embs, trunc = siglip.encode_texts(texts)
        for sid, txt, emb, tr in zip(sids, texts, embs, trunc):
            v = torch.tensor(emb, dtype=torch.float32)
            draft[sid].update(visual_line=txt.split(". ", 1)[1], text_used=txt, vector=v / torch.norm(v))
            out(f"draft line: {'TRUNCATED ' if tr else ''}{txt}")
        variants["draft"] = draft
        # sanity: re-encoding the stored textUsed reproduces the stored vectors
        sid0 = next(iter(profiles))
        e0, _ = siglip.encode_texts([profiles[sid0]["text_used"]])
        out(f"sanity: re-encoded stored profile cosine = {float(torch.dot(profiles[sid0]['vector'], torch.tensor(e0[0]) / torch.norm(torch.tensor(e0[0])))):.6f}")

    ranks = {name: rank_all(photos, prof, siglip) for name, prof in variants.items()}
    res = {name: summarize(out, name, photos, r) for name, r in ranks.items()}

    out(f"\n{'service':<56} | " + " | ".join(f"{n:>8}" for n in variants))
    for svc in sorted(res["current"]["per"], key=lambda k: res["current"]["per"][k][0] / res["current"]["per"][k][1]):
        out(f"{svc[:56]:<56} | " + " | ".join(f"{res[n]['per'][svc][0]:>4}/{res[n]['per'][svc][1]:<3}" for n in variants))
    for n in variants:
        out(f"\nTop confusions ({n}): " + ", ".join(f"{a[:22]} -> {b[:22]} x{c}" for (a, b), c in res[n]["conf"].most_common(6)))

    out("\nSTRICT-agreement replay on saved eval runs (estimate; the full eval after applying is the real check):")
    replay(out, ranks)

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"eval_visual_profiles_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
