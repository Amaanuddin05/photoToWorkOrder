"""
Apply reviewed SigLIP visual lines to ServiceImageProfile (scripts/apply_visual_profiles.py). WRITES TO THE DB.

Input: a JSON list like catalog/visual_profiles_draft.json:
  [{"title": ..., "proposed_visual_line": ...}, ...]

For each service: textUsed = f"{title}. {visual_line}" (same as app/routers/services.py),
re-encoded with the in-process SigLIP text encoder (64-token limit; truncation is reported),
and "visualLine", "textUsed", "textEmb", "truncated", "updatedAt" are updated for the current
SIGLIP_MODEL_ID in ONE transaction. The current lines are saved first to
backups/visual_profiles_before_<ts>.json, which is itself a valid input (restore file).
WoService and the text embeddings are not touched. Restart the service afterwards so it reloads
the profiles.

Usage (from the siglip folder):
  ..\\venv\\Scripts\\python.exe scripts\\apply_visual_profiles.py --file catalog\\visual_profiles_draft.json --dry-run
  ..\\venv\\Scripts\\python.exe scripts\\apply_visual_profiles.py --file catalog\\visual_profiles_draft.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import psycopg

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings  # noqa: E402
from app.model import SigLIPService  # noqa: E402


def _vec(v) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    items = json.loads(args.file.read_text(encoding="utf-8"))

    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        rows = conn.execute(
            """SELECT s.id, s.title, p."visualLine" FROM "ServiceImageProfile" p
               JOIN "WoService" s ON s.id = p."serviceId" WHERE p."imageModel" = %s;""",
            (s.SIGLIP_MODEL_ID,),
        ).fetchall()
        current = {r[1]: {"id": r[0], "visual_line": r[2]} for r in rows}
        changes = []
        for it in items:
            c = current.get(it["title"])
            if c is None:
                raise SystemExit(f"ABORT: no {s.SIGLIP_MODEL_ID} profile for {it['title']!r}. Nothing changed.")
            new = (it.get("proposed_visual_line") or "").strip()
            if new and new != (c["visual_line"] or ""):
                changes.append((it["title"], c, new))

        print(f"SigLIP model: {s.SIGLIP_MODEL_ID}; profiles to update: {len(changes)}")
        for title, c, new in changes:
            print(f"\n  {title}\n     {c['visual_line']!r}\n  -> {new!r}")
        if args.dry_run or not changes:
            print("\nDRY RUN - nothing written." if args.dry_run else "\nNothing to change.")
            return

        siglip = SigLIPService(model_id=s.SIGLIP_MODEL_ID, torch_threads=s.SIGLIP_TORCH_THREADS, load_now=True)
        texts = [f"{title}. {new}" for title, _c, new in changes]
        embs, trunc = siglip.encode_texts(texts)
        for t, tr in zip(texts, trunc):
            if tr:
                print(f"  WARNING: truncated to 64 SigLIP tokens: {t!r}")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = base_dir / "backups" / f"visual_profiles_before_{ts}.json"
        backup.parent.mkdir(exist_ok=True)
        backup.write_text(json.dumps([{"title": title, "proposed_visual_line": c["visual_line"]}
                                      for title, c, _n in changes], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved current lines (restore file): {backup}")

        with conn.transaction():
            for (title, c, new), txt, emb, tr in zip(changes, texts, embs, trunc):
                cur = conn.execute(
                    """UPDATE "ServiceImageProfile"
                       SET "visualLine" = %s, "textUsed" = %s, "textEmb" = %s::vector,
                           "truncated" = %s, "updatedAt" = now()
                       WHERE "serviceId" = %s AND "imageModel" = %s;""",
                    (new, txt, _vec(emb), bool(tr), c["id"], s.SIGLIP_MODEL_ID),
                )
                assert cur.rowcount == 1
        print(f"Updated {len(changes)} profile(s) in one transaction. Restart the service to reload profiles.")


if __name__ == "__main__":
    main()
