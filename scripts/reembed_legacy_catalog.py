"""
Re-embed the original WoService text vectors with the catalog recipe (scripts/reembed_legacy_catalog.py).

"WoService"."titleEmb" / "descEmb" are vector(768) BAAI/bge-base-en-v1.5 vectors built as
  titleEmb <- title,  descEmb <- f"{description}. {keywords}"   (app/catalog_text.py)
Services last written by siglip before the recipe fix were embedded from the plain
description; this script finds every service whose stored vectors differ from the recipe
(cosine < 0.9999) and re-embeds only those, in ONE transaction.

  * Refuses to run unless TEI serves BAAI/bge-base-en-v1.5 with the columns' dimension.
  * Idempotent: a second run changes nothing.
  * --dry-run lists what would change. ServiceImageProfile is never touched.

Usage (from the siglip folder):
  ..\\venv\\Scripts\\python.exe scripts\\reembed_legacy_catalog.py --dry-run
  ..\\venv\\Scripts\\python.exe scripts\\reembed_legacy_catalog.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import psycopg

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.catalog_text import catalog_texts  # noqa: E402
from app.config import get_settings  # noqa: E402

LEGACY_MODEL = "BAAI/bge-base-en-v1.5"
SAME = 0.9999


def _vec(v) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    s = get_settings()

    headers = {"Authorization": f"Bearer {s.TEI_API_KEY}"} if s.TEI_API_KEY else {}
    tei = httpx.Client(base_url=s.TEI_EMBED_URL.rstrip("/"), headers=headers, timeout=120.0)
    info = tei.get("/info").raise_for_status().json()
    batch = int(info.get("max_client_batch_size") or 32)

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            out.extend(tei.post("/embed", json={"inputs": texts[i:i + batch]}).raise_for_status().json())
        return np.array(out, dtype=np.float64)

    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        col_dim = conn.execute(
            """SELECT atttypmod FROM pg_attribute WHERE attrelid = '"WoService"'::regclass AND attname = 'descEmb';"""
        ).fetchone()[0]
        probe_dim = len(embed(["dimension probe"])[0])
        print(f"TEI model: {info.get('model_id')} (dim={probe_dim}); WoService columns: vector({col_dim})")
        if info.get("model_id") != LEGACY_MODEL or probe_dim != col_dim:
            raise SystemExit(f"ABORT: TEI must serve {LEGACY_MODEL} with dim {col_dim}. Nothing changed.")

        rows = conn.execute(
            'SELECT id, title, description, keywords, "titleEmb"::text, "descEmb"::text FROM "WoService" ORDER BY id;'
        ).fetchall()
        texts = [x for r in rows for x in catalog_texts(r[1], r[2], r[3])]
        fresh = embed(texts)

        todo = []
        for i, r in enumerate(rows):
            t_new, d_new = fresh[2 * i], fresh[2 * i + 1]
            t_old = np.array(json.loads(r[4])) if r[4] else None
            d_old = np.array(json.loads(r[5])) if r[5] else None
            ct = float(t_old @ t_new) if t_old is not None else -1.0
            cd = float(d_old @ d_new) if d_old is not None else -1.0
            if ct < SAME or cd < SAME:
                todo.append((r[0], r[1], ct, cd, t_new, d_new))

        print(f"Services: {len(rows)}; differing from the recipe: {len(todo)}")
        for sid, title, ct, cd, _t, _d in todo:
            print(f"  id={sid:<3} cos(title)={ct:.4f} cos(desc)={cd:.4f}  {title}")
        if args.dry_run or not todo:
            print("DRY RUN - nothing written." if args.dry_run else "Nothing to do.")
            return

        with conn.transaction():
            for sid, _title, _ct, _cd, t_new, d_new in todo:
                cur = conn.execute(
                    'UPDATE "WoService" SET "titleEmb" = %s::vector, "descEmb" = %s::vector, "embeddingModel" = %s WHERE id = %s;',
                    (_vec(t_new), _vec(d_new), LEGACY_MODEL, sid),
                )
                assert cur.rowcount == 1
        print(f"Re-embedded {len(todo)} service(s) in one transaction: {[t[0] for t in todo]}")


if __name__ == "__main__":
    main()
