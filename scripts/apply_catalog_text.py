"""
Apply reviewed catalog text to WoService (scripts/apply_catalog_text.py). WRITES TO THE DB.

Input: a JSON list like catalog/catalog_text_draft.json:
  [{"title": ..., "proposed_description": ..., "proposed_keywords": ...}, ...]
  (proposed_* missing or null = keep current; titles are never changed)

Steps:
  1. Saves the current text of the affected services to backups/catalog_text_before_<ts>.json
     (that file is itself a valid input: re-running with it restores the old text).
  2. Updates "WoService".description / keywords (+ "updatedAt") in ONE transaction.
  3. Re-embeds the original English vectors "titleEmb"/"descEmb" for the changed services
     with the catalog recipe (title, "{description}. {keywords}") — requires TEI to serve
     BAAI/bge-base-en-v1.5 with the columns' dimension; done in the same transaction, so
     text and vectors never disagree. The multilingual branch re-embeds its own per-model
     store automatically (content hash changes).
ServiceImageProfile (SigLIP) is not touched.

Usage (from the siglip folder; English TEI running):
  ..\\venv\\Scripts\\python.exe scripts\\apply_catalog_text.py --file catalog\\catalog_text_approved.json --dry-run
  ..\\venv\\Scripts\\python.exe scripts\\apply_catalog_text.py --file catalog\\catalog_text_approved.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import httpx
import psycopg

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.catalog_text import catalog_texts  # noqa: E402
from app.config import get_settings  # noqa: E402

LEGACY_MODEL = "BAAI/bge-base-en-v1.5"


def _vec(v) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    items = json.loads(args.file.read_text(encoding="utf-8"))

    headers = {"Authorization": f"Bearer {s.TEI_API_KEY}"} if s.TEI_API_KEY else {}
    tei = httpx.Client(base_url=s.TEI_EMBED_URL.rstrip("/"), headers=headers, timeout=120.0)
    info = tei.get("/info").raise_for_status().json()

    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        col_dim = conn.execute(
            """SELECT atttypmod FROM pg_attribute WHERE attrelid = '"WoService"'::regclass AND attname = 'descEmb';"""
        ).fetchone()[0]
        probe_dim = len(tei.post("/embed", json={"inputs": ["dimension probe"]}).raise_for_status().json()[0])
        if info.get("model_id") != LEGACY_MODEL or probe_dim != col_dim:
            raise SystemExit(
                f"ABORT: TEI serves {info.get('model_id')} (dim {probe_dim}); the original columns need "
                f"{LEGACY_MODEL} (dim {col_dim}). Start the English TEI first. Nothing changed."
            )

        current = {
            r[1]: {"id": r[0], "title": r[1], "description": r[2], "keywords": r[3]}
            for r in conn.execute('SELECT id, title, description, keywords FROM "WoService";').fetchall()
        }
        changes = []
        for it in items:
            c = current.get(it["title"])
            if c is None:
                raise SystemExit(f"ABORT: title not found in WoService: {it['title']!r}. Nothing changed.")
            new_d = it.get("proposed_description")
            new_k = it.get("proposed_keywords")
            new_d = c["description"] if new_d is None else new_d.strip()
            new_k = c["keywords"] if new_k is None else (new_k.strip() or None)
            if not new_d:
                raise SystemExit(f"ABORT: empty description for {it['title']!r}. Nothing changed.")
            if new_d != c["description"] or new_k != c["keywords"]:
                changes.append((c, new_d, new_k))

        print(f"TEI: {info.get('model_id')} (dim {probe_dim}); services to update: {len(changes)}")
        for c, new_d, new_k in changes:
            print(f"\n  [{c['id']}] {c['title']}")
            print(f"     description: {c['description']!r}\n               -> {new_d!r}")
            if new_k != c["keywords"]:
                print(f"     keywords   : {c['keywords']!r}\n               -> {new_k!r}")
        if args.dry_run or not changes:
            print("\nDRY RUN - nothing written." if args.dry_run else "\nNothing to change.")
            return

        texts = [x for c, d, k in changes for x in catalog_texts(c["title"], d, k)]
        vecs = tei.post("/embed", json={"inputs": texts}).raise_for_status().json()
        if len(vecs) != len(texts) or any(len(v) != col_dim for v in vecs):
            raise SystemExit("ABORT: unexpected vectors from TEI. Nothing changed.")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = base_dir / "backups" / f"catalog_text_before_{ts}.json"
        backup.parent.mkdir(exist_ok=True)
        backup.write_text(json.dumps(
            [{"title": c["title"], "proposed_description": c["description"], "proposed_keywords": c["keywords"]}
             for c, _d, _k in changes], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved current text (restore file): {backup}")

        with conn.transaction():
            for i, (c, new_d, new_k) in enumerate(changes):
                cur = conn.execute(
                    """UPDATE "WoService"
                       SET description = %s, keywords = %s,
                           "titleEmb" = %s::vector, "descEmb" = %s::vector, "embeddingModel" = %s,
                           "updatedAt" = now()
                       WHERE id = %s;""",
                    (new_d, new_k, _vec(vecs[2 * i]), _vec(vecs[2 * i + 1]), LEGACY_MODEL, c["id"]),
                )
                assert cur.rowcount == 1
        print(f"Updated {len(changes)} service(s) and their English vectors in one transaction.")


if __name__ == "__main__":
    main()
