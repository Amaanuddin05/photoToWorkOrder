"""
Schema migration for the per-model text embedding store (scripts/migrate_text_embeddings.py).

Applies sql/001_service_text_embedding.sql (CREATE TABLE IF NOT EXISTS "ServiceTextEmbedding").
Idempotent: safe to run repeatedly. Changes nothing else: the original
"WoService"."titleEmb"/"descEmb" columns and "ServiceImageProfile" are not touched.

The rows are filled by the service itself (startup / when the TEI model changes) or by
scripts/reembed_catalog.py.

Usage (from the siglip folder; needs a DB role allowed to CREATE TABLE):
  ..\\venv\\Scripts\\python.exe scripts\\migrate_text_embeddings.py --dry-run
  ..\\venv\\Scripts\\python.exe scripts\\migrate_text_embeddings.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings  # noqa: E402

SQL_FILES = ["001_service_text_embedding.sql"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show what would be applied, change nothing")
    args = ap.parse_args()

    s = get_settings()
    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        exists = conn.execute("""SELECT to_regclass('"ServiceTextEmbedding"') IS NOT NULL;""").fetchone()[0]
        print(f'"ServiceTextEmbedding" exists: {exists}')
        if args.dry_run:
            print("DRY RUN - would apply (idempotent): " + ", ".join(SQL_FILES))
            return
        with conn.transaction():
            for name in SQL_FILES:
                conn.execute((base_dir / "sql" / name).read_text(encoding="utf-8"))
                print(f"Applied {name}")
        rows = conn.execute(
            """SELECT "textModel", "dim", count(*) FROM "ServiceTextEmbedding" GROUP BY 1, 2 ORDER BY 1;"""
        ).fetchall()
        print(f"Store contents (model, dim, rows): {rows or 'empty'}")


if __name__ == "__main__":
    main()
