"""
Re-embed the service catalog for the model TEI serves (scripts/reembed_catalog.py).

Manual trigger of the same sync the service runs automatically at startup and when
the TEI model changes (app.text_embedding.sync_catalog_embeddings):
  * embeds title and description of ALL services (active and inactive) that have no
    "ServiceTextEmbedding" row for the current TEI model, a row with a different
    dimension, or a row whose title/description changed since it was embedded;
  * --force re-embeds every service for the current model;
  * plain text, no prefix; all rows written in ONE transaction under an advisory lock;
  * idempotent: re-running with the same model embeds nothing.
Does NOT touch the original "WoService"."titleEmb"/"descEmb" columns or ServiceImageProfile.

Usage (from the siglip folder):
  ..\\venv\\Scripts\\python.exe scripts\\reembed_catalog.py [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings  # noqa: E402
from app.db.connection import create_pool  # noqa: E402
from app.tei_client import TEIClient  # noqa: E402
from app.text_embedding import check_text_embedding, sync_catalog_embeddings  # noqa: E402


async def main_async(force: bool) -> None:
    s = get_settings()
    tei = TEIClient(s.TEI_EMBED_URL, s.TEI_RERANK_URL, api_key=s.TEI_API_KEY, timeout=120.0)
    pool = create_pool(s)
    await pool.open()
    try:
        pinned = s.TEXT_EMBED_MODEL_ID
        info = await tei.get_info("embed")
        if pinned and info.get("model_id") != pinned:
            raise SystemExit(
                f"ABORT: TEI serves '{info.get('model_id')}' but TEXT_EMBED_MODEL_ID pins '{pinned}'. Nothing written."
            )
        result = await sync_catalog_embeddings(pool, tei, force=force)
        st = await check_text_embedding(pool, tei, s)
        print("RE-EMBED COMPLETE (single transaction)")
        for k, v in result.items():
            print(f"  {k:<16}: {v}")
        print(f"  store models    : {st.store_models}")
        print(f"  ready           : {st.ok} - {st.message}")
    finally:
        await pool.close()
        await tei.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-embed every service for the current TEI model")
    asyncio.run(main_async(ap.parse_args().force))


if __name__ == "__main__":
    main()
