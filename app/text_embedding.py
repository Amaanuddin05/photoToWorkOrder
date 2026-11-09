"""
Model-adaptive text embeddings for the service catalog.

The catalog's text vectors live in "ServiceTextEmbedding", one row per
(service, TEI model), so the service works with WHATEVER model the TEI embedder
serves (e.g. BAAI/bge-base-en-v1.5 768-d or BAAI/bge-m3 1024-d):

  * sync_catalog_embeddings() embeds every service (active and inactive) that has
    no row for the current TEI model, a row with a different dimension, or a row
    whose title/description/keywords changed since it was embedded (contentHash).
    Texts follow catalog_texts(): the title, and "{description}. {keywords}". It runs
    at startup, in the background when /health/ready sees a new TEI model, and
    from scripts/reembed_catalog.py. All writes happen in ONE transaction under
    an advisory lock, so concurrent processes never double-embed.
  * check_text_embedding() is the readiness guard: the store must be complete and
    current for the model TEI serves (and match TEXT_EMBED_MODEL_ID if pinned).
  * embed_catalog_texts() is the one embedding call for catalog writes and
    similarity checks; resolve_query_model() gives the query path the live model.

The original "WoService"."titleEmb"/"descEmb" vector(768) columns are NOT used for
ranking here and are only written when TEI serves LEGACY_TEXT_MODEL_ID with the
columns' dimension (see legacy_columns_writable()).

TEI 1.9 /info has no dimension field, so the real dimension is measured with a
probe /embed call and cached per TEI model id + sha.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg
from psycopg_pool import AsyncConnectionPool

from app.catalog_text import catalog_texts  # single source of the catalog text recipe

logger = logging.getLogger("siglip.text_embedding")

STORE_TABLE = "ServiceTextEmbedding"
MIGRATION_HINT = "Run scripts/migrate_text_embeddings.py to create the ServiceTextEmbedding table."
_SYNC_LOCK_KEY = "siglip:text-embedding-sync"

# (model_id, model_sha, tei sha) -> measured dimension
_dim_cache: dict[tuple[Any, ...], int] = {}
# Models whose store was verified complete and current: model_id -> dim
_ready_models: dict[str, int] = {}


class EmbeddingMismatchError(RuntimeError):
    """The catalog is not (yet) embedded for the model the TEI embedder serves."""

    code = "EMBEDDING_MODEL_MISMATCH"


@dataclass
class TextEmbedStatus:
    ok: bool = False
    message: str = ""
    tei_model_id: str | None = None
    tei_dim: int | None = None
    tei_max_input_length: int | None = None
    tei_max_client_batch_size: int | None = None
    pinned_model_id: str | None = None          # settings.TEXT_EMBED_MODEL_ID (None = any)
    services_total: int | None = None
    services_embedded_current: int | None = None  # rows for this model, right dim, content unchanged
    store_models: dict[str, int] = field(default_factory=dict)  # model -> row count in the store
    legacy_columns_writable: bool | None = None
    sync_in_progress: bool = False
    last_sync: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def content_hash(title: str | None, description: str | None, keywords: str | None = None) -> str:
    """Hash of exactly what gets embedded; a change to title, description or keywords re-embeds."""
    t, d = catalog_texts(title, description, keywords)
    return hashlib.sha256(f"{t}\n{d}".encode("utf-8")).hexdigest()


def _vec(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


async def probe_tei_embedder(tei_client: Any) -> tuple[str | None, int, dict[str, Any]]:
    """Return (model_id, dim, info) for the TEI embed container."""
    info = await tei_client.get_info("embed")
    model_id = info.get("model_id")
    key = (model_id, info.get("model_sha"), info.get("sha"))
    dim = _dim_cache.get(key)
    if dim is None:
        vecs = await tei_client.embed(["dimension probe"])
        dim = len(vecs[0]) if vecs else 0
        _dim_cache[key] = dim
    return model_id, dim, info


async def _store_exists(conn: psycopg.AsyncConnection) -> bool:
    cur = await conn.execute("SELECT to_regclass(%s) IS NOT NULL;", (f'"{STORE_TABLE}"',))
    return bool((await cur.fetchone())[0])


async def legacy_columns_writable(conn: psycopg.AsyncConnection, settings: Any, model_id: str | None, dim: int | None) -> bool:
    """True when the original WoService vector columns can take vectors from this model."""
    from app.db.schema_verify import get_vector_dim

    if not model_id or model_id != getattr(settings, "LEGACY_TEXT_MODEL_ID", None):
        return False
    col_dim = await get_vector_dim(conn, "WoService", "titleEmb")
    return col_dim is None or col_dim == dim


async def sync_catalog_embeddings(
    pool: AsyncConnectionPool,
    tei_client: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Make "ServiceTextEmbedding" complete and current for the model TEI serves.
    Embeds only what is missing/stale (everything with force=True), plain text, no
    prefix, and upserts all rows in one transaction under an advisory lock.
    Returns counts. Raises EmbeddingMismatchError if the store table is missing.
    """
    model_id, dim, info = await probe_tei_embedder(tei_client)
    batch = int(info.get("max_client_batch_size") or 32)

    async with pool.connection() as conn:
        if not await _store_exists(conn):
            raise EmbeddingMismatchError(f'Table "{STORE_TABLE}" does not exist. {MIGRATION_HINT}')
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (_SYNC_LOCK_KEY,))
            cur = await conn.execute('SELECT id, title, description, keywords FROM "WoService" ORDER BY id;')
            services = await cur.fetchall()
            cur = await conn.execute(
                f'SELECT "serviceId", "dim", "contentHash" FROM "{STORE_TABLE}" WHERE "textModel" = %s;',
                (model_id,),
            )
            existing = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

            todo = []
            for sid, title, desc, kw in services:
                h = content_hash(title, desc, kw)
                row = existing.get(sid)
                if force or row is None or row[0] != dim or row[1] != h:
                    t_text, d_text = catalog_texts(title, desc, kw)
                    todo.append((sid, t_text, d_text, h))

            if todo:
                texts = [t for _sid, title, desc, _h in todo for t in (title, desc)]
                vecs = await tei_client.embed_batched(texts, batch)
                if len(vecs) != len(texts) or any(len(v) != dim for v in vecs):
                    raise EmbeddingMismatchError(
                        f"TEI returned {len(vecs)} vectors for {len(texts)} texts or an unexpected dimension "
                        f"(expected {dim}); nothing written."
                    )
                for i, (sid, _title, _desc, h) in enumerate(todo):
                    await conn.execute(
                        f"""
                        INSERT INTO "{STORE_TABLE}"
                            ("serviceId", "textModel", "dim", "titleEmb", "descEmb", "contentHash", "updatedAt")
                        VALUES (%s, %s, %s, %s::vector, %s::vector, %s, now())
                        ON CONFLICT ("serviceId", "textModel") DO UPDATE SET
                            "dim" = EXCLUDED."dim",
                            "titleEmb" = EXCLUDED."titleEmb",
                            "descEmb" = EXCLUDED."descEmb",
                            "contentHash" = EXCLUDED."contentHash",
                            "updatedAt" = now();
                        """,
                        (sid, model_id, dim, _vec(vecs[2 * i]), _vec(vecs[2 * i + 1]), h),
                    )

    result = {
        "model": model_id,
        "dim": dim,
        "services_total": len(services),
        "embedded": len(todo),
        "already_current": len(services) - len(todo),
    }
    _ready_models[model_id] = dim
    logger.info(
        "Text embedding sync for %s (dim=%d): re-embedded %d of %d services (%d already current)",
        model_id, dim, result["embedded"], result["services_total"], result["already_current"],
    )
    return result


async def check_text_embedding(pool: AsyncConnectionPool, tei_client: Any, settings: Any) -> TextEmbedStatus:
    """Live readiness check. Never raises; returns a status with problems listed."""
    st = TextEmbedStatus(pinned_model_id=getattr(settings, "TEXT_EMBED_MODEL_ID", None) or None)

    try:
        st.tei_model_id, st.tei_dim, info = await probe_tei_embedder(tei_client)
        st.tei_max_input_length = info.get("max_input_length")
        st.tei_max_client_batch_size = info.get("max_client_batch_size")
    except Exception as exc:
        st.problems.append(f"TEI embedder /info or probe /embed failed: {exc}")

    if st.tei_model_id and st.pinned_model_id and st.tei_model_id != st.pinned_model_id:
        st.problems.append(
            f"TEI embedder serves '{st.tei_model_id}' but TEXT_EMBED_MODEL_ID pins '{st.pinned_model_id}'."
        )

    try:
        async with pool.connection() as conn:
            if not await _store_exists(conn):
                st.problems.append(f'Table "{STORE_TABLE}" does not exist. {MIGRATION_HINT}')
            else:
                cur = await conn.execute(
                    f'SELECT "textModel", count(*) FROM "{STORE_TABLE}" GROUP BY 1 ORDER BY 1;'
                )
                st.store_models = {r[0]: int(r[1]) for r in await cur.fetchall()}
                cur = await conn.execute('SELECT id, title, description, keywords FROM "WoService";')
                services = await cur.fetchall()
                st.services_total = len(services)
                if st.tei_model_id and st.tei_dim:
                    cur = await conn.execute(
                        f'SELECT "serviceId", "contentHash" FROM "{STORE_TABLE}" WHERE "textModel" = %s AND "dim" = %s;',
                        (st.tei_model_id, st.tei_dim),
                    )
                    rows = dict(await cur.fetchall())
                    st.services_embedded_current = sum(
                        1 for sid, t, d, k in services if rows.get(sid) == content_hash(t, d, k)
                    )
                    missing = st.services_total - st.services_embedded_current
                    if missing:
                        st.problems.append(
                            f"{missing} of {st.services_total} services are not embedded (or are stale) for "
                            f"'{st.tei_model_id}' (dim={st.tei_dim}); the catalog is re-embedded automatically "
                            "(startup / background) or via scripts/reembed_catalog.py."
                        )
                    st.legacy_columns_writable = await legacy_columns_writable(conn, settings, st.tei_model_id, st.tei_dim)
    except Exception as exc:
        st.problems.append(f"Could not read the text embedding store: {exc}")

    st.ok = not st.problems
    if st.ok:
        _ready_models[st.tei_model_id] = st.tei_dim
        st.message = (
            f"Catalog text embeddings are current for '{st.tei_model_id}' (dim={st.tei_dim}, "
            f"{st.services_embedded_current}/{st.services_total} services)."
        )
    else:
        if st.tei_model_id:
            _ready_models.pop(st.tei_model_id, None)
        st.message = "Text embeddings not ready: " + " ".join(st.problems)
    return st


async def embed_catalog_texts(
    pool: AsyncConnectionPool,
    tei_client: Any,
    settings: Any,
    texts: list[str],
) -> tuple[list[list[float]], TextEmbedStatus]:
    """
    The one embedding call for catalog writes and similarity checks (plain text, no
    prefix). Requires the store to be ready for the model TEI serves, and checks the
    length of every returned vector. Raises EmbeddingMismatchError otherwise.
    """
    st = await check_text_embedding(pool, tei_client, settings)
    if not st.ok:
        raise EmbeddingMismatchError(st.message)
    vecs = await tei_client.embed_batched(texts, st.tei_max_client_batch_size or 32)
    bad = [len(v) for v in vecs if len(v) != st.tei_dim]
    if len(vecs) != len(texts) or bad:
        raise EmbeddingMismatchError(
            f"TEI returned {len(vecs)} vectors for {len(texts)} texts with unexpected dims {bad[:3]} "
            f"(expected {st.tei_dim})."
        )
    return vecs, st


async def resolve_query_model(tei_client: Any) -> tuple[str, int]:
    """
    Model id + dim for the per-post query path: what TEI serves right now, provided
    the store was verified complete for it. Raises EmbeddingMismatchError otherwise
    (e.g. TEI was switched and the background re-embed has not finished yet).
    """
    model_id, dim, _info = await probe_tei_embedder(tei_client)
    if _ready_models.get(model_id) != dim:
        raise EmbeddingMismatchError(
            f"Catalog is not embedded for '{model_id}' (dim={dim}) yet; re-embed in progress or required."
        )
    return model_id, dim


def start_background_sync(state: Any) -> bool:
    """Start one background sync task on app.state if none is running. Returns True if started."""
    task = getattr(state, "text_sync_task", None)
    if task is not None and not task.done():
        return False

    async def _run() -> None:
        try:
            state.text_sync_result = await sync_catalog_embeddings(state.db_pool, state.tei_client)
        except Exception as exc:
            logger.error("Background text embedding sync failed: %s", exc)
            state.text_sync_result = {"error": str(exc)}

    state.text_sync_task = asyncio.create_task(_run())
    return True


_rerank_model_cache: dict[str, str] = {}


async def rerank_model_id(tei_client: Any, settings: Any) -> str | None:
    """Reranker model label: RERANK_MODEL_ID if set, else the rerank container's /info model_id."""
    if getattr(settings, "RERANK_MODEL_ID", None):
        return settings.RERANK_MODEL_ID
    url = getattr(tei_client, "rerank_url", "")
    if url not in _rerank_model_cache:
        try:
            _rerank_model_cache[url] = (await tei_client.get_info("rerank")).get("model_id")
        except Exception:
            return None
    return _rerank_model_cache[url]
