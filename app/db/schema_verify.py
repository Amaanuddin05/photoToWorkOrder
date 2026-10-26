"""
Startup schema verification.
Raises ConfigurationError (stops startup) on missing tables/columns or dimension mismatch.
Never raises on connectivity failure (those are retried in lifespan).
"""

from __future__ import annotations

import logging

import psycopg

logger = logging.getLogger("siglip.db.schema_verify")


class ConfigurationError(RuntimeError):
    """Fatal misconfiguration — service must not start."""


REQUIRED_TABLES: dict[str, list[str]] = {
    "WoService": [
        "id", "title", "description", "isActive",
        "embeddingModel", "updatedAt", "titleEmb", "descEmb",
    ],
    "ServiceImageProfile": [
        "serviceId", "imageModel", "visualLine",
        "textUsed", "truncated", "textEmb", "updatedAt",
    ],
    "DecisionLog": [
        "id", "createdAt", "description", "photoRef", "photoHash",
        "location", "textTop", "imageTop", "pool", "rerankScores",
        "topServiceId", "confidence", "gap", "agree",
        "action", "wouldBeAction", "reason", "models",
        "configVersion", "timings", "finalServiceId", "finalAt",
        "poolContainsFinal",
    ],
}

# pgvector atttypmod directly encodes dims; -1 means unconstrained
_EXPECTED_TEXT_DIM = 768


async def verify_schema(conn: psycopg.AsyncConnection) -> dict:
    """
    Verify required tables and columns exist.
    Returns a summary dict on success.
    Raises ConfigurationError on fatal mismatch.
    """
    # 1. Check pgvector extension
    row = await conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
    )
    vec_row = await row.fetchone()
    if not vec_row:
        raise ConfigurationError(
            "pgvector extension is not installed in the database. "
            "Run: CREATE EXTENSION vector;"
        )
    pgvector_version = vec_row[0]

    # 2. Check required tables and columns
    errors: list[str] = []
    for table, required_cols in REQUIRED_TABLES.items():
        cur = await conn.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s;
            """,
            (table,),
        )
        live_cols = {r[0] for r in await cur.fetchall()}
        if not live_cols:
            errors.append(f'Table "{table}" does not exist')
            continue
        missing = [c for c in required_cols if c not in live_cols]
        if missing:
            errors.append(
                f'Table "{table}" missing columns: {missing}'
            )

    if errors:
        raise ConfigurationError(
            "Database schema verification failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # 3. Verify vector dimensions on WoService embedding columns
    title_dim = await _get_vector_dim(conn, "WoService", "titleEmb")
    desc_dim  = await _get_vector_dim(conn, "WoService", "descEmb")

    dim_errors = []
    if title_dim is not None and title_dim != _EXPECTED_TEXT_DIM:
        dim_errors.append(
            f'"WoService"."titleEmb" has dim={title_dim}, expected {_EXPECTED_TEXT_DIM}'
        )
    if desc_dim is not None and desc_dim != _EXPECTED_TEXT_DIM:
        dim_errors.append(
            f'"WoService"."descEmb" has dim={desc_dim}, expected {_EXPECTED_TEXT_DIM}'
        )
    if dim_errors:
        raise ConfigurationError(
            "Vector dimension mismatch:\n" + "\n".join(f"  - {e}" for e in dim_errors)
        )

    logger.info(
        "Schema verification passed",
        extra={
            "pgvector_version": pgvector_version,
            "title_dim": title_dim,
            "desc_dim": desc_dim,
        },
    )
    return {
        "pgvector_version": pgvector_version,
        "title_dim": title_dim,
        "desc_dim": desc_dim,
    }


async def _get_vector_dim(
    conn: psycopg.AsyncConnection, table: str, column: str
) -> int | None:
    """
    Return the declared vector dimension for a column, or None if unconstrained.
    In pgvector, atttypmod directly stores the dimension (or -1 if unconstrained).
    """
    cur = await conn.execute(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = %s::regclass
          AND attname = %s
          AND NOT attisdropped;
        """,
        (f'"{table}"', column),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    typmod = row[0]
    if typmod == -1:
        return None   # unconstrained vector
    return typmod  # pgvector stores dim directly in typmod

