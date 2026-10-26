"""
Async PostgreSQL connection pool management using psycopg[pool].
Pool is opened in FastAPI lifespan and closed on shutdown.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger("siglip.db.connection")

_pool: AsyncConnectionPool | None = None


async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
    """Register pgvector types on every new connection in the pool."""
    await register_vector_async(conn)


def create_pool(settings: "Settings") -> AsyncConnectionPool:
    """Create (but do not open) the async connection pool."""
    return AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        open=False,
        configure=_configure_connection,
        kwargs={
            "connect_timeout": settings.DB_TIMEOUT_SECONDS,
            "options": f"-c search_path={settings.DB_SCHEMA}",
        },
    )


async def open_pool(pool: AsyncConnectionPool) -> None:
    await pool.open()
    logger.info("Database connection pool opened")


async def close_pool(pool: AsyncConnectionPool) -> None:
    await pool.close()
    logger.info("Database connection pool closed")


def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool not initialised")
    return _pool


def set_pool(pool: AsyncConnectionPool) -> None:
    global _pool
    _pool = pool

