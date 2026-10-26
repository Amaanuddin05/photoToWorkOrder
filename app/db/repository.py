"""
All database read/write operations for the siglip service.
Uses the async connection pool. All identifiers are double-quoted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("siglip.db.repository")

# ── Text search SQL (identical to WoSemanticSearch/app/search.py) ─────────────
RANKING_SQL = """
    SELECT id, title, description,
           1 - ("titleEmb" <=> %(q)s::vector) AS title_sim,
           1 - ("descEmb"  <=> %(q)s::vector) AS desc_sim,
           %(w)s * (1 - ("descEmb"  <=> %(q)s::vector))
             + (1 - %(w)s) * (1 - ("titleEmb" <=> %(q)s::vector)) AS score
    FROM "WoService"
    WHERE "isActive"
    ORDER BY score DESC
    LIMIT %(top_n)s;
"""


async def text_search(
    pool: AsyncConnectionPool,
    query_vector: list[float],
    desc_weight: float,
    top_n: int,
) -> list[dict[str, Any]]:
    """Cosine similarity search against WoService title/desc embeddings."""
    vec_str = "[" + ",".join(str(v) for v in query_vector) + "]"
    async with pool.connection() as conn:
        cur = await conn.execute(
            RANKING_SQL,
            {"q": vec_str, "w": desc_weight, "top_n": top_n},
        )
        rows = await cur.fetchall()

    return [
        {
            "service": r[1],
            "service_id": r[0],
            "description": r[2],
            "rank": idx,
            "score": round(float(r[5]), 4),
            "title_sim": round(float(r[3]), 4),
            "desc_sim": round(float(r[4]), 4),
        }
        for idx, r in enumerate(rows, start=1)
    ]


async def get_all_service_profiles(
    pool: AsyncConnectionPool,
    image_model: str,
) -> dict[int, dict[str, Any]]:
    """
    Load ServiceImageProfile rows into memory at startup / after catalog upsert.
    Returns {service_id: profile_dict} matching photoToWorkOrder's load_cached_service_profiles.
    """
    import numpy as np
    import torch

    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT s.id, s.title, s.description,
                   p."visualLine", p."textUsed", p."textEmb"
            FROM "ServiceImageProfile" p
            JOIN "WoService" s ON p."serviceId" = s.id
            WHERE p."imageModel" = %s AND s."isActive"
            ORDER BY s.id;
            """,
            (image_model,),
        )
        rows = await cur.fetchall()

    profiles: dict[int, dict[str, Any]] = {}
    for r in rows:
        svc_id, title, desc, visual_line, text_used, text_emb = r
        tensor_vec = None
        if text_emb is not None and visual_line and visual_line.strip():
            if hasattr(text_emb, "to_numpy"):
                arr = text_emb.to_numpy().astype(np.float32)
            elif hasattr(text_emb, "to_list"):
                arr = np.asarray(text_emb.to_list(), dtype=np.float32)
            elif hasattr(text_emb, "tolist"):
                arr = np.asarray(text_emb.tolist(), dtype=np.float32)
            elif isinstance(text_emb, str):
                clean = text_emb.strip("[]")
                arr = np.fromstring(clean, sep=",", dtype=np.float32)
            else:
                arr = np.asarray(text_emb, dtype=np.float32)
            tensor_vec = torch.from_numpy(arr)
            norm = torch.norm(tensor_vec, p=2)
            if norm > 0:
                tensor_vec = tensor_vec / norm

        profiles[svc_id] = {
            "id": svc_id,
            "title": title,
            "description": desc or "",
            "visual_line": visual_line if (visual_line and visual_line.strip()) else None,
            "text_used": text_used,
            "vector": tensor_vec,
        }
    return profiles


async def insert_decision_log(
    pool: AsyncConnectionPool,
    data: dict[str, Any],
) -> int:
    """
    Insert a row into DecisionLog inside a transaction.
    If data contains 'idempotency_key', acquires pg_advisory_xact_lock first
    to prevent concurrent duplicate inserts with the same key.
    Returns the new row id.
    """
    idem_key = data.get("idempotency_key")

    async with pool.connection() as conn:
        async with conn.transaction():
            if idem_key:
                # Serialise concurrent requests with same key
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s));",
                    (str(idem_key),),
                )
                # Double-check: another request may have inserted while we waited
                existing = await _find_by_idempotency_key_in_txn(conn, str(idem_key))
                if existing is not None:
                    return existing["id"]

            models_val = {
                "text": data.get("text_model", ""),
                "image": data.get("image_model", ""),
                "reranker": data.get("reranker_model", ""),
            }
            if idem_key:
                models_val["idempotency_key"] = idem_key

            cur = await conn.execute(
                """
                INSERT INTO "DecisionLog" (
                    description, "photoRef", "photoHash", location,
                    "textTop", "imageTop", pool, "rerankScores",
                    "topServiceId", confidence, gap, agree,
                    action, "wouldBeAction", reason, models,
                    "configVersion", timings
                ) VALUES (
                    %(description)s, %(photo_ref)s, %(photo_hash)s, %(location)s,
                    %(text_top)s, %(image_top)s, %(pool)s, %(rerank_scores)s,
                    %(top_service_id)s, %(confidence)s, %(gap)s, %(agree)s,
                    %(action)s, %(would_be_action)s, %(reason)s, %(models)s,
                    %(config_version)s, %(timings)s
                ) RETURNING id;
                """,
                {
                    "description": data.get("description") if data.get("store_description", True) else None,
                    "photo_ref": data.get("photo_ref"),
                    "photo_hash": data.get("photo_hash"),
                    "location": data.get("location"),
                    "text_top": json.dumps(data["text_top"]) if data.get("text_top") else None,
                    "image_top": json.dumps(data["image_top"]) if data.get("image_top") else None,
                    "pool": json.dumps(data["pool"]) if data.get("pool") else None,
                    "rerank_scores": json.dumps(data["rerank_scores"]) if data.get("rerank_scores") else None,
                    "top_service_id": data.get("top_service_id"),
                    "confidence": data.get("confidence"),
                    "gap": data.get("gap"),
                    "agree": data.get("agree"),
                    "action": data["action"],
                    "would_be_action": data["would_be_action"],
                    "reason": data.get("reason"),
                    "models": json.dumps(models_val),
                    "config_version": data.get("config_version", ""),
                    "timings": json.dumps(data.get("timings", {})),
                },
            )
            row = await cur.fetchone()
            return row[0]


async def _find_by_idempotency_key_in_txn(
    conn, key: str
) -> dict[str, Any] | None:
    """Used inside a transaction to check for duplicate idempotency keys."""
    cur = await conn.execute(
        "SELECT id FROM \"DecisionLog\" WHERE models->>'idempotency_key' = %s ORDER BY id DESC LIMIT 1;",
        (key,),
    )
    row = await cur.fetchone()
    if row:
        return {"id": row[0]}
    return None


async def get_decision_log_by_idempotency_key(
    pool: AsyncConnectionPool, key: str
) -> dict[str, Any] | None:
    """Check if a decision with this idempotency key already exists (before lock)."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT id, "createdAt", description, action, "wouldBeAction",
                   "topServiceId", confidence, gap, agree, reason,
                   "textTop", "imageTop", pool, "rerankScores",
                   "configVersion", timings, models
            FROM "DecisionLog"
            WHERE models->>'idempotency_key' = %s
            ORDER BY id DESC LIMIT 1;
            """,
            (key,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "description": row[2],
            "action": row[3],
            "would_be_action": row[4],
            "top_service_id": row[5],
            "confidence": row[6],
            "gap": row[7],
            "agree": row[8],
            "reason": row[9],
            "text_top": row[10],
            "image_top": row[11],
            "pool": row[12],
            "rerank_scores": row[13],
            "config_version": row[14],
            "timings": row[15],
            "models": row[16],
        }


async def get_decision_log(
    pool: AsyncConnectionPool, decision_id: int
) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT id, "createdAt", description, action, "wouldBeAction",
                   "topServiceId", confidence, gap, agree, reason,
                   pool, models
            FROM "DecisionLog" WHERE id = %s;
            """,
            (decision_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "action": row[3],
            "would_be_action": row[4],
            "confidence": row[6],
            "pool": row[10],
        }


async def update_decision_feedback(
    pool: AsyncConnectionPool,
    decision_id: int,
    service_id: int,
) -> dict[str, Any]:
    async with pool.connection() as conn:
        # Verify decision exists
        cur = await conn.execute(
            'SELECT pool FROM "DecisionLog" WHERE id = %s;', (decision_id,)
        )
        row = await cur.fetchone()
        if not row:
            raise ValueError(f"Decision {decision_id} not found")

        pool_data = row[0]
        if isinstance(pool_data, str):
            pool_list = json.loads(pool_data)
        elif isinstance(pool_data, list):
            pool_list = pool_data
        else:
            pool_list = []

        # Verify service exists
        cur2 = await conn.execute(
            'SELECT 1 FROM "WoService" WHERE id = %s;', (service_id,)
        )
        if not await cur2.fetchone():
            raise ValueError(f"Service {service_id} not found")

        pool_contains_final = any(
            item.get("service_id") == service_id for item in pool_list
        )

        cur3 = await conn.execute(
            """
            UPDATE "DecisionLog"
            SET "finalServiceId" = %s,
                "finalAt" = now(),
                "poolContainsFinal" = %s
            WHERE id = %s
            RETURNING id, "finalServiceId", "finalAt", "poolContainsFinal";
            """,
            (service_id, pool_contains_final, decision_id),
        )
        updated = await cur3.fetchone()
        await conn.commit()

    return {
        "status": "ok",
        "decision_id": updated[0],
        "final_service_id": updated[1],
        "final_at": updated[2].isoformat() if updated[2] else None,
        "pool_contains_final": updated[3],
    }


async def get_all_services(
    pool: AsyncConnectionPool,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        if include_inactive:
            cur = await conn.execute(
                'SELECT id, title, description, keywords, "isActive", "embeddingModel", "updatedAt" '
                'FROM "WoService" ORDER BY title ASC;'
            )
        else:
            cur = await conn.execute(
                'SELECT id, title, description, keywords, "isActive", "embeddingModel", "updatedAt" '
                'FROM "WoService" WHERE "isActive" ORDER BY title ASC;'
            )
        rows = await cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "title": r[1],
            "description": r[2],
            "keywords": r[3],
            "is_active": r[4],
            "embedding_model": r[5],
            "updated_at": r[6].isoformat() if r[6] else None,
        })
    return results


async def upsert_service(
    pool: AsyncConnectionPool,
    title: str,
    description: str,
    keywords: str | None,
    is_active: bool,
    title_emb: list[float],
    desc_emb: list[float],
    embedding_model: str,
    existing_id: int | None = None,
) -> dict[str, Any]:
    """Upsert a service and its text embeddings. Returns the service row."""
    title_vec = "[" + ",".join(str(v) for v in title_emb) + "]"
    desc_vec = "[" + ",".join(str(v) for v in desc_emb) + "]"

    async with pool.connection() as conn:
        if existing_id is not None:
            cur = await conn.execute(
                """
                UPDATE "WoService"
                SET title = %s, description = %s, keywords = %s,
                    "isActive" = %s, "embeddingModel" = %s,
                    "titleEmb" = %s::vector, "descEmb" = %s::vector,
                    "updatedAt" = now()
                WHERE id = %s
                RETURNING id, title, "updatedAt";
                """,
                (title, description, keywords, is_active,
                 embedding_model, title_vec, desc_vec, existing_id),
            )
        else:
            cur = await conn.execute(
                """
                INSERT INTO "WoService"
                    (title, description, keywords, "isActive",
                     "embeddingModel", "titleEmb", "descEmb", "updatedAt")
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, now())
                ON CONFLICT (title) DO UPDATE
                    SET description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        "isActive" = EXCLUDED."isActive",
                        "embeddingModel" = EXCLUDED."embeddingModel",
                        "titleEmb" = EXCLUDED."titleEmb",
                        "descEmb" = EXCLUDED."descEmb",
                        "updatedAt" = now()
                RETURNING id, title, "updatedAt";
                """,
                (title, description, keywords, is_active,
                 embedding_model, title_vec, desc_vec),
            )
        row = await cur.fetchone()
        await conn.commit()
    return {"id": row[0], "title": row[1], "updated_at": row[2].isoformat() if row[2] else None}


async def upsert_service_image_profile(
    pool: AsyncConnectionPool,
    service_id: int,
    image_model: str,
    visual_line: str | None,
    text_used: str,
    text_emb: list[float],
    truncated: bool,
) -> None:
    emb_vec = "[" + ",".join(str(v) for v in text_emb) + "]"
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO "ServiceImageProfile"
                ("serviceId", "imageModel", "visualLine", "textUsed",
                 "truncated", "textEmb", "updatedAt")
            VALUES (%s, %s, %s, %s, %s, %s::vector, now())
            ON CONFLICT ("serviceId", "imageModel") DO UPDATE SET
                "visualLine" = EXCLUDED."visualLine",
                "textUsed" = EXCLUDED."textUsed",
                "truncated" = EXCLUDED."truncated",
                "textEmb" = EXCLUDED."textEmb",
                "updatedAt" = now();
            """,
            (service_id, image_model, visual_line, text_used, truncated, emb_vec),
        )
        await conn.commit()


async def soft_delete_service(pool: AsyncConnectionPool, service_id: int) -> bool:
    """Soft-delete by setting isActive=false. Returns True if row found."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            'UPDATE "WoService" SET "isActive" = FALSE, "updatedAt" = now() WHERE id = %s RETURNING id;',
            (service_id,),
        )
        row = await cur.fetchone()
        await conn.commit()
    return row is not None


async def get_service_by_id(
    pool: AsyncConnectionPool, service_id: int
) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            'SELECT id, title, description, keywords, "isActive", "embeddingModel", "updatedAt" '
            'FROM "WoService" WHERE id = %s;',
            (service_id,),
        )
        row = await cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0], "title": row[1], "description": row[2],
        "keywords": row[3], "is_active": row[4],
        "embedding_model": row[5],
        "updated_at": row[6].isoformat() if row[6] else None,
    }


async def get_service_by_title(
    pool: AsyncConnectionPool, title: str
) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            'SELECT id, title, description, keywords, "isActive", "embeddingModel" '
            'FROM "WoService" WHERE title = %s;',
            (title,),
        )
        row = await cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "title": row[1], "description": row[2],
            "keywords": row[3], "is_active": row[4], "embedding_model": row[5]}


async def get_recent_decisions(
    pool: AsyncConnectionPool, limit: int = 20
) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT d.id, d."createdAt", d.description, d.action, d."wouldBeAction",
                   d."topServiceId", s_top.title, d.confidence,
                   d."finalServiceId", s_final.title, d.timings
            FROM "DecisionLog" d
            LEFT JOIN "WoService" s_top ON d."topServiceId" = s_top.id
            LEFT JOIN "WoService" s_final ON d."finalServiceId" = s_final.id
            ORDER BY d.id DESC LIMIT %s;
            """,
            (limit,),
        )
        rows = await cur.fetchall()

    results = []
    for row in rows:
        timings_data = row[10]
        latency_ms = None
        if timings_data:
            if isinstance(timings_data, str):
                timings_data = json.loads(timings_data)
            if isinstance(timings_data, dict):
                latency_ms = timings_data.get("total_ms")
        results.append({
            "id": row[0],
            "created_at": row[1].isoformat() if row[1] else None,
            "description": row[2],
            "action": row[3],
            "would_be_action": row[4],
            "top_service_id": row[5],
            "top_service_title": row[6],
            "confidence": row[7],
            "final_service_id": row[8],
            "final_service_title": row[9],
            "latency_ms": latency_ms,
        })
    return results

