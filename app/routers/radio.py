"""Radio channel demo (/ui/radio): workers post messages; photo + text messages run the full pipeline.

Messages are stored in "RadioMessage"; photos under app/data/radio/. Only available when ENABLE_UI=true.
"""
from __future__ import annotations

import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.routers.posts import _require_auth

logger = logging.getLogger("siglip.routers.radio")
router = APIRouter(prefix="/v1/radio", tags=["radio-demo"])

PHOTO_DIR = Path(__file__).resolve().parent.parent / "data" / "radio"
_SQL = Path(__file__).resolve().parent.parent.parent / "sql" / "002_radio_message.sql"
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS "RadioMessage" (
    id serial PRIMARY KEY, "createdAt" timestamptz NOT NULL DEFAULT now(), sender text NOT NULL DEFAULT 'Worker',
    text text, "photoPath" text, "decisionId" integer REFERENCES "DecisionLog"(id) ON DELETE SET NULL,
    action text, "serviceTitle" text, reason text, llm jsonb);
CREATE INDEX IF NOT EXISTS "RadioMessage_createdAt_idx" ON "RadioMessage" ("createdAt");
"""


async def ensure_table(pool: Any) -> None:
    async with pool.connection() as conn:
        await conn.execute(_CREATE_SQL)


def _require_ui(request: Request):
    from app.config import get_settings
    if not get_settings().ENABLE_UI:
        raise HTTPException(status_code=404, detail="Radio demo requires ENABLE_UI=true")


def _row(r: tuple) -> dict[str, Any]:
    llm = r[9]
    if isinstance(llm, str):
        try:
            llm = json.loads(llm)
        except Exception:
            llm = None
    return {
        "id": r[0], "created_at": r[1].isoformat() if r[1] else None, "sender": r[2], "text": r[3],
        "has_photo": bool(r[4]), "photo_url": f"/v1/radio/photos/{r[0]}" if r[4] else None,
        "decision_id": r[5], "action": r[6], "service": r[7], "reason": r[8], "llm": llm,
    }


_COLS = 'id, "createdAt", sender, text, "photoPath", "decisionId", action, "serviceTitle", reason, llm'


@router.get("/messages")
async def list_messages(request: Request, limit: int = 100, _ui=Depends(_require_ui),
                        _auth: None = Depends(_require_auth)) -> list[dict[str, Any]]:
    pool = request.app.state.db_pool
    limit = max(1, min(limit, 500))
    async with pool.connection() as conn:
        cur = await conn.execute(
            f'SELECT {_COLS} FROM (SELECT * FROM "RadioMessage" ORDER BY id DESC LIMIT %s) m ORDER BY id ASC;',
            (limit,),
        )
        rows = await cur.fetchall()
    return [_row(r) for r in rows]


@router.post("/messages")
async def send_message(
    request: Request,
    sender: str = Form("Worker"),
    text: str | None = Form(None),
    image: UploadFile | None = File(None),
    llm_enabled: str | None = Form(None),
    llm_model: str | None = Form(None),
    _ui=Depends(_require_ui),
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    from app.config import get_settings
    from app.pipeline import engine

    settings = get_settings()
    state = request.app.state
    pool = state.db_pool
    text = (text or "").strip() or None
    sender = (sender or "Worker").strip()[:60] or "Worker"

    image_bytes: bytes | None = None
    if image is not None:
        raw = await image.read()
        if raw:
            if len(raw) > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    detail={"code": "IMAGE_TOO_LARGE", "message": "Image too large"})
            try:
                pil = Image.open(io.BytesIO(raw))
                pil.load()
                if pil.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError(f"unsupported format {pil.format}")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": str(exc)}) from exc
            image_bytes = raw
    if not text and image_bytes is None:
        raise HTTPException(status_code=422, detail={"code": "EMPTY", "message": "Send text and/or a photo"})

    photo_rel = None
    if image_bytes is not None:
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[Image.open(io.BytesIO(image_bytes)).format]
        photo_rel = f"{uuid.uuid4().hex}.{ext}"
        (PHOTO_DIR / photo_rel).write_bytes(image_bytes)

    async with pool.connection() as conn:
        cur = await conn.execute(
            'INSERT INTO "RadioMessage" (sender, text, "photoPath") VALUES (%s, %s, %s) RETURNING id;',
            (sender, text, photo_rel),
        )
        msg_id = (await cur.fetchone())[0]

    # Photo + description -> full pipeline (intent gate -> ranking -> decision -> optional LLM review)
    if text and image_bytes is not None:
        run_settings = settings
        if llm_enabled in ("on", "off"):
            run_settings = run_settings.model_copy(update={"LLM_ENABLED": llm_enabled == "on"})
        if llm_model and llm_model in settings.llm_allowed_models:
            run_settings = run_settings.model_copy(update={"LLM_MODEL": llm_model})
        try:
            result = await engine.process_post(
                image_bytes=image_bytes, description=text, location=None, settings=run_settings,
                pool=pool, siglip_service=state.siglip_service, tei_client=state.tei_client,
                service_profiles=state.service_profiles, semaphore=state.semaphore,
                idempotency_key=f"radio-{msg_id}", source="radio",
            )
            top = result.top_candidates[0].title if result.top_candidates else None
            llm = result.llm
            async with pool.connection() as conn:
                await conn.execute(
                    'UPDATE "RadioMessage" SET "decisionId"=%s, action=%s, "serviceTitle"=%s, reason=%s, llm=%s WHERE id=%s;',
                    (result.decision_id, result.action, top if result.action != "no_action" else None,
                     result.reason, json.dumps(llm) if llm else None, msg_id),
                )
        except Exception as exc:
            logger.error("Radio message %s: pipeline failed: %s", msg_id, exc, exc_info=True)
            async with pool.connection() as conn:
                await conn.execute('UPDATE "RadioMessage" SET reason=%s WHERE id=%s;',
                                   (f"pipeline error: {exc}", msg_id))

    async with pool.connection() as conn:
        cur = await conn.execute(f'SELECT {_COLS} FROM "RadioMessage" WHERE id=%s;', (msg_id,))
        return _row(await cur.fetchone())


@router.get("/photos/{msg_id}")
async def get_photo(request: Request, msg_id: int, _ui=Depends(_require_ui)):
    pool = request.app.state.db_pool
    async with pool.connection() as conn:
        cur = await conn.execute('SELECT "photoPath" FROM "RadioMessage" WHERE id=%s;', (msg_id,))
        row = await cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="No photo")
    path = (PHOTO_DIR / Path(row[0]).name).resolve()
    if PHOTO_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="No photo")
    return FileResponse(path)


@router.delete("/messages")
async def clear_messages(request: Request, _ui=Depends(_require_ui),
                         _auth: None = Depends(_require_auth)) -> dict[str, Any]:
    """Clear the demo channel (messages + photos). Decisions stay in DecisionLog."""
    pool = request.app.state.db_pool
    async with pool.connection() as conn:
        cur = await conn.execute('DELETE FROM "RadioMessage" RETURNING "photoPath";')
        deleted = await cur.fetchall()
        paths = [r[0] for r in deleted if r[0]]
    for p in paths:
        try:
            (PHOTO_DIR / Path(p).name).unlink(missing_ok=True)
        except OSError:
            pass
    return {"deleted": len(deleted)}
