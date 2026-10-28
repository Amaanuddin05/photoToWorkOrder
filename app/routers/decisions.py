"""
POST /v1/decisions/{id}/feedback — record operator ground truth.
Mirrors photoToWorkOrder's POST /decisions/{id}/final.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

logger = logging.getLogger("siglip.routers.decisions")
router = APIRouter(prefix="/v1", tags=["decisions"])


class FeedbackRequest(BaseModel):
    service_id: int


def _require_auth(request: Request, authorization: str | None = Header(None)) -> None:
    from app.config import get_settings
    settings = get_settings()
    key_list = settings.api_key_list
    if not key_list:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail={"code": "MISSING_AUTH", "message": "Authorization required"})
    provided = authorization[7:].strip()
    if not any(secrets.compare_digest(provided, k) for k in key_list):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail={"code": "INVALID_API_KEY", "message": "Invalid API key"})


@router.post("/decisions/{decision_id}/feedback")
async def record_feedback(
    decision_id: int,
    body: FeedbackRequest,
    request: Request,
    _auth: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Record the correct service for a decision (operator ground truth).
    Computes poolContainsFinal automatically from the stored pool.
    Mirrors photoToWorkOrder's POST /decisions/{decision_id}/final.
    """
    from app.db import repository as repo

    try:
        result = await repo.update_decision_feedback(
            request.app.state.db_pool,
            decision_id=decision_id,
            service_id=body.service_id,
        )
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg and "Decision" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail={"code": "DECISION_NOT_FOUND", "message": msg})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"code": "SERVICE_NOT_FOUND", "message": msg})
    return result


@router.get("/decisions")
async def list_decisions(
    request: Request,
    limit: int = 20,
    _auth: None = Depends(_require_auth),
) -> list[dict[str, Any]]:
    """Returns newest decisions first, joined with service titles."""
    from app.db import repository as repo

    limit = max(1, min(limit, 100))
    return await repo.get_recent_decisions(request.app.state.db_pool, limit)

