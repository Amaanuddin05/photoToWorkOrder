"""LLM review of low-confidence (draft) posts with Claude.

The pipeline calls review_draft() only for drafts that have both a photo and a description and are not
shadow / always-draft / safety drafts. Claude sees the photo, the description and the candidate services and
returns a constrained JSON choice. A high-confidence pick whose photo matches may promote the post to
auto_create (the engine decides; see apply_review()).

Token usage and an approximate USD cost (LLM_PRICING, $ per million tokens) are returned for every call.
The API key comes from ANTHROPIC_API_KEY in the environment and is never logged.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any

from PIL import Image

logger = logging.getLogger("siglip.llm_review")

# Models that accept server-side refusal fallbacks (fallbacks="default"); Haiku has none.
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}

_SYSTEM = (
    "You triage maintenance posts sent by facility workers on a radio channel (school campus). "
    "Each post has a photo and a short description, in English or Arabic. Choose the ONE catalog service "
    "that should handle the work, only from the candidate list, or NONE when the post is not a maintenance "
    "request or none of the candidates fits. Use both the photo and the description. "
    "confidence=high only when the description clearly asks for that service AND the photo shows it. "
    "photo_matches=true only when the photo shows the problem or area the chosen service handles. "
    "Also classify the message intent: issue = reports a problem or requests work; done = reports that work "
    "was completed; other = chit-chat, announcements or anything that is not a maintenance request "
    "(if intent is not issue, service should be NONE). "
    "Keep reason under 25 words."
)

_client = None


def _api_key(settings: Any) -> str | None:
    return getattr(settings, "ANTHROPIC_API_KEY", None) or os.environ.get("ANTHROPIC_API_KEY")


def _get_client(settings: Any):
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.AsyncAnthropic(
            api_key=_api_key(settings),
            timeout=float(getattr(settings, "LLM_TIMEOUT_SECONDS", 30)),
            max_retries=1,
        )
    return _client


def _image_block(image_bytes: bytes, max_side: int) -> dict[str, Any]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": base64.standard_b64encode(buf.getvalue()).decode("ascii")},
    }


def _schema(titles: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["issue", "done", "other"]},
                "service": {"type": "string", "enum": titles + ["NONE"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "photo_matches": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["intent", "service", "confidence", "photo_matches", "reason"],
            "additionalProperties": False,
        },
    }


def _price(pricing: dict[str, dict[str, float]], model: str) -> dict[str, float]:
    p = pricing.get(model) or {}
    inp = float(p.get("input", 0.0))
    return {
        "input": inp,
        "output": float(p.get("output", 0.0)),
        "cache_read": float(p.get("cache_read", inp * 0.1)),
        "cache_write": float(p.get("cache_write", inp * 1.25)),
    }


def _cost(usage: dict[str, int], price: dict[str, float]) -> float:
    return (
        usage.get("input_tokens", 0) * price["input"]
        + usage.get("output_tokens", 0) * price["output"]
        + usage.get("cache_read_input_tokens", 0) * price["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]
    ) / 1_000_000


async def _catalog_lines(pool: Any, candidate_titles: list[str]) -> dict[str, str]:
    """Catalog description per candidate title (best effort; empty when the DB is unavailable)."""
    if pool is None or not candidate_titles:
        return {}
    try:
        async with pool.connection() as conn:
            cur = await conn.execute(
                'SELECT title, description, keywords FROM "WoService" WHERE title = ANY(%s);',
                (candidate_titles,),
            )
            rows = await cur.fetchall()
        return {r[0]: (f"{r[1] or ''} Keywords: {r[2]}" if r[2] else (r[1] or "")).strip() for r in rows}
    except Exception as exc:
        logger.warning("LLM review: could not read catalog descriptions: %s", exc)
        return {}


async def review_post(
    *,
    settings: Any,
    model: str,
    description: str,
    image_bytes: bytes | None,
    candidate_titles: list[str],
    pool: Any = None,
) -> dict[str, Any]:
    """Ask Claude for the intent and the service. Never raises: errors come back in result['error']."""
    pricing = settings.llm_pricing
    result: dict[str, Any] = {
        "used": True, "model": model, "served_model": None, "intent": None, "service": None, "confidence": None,
        "photo_matches": None, "reason": None, "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "cost_usd": 0.0,
        "latency_ms": 0.0, "error": None, "promoted": False, "trigger": None,
    }
    if not _api_key(settings):
        result["error"] = "ANTHROPIC_API_KEY is not set"
        return result
    if not candidate_titles:
        result["error"] = "no candidate services"
        return result

    descs = await _catalog_lines(pool, candidate_titles)
    cand_text = "\n".join(
        f"- {t}: {descs.get(t, '')}".rstrip(": ") for t in candidate_titles
    )
    user_text = (
        f"Candidate services:\n{cand_text}\n\n"
        f"Worker's description: {description.strip()}\n\n"
        + ("" if image_bytes else "(No photo attached: set photo_matches=false.)\n")
        + "Classify the intent and pick the service for this post."
    )
    content: list[dict[str, Any]] = []
    if image_bytes:
        content.append(_image_block(image_bytes, int(getattr(settings, "LLM_IMAGE_MAX_SIDE", 768))))
    content.append({"type": "text", "text": user_text})
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=1024,
        system=_SYSTEM,
        output_config={"effort": "low", "format": _schema(candidate_titles)},
        messages=[{"role": "user", "content": content}],
    )
    if model in _FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"

    import anthropic
    t0 = time.perf_counter()
    try:
        client = _get_client(settings)
        resp = await client.beta.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        result["error"] = "authentication failed (check ANTHROPIC_API_KEY)"
    except anthropic.RateLimitError:
        result["error"] = "rate limited"
    except anthropic.BadRequestError as exc:
        result["error"] = f"bad request: {getattr(exc, 'message', exc)}"
    except anthropic.APIStatusError as exc:
        result["error"] = f"API error {exc.status_code}"
    except anthropic.APIConnectionError:
        result["error"] = "connection error / timeout"
    except Exception as exc:  # never break the pipeline
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if result["error"]:
        logger.warning("LLM review failed: %s", result["error"])
        return result

    # Usage + cost. With a server-side fallback the per-model iterations are priced separately.
    u = resp.usage
    usage = {
        "input_tokens": int(u.input_tokens or 0),
        "output_tokens": int(u.output_tokens or 0),
        "cache_read_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
    }
    result.update(usage)
    result["served_model"] = resp.model
    cost = None
    iters = getattr(u, "iterations", None) or []
    if len(iters) > 1:
        try:
            cost = 0.0
            for it in iters:
                it_model = getattr(it, "model", None) or resp.model
                it_usage = {k: int(getattr(it, k, 0) or 0) for k in usage}
                cost += _cost(it_usage, _price(pricing, it_model))
        except Exception:
            cost = None
    if cost is None:
        cost = _cost(usage, _price(pricing, resp.model or model))
    result["cost_usd"] = round(cost, 6)

    if resp.stop_reason == "refusal":
        result["error"] = "model refused"
        return result
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        data = json.loads(text)
        result.update({
            "intent": data.get("intent"),
            "service": data.get("service"),
            "confidence": data.get("confidence"),
            "photo_matches": bool(data.get("photo_matches")),
            "reason": (data.get("reason") or "")[:300],
        })
    except Exception:
        result["error"] = f"unparseable output (stop_reason={resp.stop_reason})"
    return result


def apply_intent_review(result: Any, llm: dict[str, Any]) -> bool:
    """When the intent gate was unsure: a done/other verdict from the LLM -> no_action. Returns True if applied."""
    if llm.get("error") or llm.get("intent") not in ("done", "other"):
        return False
    result.llm = llm
    result.reason = (f"llm intent={llm['intent']} ({llm.get('confidence')}) - not a new issue report"
                     f" - {llm.get('reason')} | was: {result.reason}")
    result.action = "no_action"
    result.would_be_action = "no_action"
    result.shadow = False
    if result.intent is not None:
        result.intent["acted"] = True
        result.intent["by"] = "llm"
    return True


def apply_review(result: Any, llm: dict[str, Any], *, auto_create_enabled: bool,
                 blocked_services: list[str]) -> None:
    """Update a draft ProcessedPost in place from the LLM review."""
    result.llm = llm
    pick = llm.get("service")
    if llm.get("error") or not pick:
        result.reason = f"{result.reason} | llm: not used ({llm.get('error') or 'no answer'})"
        return
    if pick == "NONE":
        result.reason = f"{result.reason} | llm: NONE ({llm.get('confidence')}) - {llm.get('reason')}"
        return

    # Move the picked service to the top of the candidates (shown + logged as the top service).
    for lst in (result.all_candidates, result.top_candidates):
        idx = next((i for i, c in enumerate(lst) if c.title == pick), None)
        if idx:
            lst.insert(0, lst.pop(idx))
        elif idx is None and lst is result.top_candidates:
            src = next((c for c in result.all_candidates if c.title == pick), None)
            if src is not None:
                lst.insert(0, src)
                del lst[3:]

    promote = (llm.get("confidence") == "high" and llm.get("photo_matches")
               and pick not in blocked_services)
    summary = f"llm: {pick} ({llm.get('confidence')}, photo {'matches' if llm.get('photo_matches') else 'does not match'}) - {llm.get('reason')}"
    if not promote:
        result.reason = f"{summary} | stays draft | was: {result.reason}"
        return
    llm["promoted"] = True
    result.would_be_action = "auto_create"
    if auto_create_enabled:
        result.action = "auto_create"
        result.shadow = False
        result.reason = f"{summary} | was: {result.reason}"
    else:
        result.shadow = True
        result.reason = f"Shadow mode: {summary}, routed to draft"
