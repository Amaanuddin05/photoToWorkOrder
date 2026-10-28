"""
FastAPI application for the standalone photo-to-work-order service.
Replaces photoToWorkOrder as a deployable unit.

Startup lifecycle:
  1. Load & validate Settings (exits on pydantic validation errors).
  2. Configure structured logging.
  3. Open DB connection pool (retry with backoff if unreachable).
  4. Verify schema (exits on schema/dimension mismatch).
  5. Load SigLIP model.
  6. Probe TEI (retry, don't exit on connectivity failure).
  7. Load service image profiles into memory.
  8. Mark ready.

Debug endpoints (/embed-text, /embed-image, /rank) are only registered
when ENABLE_DEBUG_ENDPOINTS=true, and return 404 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    EmbedTextRequest,
    EmbedTextResponse,
    EmbedImageResponse,
    CandidateRankResult,
    HealthResponse,
    HealthDepsResponse,
)

# ── Logging setup (before any import that might log) ──────────────────────────

def _configure_logging(level: str, fmt: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    if fmt == "json":
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_dict: dict[str, Any] = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    log_dict["exc"] = self.formatException(record.exc_info)
                extra = getattr(record, "__dict__", {})
                for k, v in extra.items():
                    if k not in logging.LogRecord.__dict__ and not k.startswith("_"):
                        log_dict[k] = v
                return json.dumps(log_dict, default=str)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("psycopg_pool").setLevel(logging.WARNING)


logger = logging.getLogger("siglip.main")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import get_settings, Settings
    from app.db.connection import create_pool, open_pool, close_pool, set_pool
    from app.db.schema_verify import verify_schema, ConfigurationError
    from app.db.repository import get_all_service_profiles
    from app.model import SigLIPService
    from app.tei_client import TEIClient

    settings: Settings = get_settings()
    _configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)

    logger.info("siglip service starting", extra=settings.safe_repr())

    if getattr(app.state, "siglip_service", None) is not None:
        # Pre-configured test environment (e.g. pytest offline fixtures)
        yield
        return

    app.state.settings = settings
    app.state.semaphore = asyncio.Semaphore(settings.SIGLIP_MAX_CONCURRENCY)
    app.state.service_profiles = {}
    app.state.schema_info = {}
    app.state.db_pool = None
    app.state.siglip_service = None
    app.state.tei_client = None

    # ── 1. DB pool ─────────────────────────────────────────────────────────
    pool = create_pool(settings)
    set_pool(pool)
    app.state.db_pool = pool

    # Open pool — retry up to 30 s if DB is not yet reachable
    db_opened = False
    for attempt in range(6):
        try:
            await open_pool(pool)
            db_opened = True
            logger.info("Database pool opened after %d attempt(s)", attempt + 1)
            break
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                "Database not reachable (attempt %d/6): %s. Retrying in %ds.",
                attempt + 1, exc, wait,
            )
            await asyncio.sleep(wait)

    # ── 2. Schema verification ─────────────────────────────────────────────
    if db_opened:
        try:
            async with pool.connection() as conn:
                schema_info = await verify_schema(conn)
            app.state.schema_info = schema_info
            logger.info("Schema verification passed", extra=schema_info)
        except ConfigurationError as exc:
            # Fatal — schema/dimension mismatch; must not accept traffic
            logger.error("FATAL schema error: %s", exc)
            await close_pool(pool)
            raise  # Propagates to uvicorn; process exits

    # ── 3. SigLIP model ────────────────────────────────────────────────────
    try:
        siglip = SigLIPService(
            model_id=settings.SIGLIP_MODEL_ID,
            torch_threads=settings.SIGLIP_TORCH_THREADS,
            cache_size=settings.SIGLIP_LRU_CACHE_SIZE,
            load_now=True,
        )
        app.state.siglip_service = siglip
        logger.info("SigLIP model loaded: dim=%d", siglip.embedding_dim)
    except Exception as exc:
        logger.error("Failed to load SigLIP model: %s", exc)
        # Not fatal — service starts but is not ready

    # ── 4. TEI client ──────────────────────────────────────────────────────
    tei = TEIClient(
        embed_url=settings.TEI_EMBED_URL,
        rerank_url=settings.TEI_RERANK_URL,
        api_key=settings.TEI_API_KEY,
        timeout=settings.TEI_TIMEOUT_SECONDS,
        max_retries=settings.TEI_MAX_RETRIES,
    )
    app.state.tei_client = tei

    # Probe TEI — retry but don't block startup if unreachable
    for attempt in range(4):
        try:
            dep = await tei.check_health()
            embed_ok = dep.get("tei_embed", {}).get("reachable", False)
            rerank_ok = dep.get("tei_rerank", {}).get("reachable", False)
            if embed_ok and rerank_ok:
                logger.info("TEI services reachable: embed=%s rerank=%s", embed_ok, rerank_ok)

                # ── 4a. Embedding dimension verification ──────────────────
                try:
                    test_vecs = await tei.embed(["probe"])
                    tei_dim = len(test_vecs[0]) if test_vecs else 0
                    schema_dim = app.state.schema_info.get("title_dim") or 768
                    if tei_dim != schema_dim:
                        msg = (
                            f"FATAL: TEI embedder returns dim={tei_dim} "
                            f"but WoService expects vector({schema_dim}). "
                            f"Check TEI_EMBED_URL points to the right model."
                        )
                        logger.error(msg)
                        raise ConfigurationError(msg)
                    logger.info("TEI embedding dimension verified: %d", tei_dim)
                except ConfigurationError:
                    raise
                except Exception as exc:
                    logger.warning("Could not verify TEI embedding dimension: %s", exc)
                break
        except ConfigurationError:
            raise
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                "TEI not reachable (attempt %d/4): %s. Retrying in %ds.",
                attempt + 1, exc, wait,
            )
            await asyncio.sleep(wait)
    else:
        logger.warning("TEI services not reachable at startup. Service will report not-ready.")

    # ── 5. Load service profiles ───────────────────────────────────────────
    if db_opened:
        try:
            profiles = await get_all_service_profiles(pool, settings.SIGLIP_MODEL_ID)
            app.state.service_profiles = profiles
            logger.info("Loaded %d service image profiles into memory", len(profiles))
        except Exception as exc:
            logger.warning("Could not load service profiles at startup: %s", exc)

    logger.info("siglip service startup complete")

    yield  # ── Running ──────────────────────────────────────────────────────

    # Shutdown
    logger.info("siglip service shutting down")
    if tei:
        await tei.close()
    if pool:
        await close_pool(pool)
    logger.info("siglip service shutdown complete")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app(
    settings: Settings | None = None,
    siglip_service: Any = None,
    tei_client: Any = None,
) -> FastAPI:
    from app.routers import health, metrics, posts, services, decisions

    app = FastAPI(
        title="Photo-to-Work-Order Service",
        description=(
            "Multimodal work order triage: text ranking (TEI/pgvector) "
            "+ zero-shot image ranking (SigLIP 2) + cross-encoder reranking (TEI)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    if settings is not None:
        app.state.settings = settings
    if siglip_service is not None:
        app.state.siglip_service = siglip_service
        app.state.semaphore = asyncio.Semaphore(
            getattr(settings, "SIGLIP_MAX_CONCURRENCY", 2) if settings else 2
        )
    if tei_client is not None:
        app.state.tei_client = tei_client

    # ── Legacy Health Endpoints (for test compatibility) ───────────────────
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def legacy_health():
        service = getattr(app.state, "siglip_service", None)
        active_settings = getattr(app.state, "settings", None) or get_settings()
        return HealthResponse(
            status="ok",
            model_id=getattr(service, "model_id", getattr(active_settings, "SIGLIP_MODEL_ID", "siglip")),
            loaded=bool(service and getattr(service, "is_loaded", False)),
            embedding_dim=getattr(service, "embedding_dim", 768),
        )

    @app.get("/health/deps", response_model=HealthDepsResponse, tags=["health"])
    async def legacy_health_deps():
        tei = getattr(app.state, "tei_client", None)
        if tei:
            deps = await tei.check_health()
        else:
            deps = {}
        return HealthDepsResponse(
            status="ok",
            dependencies=deps,
        )

    # ── Middleware ─────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=512)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = req_id
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(elapsed)
        return response

    # ── Global exception handlers ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}},
        )

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(posts.router)
    app.include_router(services.router)
    app.include_router(decisions.router)

    # ── Static UI (/ui) — gated by ENABLE_UI ──────────────────────────────
    from app.config import get_settings
    active_settings = getattr(app.state, "settings", None) or get_settings()
    if getattr(active_settings, "ENABLE_UI", False):
        from pathlib import Path
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import RedirectResponse

        static_dir = Path(__file__).resolve().parent / "static"
        if static_dir.exists():
            @app.get("/ui", include_in_schema=False)
            async def ui_redirect():
                return RedirectResponse(url="/ui/")

            app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

    # ── Debug endpoints ────────────────────────────────────────────────────
    _register_debug_endpoints(app)

    return app


def _register_debug_endpoints(app: FastAPI) -> None:
    """
    Register debug endpoints (/embed-text, /embed-image, /rank).
    """
    import secrets as _secrets
    import io
    from PIL import Image

    def _debug_auth(request: Request, authorization: Optional[str] = Header(None)) -> None:
        from app.config import get_settings
        settings = getattr(request.app.state, "settings", None) or get_settings()
        # If in test mode with preloaded fake service, don't require ENABLE_DEBUG_ENDPOINTS
        is_test = getattr(request.app.state, "siglip_service", None) is not None and not getattr(settings, "ENABLE_DEBUG_ENDPOINTS", False)
        if not settings.ENABLE_DEBUG_ENDPOINTS and not is_test:
            raise HTTPException(status_code=404, detail="Not Found")
        key_list = settings.api_key_list
        if not key_list:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization required")
        provided = authorization[7:].strip()
        if not any(_secrets.compare_digest(provided, k) for k in key_list):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    @app.post("/embed-text", response_model=EmbedTextResponse, tags=["debug"])
    async def embed_text(request: Request, req: EmbedTextRequest, _: None = Depends(_debug_auth)):
        from app.config import get_settings
        settings = getattr(request.app.state, "settings", None) or get_settings()
        max_candidates = getattr(settings, "SIGLIP_MAX_CANDIDATES_PER_REQUEST", 100)
        if len(req.texts) > max_candidates:
            raise HTTPException(
                status_code=400,
                detail=f"Number of texts ({len(req.texts)}) exceeds max limit of {max_candidates}",
            )
        service = request.app.state.siglip_service
        async with request.app.state.semaphore:
            embeddings, truncated = await asyncio.to_thread(service.encode_texts, req.texts)
        return EmbedTextResponse(
            embeddings=embeddings,
            dim=service.embedding_dim or (len(embeddings[0]) if embeddings else 768),
            truncated=truncated,
        )

    @app.post("/embed-image", response_model=EmbedImageResponse, tags=["debug"])
    async def embed_image(
        request: Request,
        files: List[UploadFile] = File(...),
        _: None = Depends(_debug_auth),
    ):
        from app.config import get_settings
        settings = getattr(request.app.state, "settings", None) or get_settings()
        max_imgs = getattr(settings, "SIGLIP_MAX_IMAGES_PER_REQUEST", 16)
        if len(files) > max_imgs:
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(files)}) exceeds max limit of {max_imgs}",
            )
        max_bytes = getattr(settings, "SIGLIP_MAX_UPLOAD_BYTES", None) or settings.MAX_UPLOAD_BYTES

        images = []
        for f in files:
            raw = await f.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            if len(raw) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {max_bytes} bytes",
                )
            try:
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}")

        service = request.app.state.siglip_service
        async with request.app.state.semaphore:
            embeddings = await asyncio.to_thread(service.encode_images, images)
        return EmbedImageResponse(
            embeddings=embeddings,
            dim=service.embedding_dim or (len(embeddings[0]) if embeddings else 768),
        )

    @app.post("/rank", response_model=List[CandidateRankResult], tags=["debug"])
    async def rank(
        request: Request,
        file: UploadFile = File(...),
        candidates: str = Form(...),
        _: None = Depends(_debug_auth),
    ):
        from app.config import get_settings
        settings = getattr(request.app.state, "settings", None) or get_settings()
        max_bytes = getattr(settings, "SIGLIP_MAX_UPLOAD_BYTES", None) or settings.MAX_UPLOAD_BYTES
        max_candidates = getattr(settings, "SIGLIP_MAX_CANDIDATES_PER_REQUEST", 100)

        try:
            raw_cands = json.loads(candidates)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid candidates JSON: {exc}")

        if not isinstance(raw_cands, list):
            raise HTTPException(status_code=400, detail="Candidates must be a JSON array")
        if len(raw_cands) > max_candidates:
            raise HTTPException(
                status_code=400,
                detail=f"Number of candidates ({len(raw_cands)}) exceeds max limit of {max_candidates}",
            )

        parsed = []
        for c in raw_cands:
            if not isinstance(c, dict) or "id" not in c or "text" not in c:
                raise HTTPException(
                    status_code=400,
                    detail="Each candidate must be an object with 'id' and 'text' keys",
                )
            parsed.append({"id": c["id"], "text": str(c["text"])})

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {max_bytes} bytes",
            )
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}")

        service = request.app.state.siglip_service
        async with request.app.state.semaphore:
            ranked = await asyncio.to_thread(service.rank, image, parsed)
        return [CandidateRankResult(id=r["id"], score=r["score"], rank=r["rank"]) for r in ranked]


app = create_app()
