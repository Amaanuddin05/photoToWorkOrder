"""
Pydantic-settings configuration for the siglip service.
All values read from environment variables / .env file.
No secrets are printed in repr or logs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/wo_search"
    DB_SCHEMA: str = "public"
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10
    DB_TIMEOUT_SECONDS: float = 10.0

    # ── TEI Dependencies ──────────────────────────────────────────────────────
    TEI_EMBED_URL: str = "http://tei-embed:80"
    TEI_RERANK_URL: str = "http://tei-rerank:80"
    TEI_API_KEY: str | None = None
    TEI_TIMEOUT_SECONDS: float = 10.0
    TEI_MAX_RETRIES: int = 2

    # ── SigLIP Vision Model ───────────────────────────────────────────────────
    SIGLIP_MODEL_ID: str = "google/siglip2-base-patch16-224"
    SIGLIP_TORCH_THREADS: int | None = None
    SIGLIP_MAX_CONCURRENCY: int = 2
    SIGLIP_LRU_CACHE_SIZE: int = 256
    SIGLIP_API_KEY: str | None = None
    SIGLIP_MAX_UPLOAD_BYTES: int | None = None
    SIGLIP_MAX_IMAGES_PER_REQUEST: int = 16
    SIGLIP_MAX_CANDIDATES_PER_REQUEST: int = 100

    # ── Pipeline Weights ──────────────────────────────────────────────────────
    PIPELINE_TEXT_DESC_WEIGHT: float = 0.7
    PIPELINE_TEXT_TOP_N: int = 10
    PIPELINE_IMAGE_TOP_N: int = 5
    PIPELINE_POOL_MAX: int = 10
    PIPELINE_RERANKER_TEMP: float = 1.0

    # ── Decision Thresholds ───────────────────────────────────────────────────
    DECISION_CONFIDENCE_THRESHOLD: float = 0.40
    DECISION_GAP_THRESHOLD: float = 0.10
    DECISION_NO_ACTION_FLOOR: float = 0.25
    DECISION_REQUIRE_AGREE: bool = False
    AUTO_CREATE_ENABLED: bool = False
    MIN_DESCRIPTION_WORDS: int = 3
    ALWAYS_DRAFT_SERVICES: str = (
        '["Emergency Response & Area Securing", '
        '"Emergency Reactive Cleaning (Spills, Vomit, Incidents)"]'
    )

    # ── Security & Auth ───────────────────────────────────────────────────────
    API_KEYS: str = ""          # comma-separated; empty = no auth required
    METRICS_API_KEY: str | None = None

    # ── Operational Limits ────────────────────────────────────────────────────
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024   # 10 MB
    STORE_DESCRIPTION: bool = True
    ENABLE_DEBUG_ENDPOINTS: bool = False
    ENABLE_UI: bool = False

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"     # "json" or "console"

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def api_key_list(self) -> list[str]:
        keys = [k.strip() for k in self.API_KEYS.split(",") if k.strip()]
        if self.SIGLIP_API_KEY and self.SIGLIP_API_KEY.strip() not in keys:
            keys.append(self.SIGLIP_API_KEY.strip())
        return keys

    @property
    def always_draft_services_list(self) -> list[str]:
        try:
            return json.loads(self.ALWAYS_DRAFT_SERVICES)
        except Exception:
            return []

    @property
    def config_version(self) -> str:
        """Stable hash of pipeline-affecting config values."""
        payload = "|".join(str(v) for v in [
            self.SIGLIP_MODEL_ID,
            self.PIPELINE_TEXT_DESC_WEIGHT,
            self.PIPELINE_TEXT_TOP_N,
            self.PIPELINE_IMAGE_TOP_N,
            self.PIPELINE_POOL_MAX,
            self.PIPELINE_RERANKER_TEMP,
            self.DECISION_CONFIDENCE_THRESHOLD,
            self.DECISION_GAP_THRESHOLD,
            self.DECISION_NO_ACTION_FLOOR,
            self.DECISION_REQUIRE_AGREE,
            self.MIN_DESCRIPTION_WORDS,
        ])
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def safe_repr(self) -> dict[str, Any]:
        """Return a repr safe for logging: secrets masked."""
        return {
            "SIGLIP_MODEL_ID": self.SIGLIP_MODEL_ID,
            "TEI_EMBED_URL": self.TEI_EMBED_URL,
            "TEI_RERANK_URL": self.TEI_RERANK_URL,
            "TEI_API_KEY": "***" if self.TEI_API_KEY else None,
            "API_KEYS": f"[{len(self.api_key_list)} keys]",
            "DATABASE_URL": "***@" + self.DATABASE_URL.split("@")[-1] if "@" in self.DATABASE_URL else "***",
            "AUTO_CREATE_ENABLED": self.AUTO_CREATE_ENABLED,
            "ENABLE_DEBUG_ENDPOINTS": self.ENABLE_DEBUG_ENDPOINTS,
            "ENABLE_UI": self.ENABLE_UI,
            "config_version": self.config_version,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
