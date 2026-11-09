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
    # Optional pin for the TEI embed model. Unset (default) = use whatever model TEI
    # serves: the catalog is re-embedded for it automatically (ServiceTextEmbedding).
    # Set = TEI must serve exactly this model, otherwise the service is not-ready.
    TEXT_EMBED_MODEL_ID: str | None = None
    # Optional label for the TEI rerank model (logged only when ENABLE_RERANKER=true).
    # Unset = the model id is read from the rerank container's /info.
    RERANK_MODEL_ID: str | None = None
    # Model of the ORIGINAL "WoService"."titleEmb"/"descEmb" columns (English stack and
    # WoSemanticSearch). They are only written when TEI serves this model with the
    # columns' dimension; otherwise they are left untouched.
    LEGACY_TEXT_MODEL_ID: str = "BAAI/bge-base-en-v1.5"

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
    ENABLE_RERANKER: bool = True
    # Softmax temperature for bi-encoder mode (no cross-encoder reranker).
    # Lower values amplify score gaps: 0.04–0.08 recommended.
    PIPELINE_BIENCODER_TEMP: float = 0.06
    # Agreement bonus added to candidate score when text AND image both rank it #1.
    PIPELINE_BIENCODER_AGREE_BOOST: float = 0.05

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
    # ── Text Score Guard (Bi-encoder mode) ────────────────────────────────────
    # Global values (calibrated on BAAI/bge-base-en-v1.5); used for any model
    # without an entry in TEXT_GUARD_PROFILES.
    DECISION_TEXT_NO_ACTION_FLOOR: float = 0.40
    DECISION_TEXT_AUTO_MIN: float = 0.45
    DECISION_TEXT_GUARD_ENABLED: bool = True
    # Per-model guard values (the guard uses absolute cosine scores, whose scale
    # depends on the embedding model). JSON: {model_id: {"no_action_floor": x, "auto_min": y}}.
    # bge-m3 calibrated with scripts/calibrate_text_guard.py (EN + AR ground truth and
    # held-out sets): floor 0.35 keeps every real request above it; auto_min unchanged.
    TEXT_GUARD_PROFILES: str = '{"BAAI/bge-m3": {"no_action_floor": 0.35, "auto_min": 0.45}}'
    # Internal: when True (eval runs, which resolve the values up front), the
    # DECISION_TEXT_* values are used as-is and profiles are ignored.
    TEXT_GUARD_PINNED: bool = False

    # ── Security & Auth ───────────────────────────────────────────────────────
    API_KEYS: str = ""          # comma-separated; empty = no auth required
    METRICS_API_KEY: str | None = None

    # ── Operational Limits ────────────────────────────────────────────────────
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024   # 10 MB
    STORE_DESCRIPTION: bool = True
    ENABLE_DEBUG_ENDPOINTS: bool = False
    ENABLE_UI: bool = False

    # ── Evaluation Suite ──────────────────────────────────────────────────────
    ENABLE_EVAL: bool = False
    EVAL_MAX_ZIP_BYTES: int = 100 * 1024 * 1024                 # 100 MB max zip upload
    EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES: int = 500 * 1024 * 1024  # 500 MB max uncompressed
    EVAL_MAX_IMAGE_COUNT: int = 1000                            # Max files in eval zip
    EVAL_MAX_IMAGE_BYTES: int = 10 * 1024 * 1024                # 10 MB per image
    EVAL_RUNS_DIR: str = "app/eval/runs"
    EVAL_MAX_HISTORY_RUNS: int = 20

    # ── Demo Services & Zero-Shot Testing ─────────────────────────────────────
    DEMO_REGISTRY_PATH: str = "app/data/demo_services.json"
    DEMO_HARD_DELETE: bool = False

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

    def text_guard_for(self, model_id: str | None) -> tuple[float, float, str]:
        """
        (no_action_floor, auto_min, source) of the text guard for this embedding model:
        its TEXT_GUARD_PROFILES entry if any (unless TEXT_GUARD_PINNED), otherwise the
        global DECISION_TEXT_NO_ACTION_FLOOR / DECISION_TEXT_AUTO_MIN.
        """
        if not self.TEXT_GUARD_PINNED and model_id:
            try:
                prof = json.loads(self.TEXT_GUARD_PROFILES or "{}").get(model_id)
            except Exception:
                prof = None
            if prof:
                return (
                    float(prof.get("no_action_floor", self.DECISION_TEXT_NO_ACTION_FLOOR)),
                    float(prof.get("auto_min", self.DECISION_TEXT_AUTO_MIN)),
                    f"profile:{model_id}",
                )
        return float(self.DECISION_TEXT_NO_ACTION_FLOOR), float(self.DECISION_TEXT_AUTO_MIN), "global"

    @property
    def config_version(self) -> str:
        """Stable hash of pipeline-affecting config values."""
        payload = "|".join(str(v) for v in [
            self.SIGLIP_MODEL_ID,
            self.TEXT_EMBED_MODEL_ID,
            self.RERANK_MODEL_ID if self.ENABLE_RERANKER else "",
            self.PIPELINE_TEXT_DESC_WEIGHT,
            self.PIPELINE_TEXT_TOP_N,
            self.PIPELINE_IMAGE_TOP_N,
            self.PIPELINE_POOL_MAX,
            self.PIPELINE_RERANKER_TEMP,
            self.ENABLE_RERANKER,
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
            "TEXT_EMBED_MODEL_ID": self.TEXT_EMBED_MODEL_ID,
            "RERANK_MODEL_ID": self.RERANK_MODEL_ID,
            "TEI_API_KEY": "***" if self.TEI_API_KEY else None,
            "API_KEYS": f"[{len(self.api_key_list)} keys]",
            "DATABASE_URL": "***@" + self.DATABASE_URL.split("@")[-1] if "@" in self.DATABASE_URL else "***",
            "AUTO_CREATE_ENABLED": self.AUTO_CREATE_ENABLED,
            "ENABLE_DEBUG_ENDPOINTS": self.ENABLE_DEBUG_ENDPOINTS,
            "ENABLE_UI": self.ENABLE_UI,
            "ENABLE_EVAL": self.ENABLE_EVAL,
            "DEMO_REGISTRY_PATH": self.DEMO_REGISTRY_PATH,
            "DEMO_HARD_DELETE": self.DEMO_HARD_DELETE,
            "config_version": self.config_version,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
