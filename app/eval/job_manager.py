"""
Background evaluation job manager.
Orchestrates sequential evaluation runs, live SSE event broadcasting,
warm-up requests, cancellation, thumbnail encoding, disk cleanup,
and export artifact generation.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import datetime
import hashlib
import io
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from PIL import Image

from app.config import Settings
from app.eval.metrics import (
    EvalRowResult,
    classify_none_action,
    compute_metrics,
    compute_signal_quality,
    is_row_correct,
)
from app.eval.validator import EvalRowInput, ValidationReport
from app.eval.zip_handler import ExtractedDataset, cleanup_temp_dir
from app.pipeline import engine

logger = logging.getLogger("siglip.eval.job_manager")


def _encode_thumbnail_b64(img_path: str, max_size: int = 96) -> str | None:
    """Creates a lightweight JPEG base64 data URI for browser tables."""
    try:
        with Image.open(img_path) as im:
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            rgb_im = im.convert("RGB")
            buf = io.BytesIO()
            rgb_im.save(buf, format="JPEG", quality=75, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _save_preview_image(img_path: str, out_path: Path, max_long_side: int = 1280, quality: int = 80) -> bool:
    """Saves a larger preview JPEG (max 1280px on long side, quality ~80) for modal display."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(img_path) as im:
            im.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
            rgb_im = im.convert("RGB")
            rgb_im.save(out_path, format="JPEG", quality=quality, optimize=True)
            return True
    except Exception as exc:
        logger.warning("Failed to save preview image to %s: %s", out_path, exc)
        return False


def compute_catalog_hash(service_profiles: dict[int, dict[str, Any]] | list[dict[str, Any]] | None) -> str:
    """
    Computes stable SHA-256 hash of active service catalog:
    title + description + visual line per service.
    """
    if not service_profiles:
        return "none"
    items = []
    if isinstance(service_profiles, dict):
        iterable = service_profiles.values()
    else:
        iterable = service_profiles or []
    for s in iterable:
        title = (s.get("title") or "").strip()
        desc = (s.get("description") or "").strip()
        vl = (s.get("visual_line") or s.get("visualLine") or "").strip()
        items.append((title, desc, vl))
    items.sort(key=lambda x: x[0])
    lines = [f"{t}:::{d}:::{v}" for t, d, v in items]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


@dataclass
class EvalJob:
    job_id: str
    created_at: str
    dataset: ExtractedDataset
    validation_report: ValidationReport
    status: str = "validated"  # "validated", "running", "completed", "stopped", "error"
    error_message: str | None = None

    # Effective settings for this run (per-request only)
    effective_settings: dict[str, Any] = field(default_factory=dict)

    # Runtime tracking
    started_at: str | None = None
    completed_at: str | None = None
    total_elapsed_sec: float = 0.0
    current_index: int = 0
    total_rows: int = 0

    # Results
    row_results: list[EvalRowResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    # Control
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _listeners: list[asyncio.Queue] = field(default_factory=list)

    def to_dict(self, include_rows: bool = True) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "status": self.status,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_elapsed_sec": round(self.total_elapsed_sec, 2),
            "current_index": self.current_index,
            "total_rows": self.total_rows,
            "effective_settings": self.effective_settings,
            "summary": self.summary,
        }
        if include_rows:
            data["rows"] = [r.to_dict() for r in self.row_results]
        return data


class EvalJobManager:
    """
    Manages active evaluation jobs in memory and saves finished runs to disk.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, EvalJob] = {}
        self.runs_dir = Path(settings.EVAL_RUNS_DIR)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._load_saved_runs_index()

    def _load_saved_runs_index(self) -> None:
        """Loads metadata for previously completed runs from disk."""
        try:
            for filepath in sorted(self.runs_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
                if len(self.jobs) >= self.settings.EVAL_MAX_HISTORY_RUNS:
                    break
                job_id = filepath.stem
                if job_id not in self.jobs:
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Reconstruct minimal job stub for history viewing
                            job = EvalJob(
                                job_id=data.get("job_id", job_id),
                                created_at=data.get("created_at", ""),
                                dataset=None,  # Temp dir already cleaned up
                                validation_report=None,
                                status=data.get("status", "completed"),
                                started_at=data.get("started_at"),
                                completed_at=data.get("completed_at"),
                                total_elapsed_sec=data.get("total_elapsed_sec", 0.0),
                                current_index=data.get("total_rows", 0),
                                total_rows=data.get("total_rows", 0),
                                effective_settings=data.get("effective_settings", {}),
                                summary=data.get("summary", {}),
                            )
                            # Load rows if present
                            if "rows" in data:
                                valid_keys = set(EvalRowResult.__dataclass_fields__.keys())
                                job.row_results = [
                                    EvalRowResult(**{k: v for k, v in r.items() if k in valid_keys})
                                    for r in data["rows"]
                                ]
                                for row in job.row_results:
                                    if row.top1_raw_score is None and row.candidates:
                                        row.top1_raw_score = row.candidates[0].get("raw_score")
                                eff = job.effective_settings or {}
                                # Upgrade legacy run summary to include none_panel & accurate top1 accuracy if missing
                                if "none_panel" not in job.summary or "signal_quality" not in job.summary:
                                    titles = list(set(r.expected_service for r in job.row_results if not r.is_none))
                                    job.summary = compute_metrics(
                                        job.row_results,
                                        all_service_titles=titles,
                                        current_conf_threshold=float(eff.get("confidence_threshold", self.settings.DECISION_CONFIDENCE_THRESHOLD)),
                                        current_gap_threshold=float(eff.get("gap_threshold", self.settings.DECISION_GAP_THRESHOLD)),
                                        current_no_action_floor=float(eff.get("no_action_floor", self.settings.DECISION_NO_ACTION_FLOOR)),
                                        total_elapsed_sec=job.total_elapsed_sec,
                                        require_agree=self.settings.DECISION_REQUIRE_AGREE,
                                        always_draft_services=self.settings.always_draft_services_list,
                                        min_description_words=self.settings.MIN_DESCRIPTION_WORDS,
                                        enable_reranker=bool(eff.get("enable_reranker", getattr(self.settings, "ENABLE_RERANKER", True))),
                                    )
                            self.jobs[job.job_id] = job
                    except Exception as e:
                        logger.warning("Could not read saved run %s: %s", filepath, e)
            self._cleanup_old_runs()
        except Exception as exc:
            logger.warning("Error reading runs directory: %s", exc)

    def get_preview_path(self, job_id: str, row_id: int) -> Path | None:
        """Returns absolute Path to larger preview JPEG if it exists."""
        safe_job_id = Path(job_id).name
        preview_file = self.runs_dir / safe_job_id / "previews" / f"{row_id}.jpg"
        if preview_file.is_file():
            return preview_file
        return None

    def _cleanup_old_runs(self) -> None:
        """Prunes old runs and their preview folders beyond EVAL_MAX_HISTORY_RUNS."""
        try:
            max_runs = self.settings.EVAL_MAX_HISTORY_RUNS
            json_files = sorted(self.runs_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
            if len(json_files) > max_runs:
                for old_file in json_files[max_runs:]:
                    old_job_id = old_file.stem
                    try:
                        old_file.unlink(missing_ok=True)
                        old_dir = self.runs_dir / old_job_id
                        if old_dir.is_dir():
                            shutil.rmtree(old_dir, ignore_errors=True)
                        self.jobs.pop(old_job_id, None)
                        logger.info("Pruned old evaluation run and previews: %s", old_job_id)
                    except Exception as e:
                        logger.warning("Error deleting old run %s: %s", old_job_id, e)
        except Exception as exc:
            logger.warning("Error running run history cleanup: %s", exc)

    def create_job(
        self,
        job_id: str,
        dataset: ExtractedDataset,
        report: ValidationReport,
    ) -> EvalJob:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        job = EvalJob(
            job_id=job_id,
            created_at=now_iso,
            dataset=dataset,
            validation_report=report,
            status="validated",
            total_rows=report.valid_rows_count,
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> EvalJob | None:
        return self.jobs.get(job_id)

    def list_runs(self) -> list[dict[str, Any]]:
        """Returns sorted list of finished and active runs."""
        res = []
        for j in self.jobs.values():
            res.append({
                "job_id": j.job_id,
                "created_at": j.created_at,
                "status": j.status,
                "total_rows": j.total_rows,
                "completed_rows": len([r for r in j.row_results if not r.is_warmup]),
                "top1_accuracy": j.summary.get("overall", {}).get("top1_accuracy"),
                "auto_precision": j.summary.get("actions", {}).get("auto_precision"),
                "wrong_auto_count": j.summary.get("actions", {}).get("wrong_auto_count", 0),
                "effective_settings": j.effective_settings,
            })
        res.sort(key=lambda x: x["created_at"], reverse=True)
        return res[:self.settings.EVAL_MAX_HISTORY_RUNS]

    async def start_job(
        self,
        job_id: str,
        app_state: Any,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status == "running":
            raise ValueError(f"Job {job_id} is already running")

        job.status = "running"
        job.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        job._stop_event.clear()
        job.row_results.clear()

        # Compute catalog hash from active service profiles (Requirement 1)
        profiles = getattr(app_state, "service_profiles", None)
        catalog_hash = compute_catalog_hash(profiles)

        # Compute effective per-request settings (snapshot of all pipeline-affecting parameters)
        ov = overrides or {}
        eff = {
            "overrides": dict(ov),
            "enable_reranker": bool(ov.get("enable_reranker", getattr(self.settings, "ENABLE_RERANKER", True))),
            "description_weight": float(ov.get("description_weight", self.settings.PIPELINE_TEXT_DESC_WEIGHT)),
            "text_top_n": int(self.settings.PIPELINE_TEXT_TOP_N),
            "image_top_n": int(self.settings.PIPELINE_IMAGE_TOP_N),
            "pool_max": int(self.settings.PIPELINE_POOL_MAX),
            "reranker_temp": float(self.settings.PIPELINE_RERANKER_TEMP),
            "biencoder_temp": float(self.settings.PIPELINE_BIENCODER_TEMP),
            "agree_boost": float(self.settings.PIPELINE_BIENCODER_AGREE_BOOST),
            "confidence_threshold": float(ov.get("confidence_threshold", self.settings.DECISION_CONFIDENCE_THRESHOLD)),
            "gap_threshold": float(ov.get("gap_threshold", self.settings.DECISION_GAP_THRESHOLD)),
            "no_action_floor": float(ov.get("no_action_floor", self.settings.DECISION_NO_ACTION_FLOOR)),
            "require_agree": bool(self.settings.DECISION_REQUIRE_AGREE),
            "always_draft_services": list(self.settings.always_draft_services_list),
            "min_description_words": int(self.settings.MIN_DESCRIPTION_WORDS),
            "auto_create_enabled": bool(ov.get("auto_create_enabled", self.settings.AUTO_CREATE_ENABLED)),
            "text_guard_enabled": bool(ov.get("text_guard_enabled", getattr(self.settings, "DECISION_TEXT_GUARD_ENABLED", True))),
            "text_no_action_floor": float(ov.get("text_no_action_floor", getattr(self.settings, "DECISION_TEXT_NO_ACTION_FLOOR", 0.40))),
            "text_auto_min": float(ov.get("text_auto_min", getattr(self.settings, "DECISION_TEXT_AUTO_MIN", 0.45))),
            "agree_mode": (ov.get("agree_mode") if ov.get("agree_mode") in ("strict", "balanced", "relaxed")
                           else getattr(self.settings, "DECISION_AGREE_MODE", "strict")),
            "agree_text_margin": float(getattr(self.settings, "DECISION_AGREE_TEXT_MARGIN", 0.05)),
            "agree_image_top_k": int(getattr(self.settings, "DECISION_AGREE_IMAGE_TOP_K", 3)),
            "model_ids": {
                "text": "BAAI/bge-base-en-v1.5",
                "image": self.settings.SIGLIP_MODEL_ID,
                "reranker": "BAAI/bge-reranker-base",
            },
            "catalog_hash": catalog_hash,
        }

        # Check demo services in registry
        try:
            from app.demo_registry import load_demo_registry
            demo_entries = await load_demo_registry(self.settings.DEMO_REGISTRY_PATH)
            eff["has_demo_services"] = bool(demo_entries)
            eff["demo_services_count"] = len(demo_entries)
            eff["demo_service_titles"] = [d.get("title") for d in demo_entries if d.get("title")]
        except Exception as exc:
            logger.warning("Could not check demo registry at eval start: %s", exc)
            eff["has_demo_services"] = False
            eff["demo_services_count"] = 0
            eff["demo_service_titles"] = []

        job.effective_settings = eff

        # Launch background execution task
        asyncio.create_task(self._run_evaluation_worker(job, app_state))

    def stop_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job and job.status == "running":
            job._stop_event.set()
            return True
        return False

    async def subscribe(self, job_id: str) -> AsyncGenerator[str, None]:
        """SSE event generator for real-time progress updates."""
        job = self.get_job(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        queue: asyncio.Queue = asyncio.Queue()
        job._listeners.append(queue)

        try:
            # First send catch-up event with existing state
            init_payload = {
                "type": "init",
                "job_id": job.job_id,
                "status": job.status,
                "current_index": job.current_index,
                "total_rows": job.total_rows,
                "effective_settings": job.effective_settings,
                "rows": [r.to_dict() for r in job.row_results],
                "summary": job.summary,
            }
            yield f"event: init\ndata: {json.dumps(init_payload)}\n\n"

            # If already done, close stream
            if job.status in ("completed", "stopped", "error"):
                return

            while True:
                msg = await queue.get()
                yield msg
                if job.status in ("completed", "stopped", "error") and queue.empty():
                    break
        finally:
            if queue in job._listeners:
                job._listeners.remove(queue)

    def _broadcast(self, job: EvalJob, event_name: str, payload: dict[str, Any]) -> None:
        msg = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
        for q in list(job._listeners):
            try:
                q.put_nowait(msg)
            except Exception:
                pass

    async def _run_evaluation_worker(self, job: EvalJob, app_state: Any) -> None:
        """
        Executes pipeline dry-run sequentially for each row in the dataset.
        Includes a warm-up pass, collects metrics, streams SSE events,
        and ensures temp directory is deleted after the run.
        """
        t_start = time.perf_counter()
        active_service_titles = [
            p.get("title") for p in app_state.service_profiles.values() if p.get("title")
        ]

        # Create settings copy with per-request overrides
        run_settings = self.settings.model_copy(update={
            "PIPELINE_TEXT_DESC_WEIGHT": job.effective_settings["description_weight"],
            "DECISION_CONFIDENCE_THRESHOLD": job.effective_settings["confidence_threshold"],
            "DECISION_GAP_THRESHOLD": job.effective_settings["gap_threshold"],
            "DECISION_NO_ACTION_FLOOR": job.effective_settings["no_action_floor"],
            "ENABLE_RERANKER": job.effective_settings.get("enable_reranker", True),
            "DECISION_TEXT_GUARD_ENABLED": job.effective_settings.get("text_guard_enabled", getattr(self.settings, "DECISION_TEXT_GUARD_ENABLED", True)),
            "DECISION_TEXT_NO_ACTION_FLOOR": job.effective_settings.get("text_no_action_floor", getattr(self.settings, "DECISION_TEXT_NO_ACTION_FLOOR", 0.40)),
            "DECISION_TEXT_AUTO_MIN": job.effective_settings.get("text_auto_min", getattr(self.settings, "DECISION_TEXT_AUTO_MIN", 0.45)),
            "DECISION_AGREE_MODE": job.effective_settings.get("agree_mode", getattr(self.settings, "DECISION_AGREE_MODE", "strict")),
        })

        valid_rows = [
            r for r in job.validation_report.valid_rows
            if r.image_path is not None
        ]
        job.total_rows = len(valid_rows)

        try:
            # ── 1. Warm-Up Request (excluded from timing stats) ───────────────
            if valid_rows:
                self._broadcast(job, "status", {
                    "status": "warmup",
                    "message": "Running warm-up request...",
                })
                warmup_input = valid_rows[0]
                try:
                    with open(warmup_input.image_path, "rb") as f:
                        w_bytes = f.read()
                    await engine.process_post(
                        image_bytes=w_bytes,
                        description=warmup_input.description,
                        location=None,
                        settings=run_settings,
                        pool=app_state.db_pool,
                        siglip_service=app_state.siglip_service,
                        tei_client=app_state.tei_client,
                        service_profiles=app_state.service_profiles,
                        semaphore=app_state.semaphore,
                        idempotency_key=None,
                        dry_run=True,
                        return_diagnostics=True,
                    )
                except Exception as w_exc:
                    logger.warning("Warm-up pass encountered non-fatal error: %s", w_exc)

            # ── 2. Sequential Evaluation Loop ────────────────────────────────
            t_eval_start = time.perf_counter()
            for idx, r_in in enumerate(valid_rows, start=1):
                if job._stop_event.is_set():
                    job.status = "stopped"
                    self._broadcast(job, "stopped", {
                        "message": "Evaluation stopped by user.",
                        "current_index": idx - 1,
                    })
                    break

                job.current_index = idx

                # Read image bytes & save larger preview for lightbox modal
                img_bytes: bytes | None = None
                thumb_b64: str | None = None
                if r_in.image_path and os.path.exists(r_in.image_path):
                    try:
                        with open(r_in.image_path, "rb") as f:
                            img_bytes = f.read()
                        thumb_b64 = _encode_thumbnail_b64(r_in.image_path)
                        # Save larger preview JPEG (max 1280px on long side, quality ~80) to disk
                        previews_dir = self.runs_dir / job.job_id / "previews"
                        preview_file = previews_dir / f"{r_in.row_id}.jpg"
                        _save_preview_image(r_in.image_path, preview_file, max_long_side=1280, quality=80)
                    except Exception as e:
                        logger.warning("Failed to read/preview image %s: %s", r_in.image_path, e)

                # Process through real pipeline with dry_run=True
                result = await engine.process_post(
                    image_bytes=img_bytes,
                    description=r_in.description if r_in.description else None,
                    location=None,
                    settings=run_settings,
                    pool=app_state.db_pool,
                    siglip_service=app_state.siglip_service,
                    tei_client=app_state.tei_client,
                    service_profiles=app_state.service_profiles,
                    semaphore=app_state.semaphore,
                    idempotency_key=None,
                    dry_run=True,
                    return_diagnostics=True,
                )

                # Classify correctness using single shared helper (Requirement 4)
                predicted_top1 = (
                    result.top_candidates[0].title
                    if result.top_candidates else "NONE"
                )
                action = result.action
                exp_svc = r_in.expected_service

                correct = is_row_correct(
                    is_none=r_in.is_none,
                    action=action,
                    predicted_service=predicted_top1,
                    expected_service=exp_svc,
                )

                # Pool ranks of expected service
                text_pool_rank: int | None = None
                image_pool_rank: int | None = None
                reranker_rank: int | None = None
                entered_pool = False

                if not r_in.is_none:
                    # Search through candidates
                    for c_idx, c in enumerate(result.all_candidates or result.top_candidates, start=1):
                        if c.title == exp_svc:
                            reranker_rank = c.rerank_rank or c_idx
                            entered_pool = True
                            if c.text_rank is not None:
                                text_pool_rank = int(c.text_rank)
                            if c.image_rank is not None:
                                image_pool_rank = int(c.image_rank)
                            break

                full_t_rank = result.full_text_ranks.get(exp_svc) if not r_in.is_none else None
                full_i_rank = result.full_image_ranks.get(exp_svc) if not r_in.is_none else None

                # Extract all service text and image scores (Requirement 2 & Part 2)
                all_service_text_scores = getattr(result, "all_service_text_scores", {})
                all_service_image_scores = getattr(result, "all_service_image_scores", {})
                service_scores = getattr(result, "service_scores", {})

                # Extract explicit top-1 raw score and top-1 image similarity (Requirement 2)
                top1_cand = result.top_candidates[0] if result.top_candidates else None
                top1_raw_score = None
                if top1_cand:
                    top1_raw_score = top1_cand.raw_score if top1_cand.raw_score is not None else top1_cand.rerank_score

                top1_img_sim = all_service_image_scores.get(predicted_top1) if predicted_top1 in all_service_image_scores else None
                top1_txt_sim = all_service_text_scores.get(predicted_top1) if predicted_top1 in all_service_text_scores else None

                # Format candidate pool for table expansion (keeping raw reranker score for pool candidates)
                candidates_out = []
                for c in (result.all_candidates or result.top_candidates):
                    raw_s = c.raw_score if c.raw_score is not None else c.rerank_score
                    candidates_out.append({
                        "service_id": c.service_id,
                        "title": c.title,
                        "confidence": c.confidence,
                        "confidence_pct": c.confidence_pct,
                        "raw_score": raw_s,
                        "raw_reranker_score": raw_s,
                        "text_score": getattr(c, "text_score", None),
                        "text_rank": c.text_rank,
                        "image_rank": c.image_rank,
                        "pool_rank": c.pool_rank,
                        "rerank_rank": c.rerank_rank,
                    })

                raw_rerank_scores = []
                for item in (result.raw_rerank_scores or []):
                    raw_rerank_scores.append({
                        "service": item.get("service"),
                        "raw_score": item.get("rerank_score", 0.0),
                        "confidence": item.get("confidence", 0.0),
                    })

                row_res = EvalRowResult(
                    row_id=r_in.row_id,
                    image=r_in.image_name,
                    description=r_in.description,
                    expected_service=exp_svc,
                    predicted_service=predicted_top1,
                    action=action,
                    confidence=result.confidence,
                    gap=result.gap,
                    correct=correct,
                    is_warmup=False,
                    is_none=r_in.is_none,
                    is_empty_desc=r_in.is_empty_desc,
                    text_pool_rank=text_pool_rank,
                    image_pool_rank=image_pool_rank,
                    reranker_rank=reranker_rank,
                    full_text_rank=full_t_rank,
                    full_image_rank=full_i_rank,
                    entered_pool=entered_pool,
                    text_top1=result.text_top1,
                    image_top1=result.image_top1,
                    agree=result.agree,
                    raw_reranker_scores=raw_rerank_scores,
                    candidates=candidates_out,
                    timings=result.timings,
                    thumbnail_b64=thumb_b64,
                    all_service_text_scores=all_service_text_scores,
                    all_service_image_scores=all_service_image_scores,
                    service_scores=service_scores,
                    top1_raw_score=top1_raw_score,
                    top1_image_similarity=top1_img_sim,
                    top1_text_similarity=top1_txt_sim,
                )
                job.row_results.append(row_res)

                # Recompute live summary
                job.total_elapsed_sec = time.perf_counter() - t_eval_start
                summary = compute_metrics(
                    job.row_results,
                    all_service_titles=active_service_titles,
                    current_conf_threshold=job.effective_settings["confidence_threshold"],
                    current_gap_threshold=job.effective_settings["gap_threshold"],
                    current_no_action_floor=job.effective_settings["no_action_floor"],
                    total_elapsed_sec=job.total_elapsed_sec,
                    require_agree=self.settings.DECISION_REQUIRE_AGREE,
                    always_draft_services=self.settings.always_draft_services_list,
                    min_description_words=self.settings.MIN_DESCRIPTION_WORDS,
                    enable_reranker=job.effective_settings.get("enable_reranker", True),
                )
                job.summary = summary

                # Stream live row progress event
                self._broadcast(job, "row", {
                    "index": idx,
                    "total": job.total_rows,
                    "row": row_res.to_dict(),
                    "summary": summary,
                })

            if not job._stop_event.is_set():
                job.status = "completed"

            job.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            job.total_elapsed_sec = time.perf_counter() - t_eval_start

            # Final summary calculation
            job.summary = compute_metrics(
                job.row_results,
                all_service_titles=active_service_titles,
                current_conf_threshold=job.effective_settings["confidence_threshold"],
                current_gap_threshold=job.effective_settings["gap_threshold"],
                current_no_action_floor=job.effective_settings["no_action_floor"],
                total_elapsed_sec=job.total_elapsed_sec,
                require_agree=self.settings.DECISION_REQUIRE_AGREE,
                always_draft_services=self.settings.always_draft_services_list,
                min_description_words=self.settings.MIN_DESCRIPTION_WORDS,
                enable_reranker=job.effective_settings.get("enable_reranker", True),
            )

            # Check demo services in catalog and add warning to run summary if active
            try:
                from app.demo_registry import load_demo_registry
                demo_entries = await load_demo_registry(self.settings.DEMO_REGISTRY_PATH)
                if demo_entries:
                    demo_titles = [d.get("title") for d in demo_entries if d.get("title")]
                    demo_warn = (
                        f"DEMO SERVICES ACTIVE: {len(demo_entries)} demo service(s) currently active in catalog "
                        f"({', '.join(demo_titles)}). Demo services change the catalog hash and can skew evaluation results."
                    )
                    warnings_list = job.summary.setdefault("warnings", [])
                    if demo_warn not in warnings_list:
                        warnings_list.insert(0, demo_warn)
                    job.effective_settings["has_demo_services"] = True
                    job.effective_settings["demo_services_count"] = len(demo_entries)
                    job.effective_settings["demo_service_titles"] = demo_titles
            except Exception as exc:
                logger.warning("Could not check demo registry at eval completion: %s", exc)

            # Persist finished run to disk
            self._save_run_to_disk(job)

            self._broadcast(job, "complete", {
                "status": job.status,
                "total_rows": job.total_rows,
                "elapsed_sec": round(job.total_elapsed_sec, 2),
                "summary": job.summary,
            })

        except Exception as exc:
            logger.error("Evaluation run failed: %s", exc, exc_info=True)
            job.status = "error"
            job.error_message = str(exc)
            self._broadcast(job, "error", {"error": str(exc)})

        finally:
            # Delete extracted temporary directory after the run
            if job.dataset and job.dataset.temp_dir:
                cleanup_temp_dir(job.dataset.temp_dir)
                job.dataset = None

    def _save_run_to_disk(self, job: EvalJob) -> None:
        """Saves run artifact JSON to EVAL_RUNS_DIR and prunes old runs."""
        try:
            out_file = self.runs_dir / f"{job.job_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(include_rows=True), f, indent=2)
            logger.info("Saved evaluation run to %s", out_file)
            self._cleanup_old_runs()
        except Exception as exc:
            logger.warning("Failed to save run to disk: %s", exc)

    def export_csv(self, job_id: str) -> str:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        output = io.StringIO()
        fieldnames = [
            "row_id", "image", "description", "expected_service", "predicted_service",
            "action", "none_level", "confidence", "gap", "correct", "is_none", "false_match", "agree",
            "top1_raw_score", "top1_image_similarity", "top1_text_similarity",
            "text_pool_rank", "image_pool_rank", "reranker_rank",
            "full_text_rank", "full_image_rank", "entered_pool",
            "text_top1", "image_top1", "total_ms",
            "image_decode_ms", "siglip_encode_ms", "text_embed_ms", "pgvector_search_ms",
            "rerank_ms", "decision_ms"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in job.row_results:
            is_fm = bool(r.is_none and r.action not in ("no_action", "no action"))
            none_cls = classify_none_action(r.action) if r.is_none else None
            none_lvl = none_cls["level"] if none_cls else ""
            writer.writerow({
                "row_id": r.row_id,
                "image": r.image,
                "description": r.description,
                "expected_service": "NONE" if r.is_none else r.expected_service,
                "predicted_service": r.predicted_service,
                "action": r.action,
                "none_level": none_lvl,
                "confidence": r.confidence,
                "gap": r.gap,
                "correct": r.correct,
                "is_none": r.is_none,
                "false_match": is_fm,
                "agree": r.agree,
                "top1_raw_score": r.top1_raw_score,
                "top1_image_similarity": r.top1_image_similarity,
                "top1_text_similarity": r.top1_text_similarity,
                "text_pool_rank": r.text_pool_rank if r.text_pool_rank is not None else "not in pool",
                "image_pool_rank": r.image_pool_rank if r.image_pool_rank is not None else "not in pool",
                "reranker_rank": r.reranker_rank if r.reranker_rank is not None else "not in pool",
                "full_text_rank": r.full_text_rank,
                "full_image_rank": r.full_image_rank,
                "entered_pool": r.entered_pool,
                "text_top1": r.text_top1,
                "image_top1": r.image_top1,
                "total_ms": r.timings.get("total_ms"),
                "image_decode_ms": r.timings.get("image_decode_ms"),
                "siglip_encode_ms": r.timings.get("siglip_encode_ms"),
                "text_embed_ms": r.timings.get("text_embed_ms"),
                "pgvector_search_ms": r.timings.get("pgvector_search_ms"),
                "rerank_ms": r.timings.get("rerank_ms"),
                "decision_ms": r.timings.get("decision_ms"),
            })
        return output.getvalue()

    def export_json(self, job_id: str) -> str:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        return json.dumps(job.to_dict(include_rows=True), indent=2)

    def export_markdown(self, job_id: str) -> str:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        s = job.summary
        ov = s.get("overall", {})
        sub = s.get("subsets", {})
        act = s.get("actions", {})
        none_p = s.get("none_panel", {})
        sq = s.get("signal_quality", {})
        lat = s.get("latency", {})
        stages = lat.get("stages", {})
        eff = job.effective_settings or {}

        md = []
        md.append(f"# SigLIP Pipeline Evaluation Report")
        md.append(f"- **Job ID**: `{job.job_id}`")
        md.append(f"- **Completed**: {job.completed_at or 'In Progress'}")
        md.append(f"- **Total Rows Evaluated**: {s.get('n', len(job.row_results))}")
        md.append(f"- **Throughput**: {lat.get('throughput_fps', 0)} img/sec ({job.total_elapsed_sec:.1f}s total)")
        md.append("")

        # Visible banner when reranker is OFF (Requirement 1)
        if eff.get("enable_reranker") is False:
            md.append("> ⚠️ **RERANKER OFF (BI-ENCODER MODE)**: Cross-encoder reranker was disabled for this run. Predictions use direct multimodal bi-encoder scoring with agreement boost.")
            md.append("")

        findings = s.get("findings", [])
        if findings:
            md.append("## 📌 Key Findings & Warnings")
            for f_item in findings:
                md.append(f"- ⚠️ **{f_item}**")
            md.append("")

        md.append("## 1. Run Record Snapshot (Settings & Active Catalog)")
        md.append(f"- **Active Catalog Hash**: `{eff.get('catalog_hash', 'none')}`")
        md.append(f"- **Enable Reranker**: `{eff.get('enable_reranker')}`")
        md.append(f"- **Description Weight**: `{eff.get('description_weight')}`")
        md.append(f"- **Confidence Threshold**: `{eff.get('confidence_threshold')}`")
        md.append(f"- **Gap Threshold**: `{eff.get('gap_threshold')}`")
        md.append(f"- **No-Action Floor**: `{eff.get('no_action_floor')}`")
        md.append(f"- **Text Top N**: `{eff.get('text_top_n')}`")
        md.append(f"- **Image Top N**: `{eff.get('image_top_n')}`")
        md.append(f"- **Pool Max**: `{eff.get('pool_max')}`")
        md.append(f"- **Reranker Softmax Temp**: `{eff.get('reranker_temp')}`")
        md.append(f"- **Bi-encoder Softmax Temp**: `{eff.get('biencoder_temp')}`")
        md.append(f"- **Bi-encoder Agree Boost**: `{eff.get('agree_boost')}`")
        md.append(f"- **Require Multimodal Agree**: `{eff.get('require_agree')}`")
        md.append(f"- **Min Description Words**: `{eff.get('min_description_words')}`")
        md.append(f"- **Auto-Create Enabled (Actually Used)**: `{eff.get('auto_create_enabled')}`")
        model_ids = eff.get('model_ids', {})
        md.append(f"- **Model IDs**: Text: `{model_ids.get('text')}`, Image: `{model_ids.get('image')}`, Reranker: `{model_ids.get('reranker')}`")
        md.append(f"- **Always-Draft Services**: `{json.dumps(eff.get('always_draft_services', []))}`")
        if eff.get("overrides"):
            md.append(f"- **Per-Run Overrides**: `{json.dumps(eff.get('overrides'))}`")
        md.append("")

        md.append("## 2. Accuracy & Retrieval Metrics (Normal Ground Truth)")
        md.append(f"- **Top-1 Accuracy**: **{ov.get('top1_accuracy')}%**")
        md.append(f"- **Top-3 Accuracy**: {ov.get('top3_accuracy')}%")
        md.append(f"- **Top-5 Accuracy**: {ov.get('top5_accuracy')}%")
        md.append(f"- **Candidate Pool Recall**: **{ov.get('pool_recall')}%** (reranker upper bound)")
        md.append(f"- **Text Top-1 Accuracy**: {ov.get('text_top1_accuracy')}%")
        md.append(f"- **Image Top-1 Accuracy**: {ov.get('image_top1_accuracy')}%")
        md.append(f"- **Agreement Rate**: {ov.get('agree_rate')}% (Accuracy when agree: {ov.get('accuracy_when_agree')}%, disagree: {ov.get('accuracy_when_disagree')}%)")
        md.append("")

        if none_p and none_p.get("n", 0) > 0:
            levels = none_p.get("levels", {})
            l_no_act = levels.get("no_action", {})
            l_draft = levels.get("draft", {})
            l_auto = levels.get("auto_create", {})

            md.append("## 3. NONE Ground Truth Panel (Out-of-Scope Three-Level Reporting)")
            md.append(f"- **Total NONE Rows (n)**: {none_p.get('n')}")
            md.append(f"- **Level 1: No Action (Correct Suppression)**: **{l_no_act.get('rate', 0)}%** ({l_no_act.get('count', 0)}/{none_p.get('n')})")
            md.append(f"- **Level 2: Draft (Soft False Match)**: **{l_draft.get('rate', 0)}%** ({l_draft.get('count', 0)}/{none_p.get('n')})")
            md.append(f"- **Level 3: Auto-Create (Harmful False Match / WRONG-AUTO)**: **{l_auto.get('rate', 0)}%** ({l_auto.get('count', 0)}/{none_p.get('n')})")
            md.append(f"- **TOTAL FALSE-MATCH COUNT**: **{none_p.get('false_match_count')}**")
            md.append(f"- **CRITICAL WRONG-AUTO COUNT**: **{none_p.get('wrong_auto_count', 0)}**")
            md.append("")

            f_matches = none_p.get("false_matches", [])
            if f_matches:
                md.append("### False Matches Details")
                md.append("| Image | Description | Predicted Service | Action | Level / Severity | Confidence | Gap |")
                md.append("|---|---|---|---|---|---|---|")
                for fm in f_matches:
                    conf_str = f"{fm.get('confidence', 0.0) * 100:.1f}%"
                    gap_str = f"{fm.get('gap', 0.0) * 100:.1f}%"
                    lvl_badge = "⚠️ HARMFUL WRONG-AUTO" if fm.get("is_harmful") else "Soft False Match (Draft)"
                    md.append(f"| `{fm.get('image')}` | {fm.get('description') or '<empty>'} | {fm.get('predicted_service')} | `{fm.get('action')}` | {lvl_badge} | {conf_str} | {gap_str} |")
                md.append("")

            hist = none_p.get("histogram", [])
            if hist:
                md.append("### Top-1 Confidence Distribution (NONE vs Correct Normal)")
                md.append("| Confidence Bin | NONE Rows Count | Correct Normal Rows Count |")
                md.append("|---|---|---|")
                for h in hist:
                    md.append(f"| {h.get('label')} | {h.get('none_count')} | {h.get('normal_correct_count')} |")
                md.append("")

        md.append("## 4. Subsets Breakdown")
        for sub_name, data in sub.items():
            md.append(f"- **{sub_name.replace('_', ' ').title()}**: {data.get('accuracy')}% (n={data.get('n')}, correct={data.get('correct')})")
        md.append("")

        md.append("## 5. Triage Actions & Safety")
        md.append(f"- **Auto-Create Count**: {act.get('auto_create_count')} ({act.get('coverage_pct')}% coverage)")
        md.append(f"- **Auto-Create Precision**: **{act.get('auto_precision')}%**")
        md.append(f"- **Draft Count**: {act.get('draft_count')}")
        md.append(f"- **No-Action Count**: {act.get('no_action_count')}")
        md.append(f"- **WRONG-AUTO COUNT**: **{act.get('wrong_auto_count', 0)}**")
        md.append("")

        md.append("## 6. Latency Statistics (Excluding Warmup)")
        md.append(f"- **Mean Latency**: {lat.get('mean_ms')} ms")
        md.append(f"- **P50 (Median)**: {lat.get('p50_ms')} ms")
        md.append(f"- **P95 Latency**: {lat.get('p95_ms')} ms")
        md.append(f"- **Max Latency**: {lat.get('max_ms')} ms")
        md.append("")
        md.append("### Stage Breakdown (Mean)")
        for sk, sv in stages.items():
            md.append(f"- **{sv.get('label')}**: {sv.get('mean_ms')} ms")
        md.append("")

        # Section 7: Signal Quality Table (Requirement 5)
        signals_list = sq.get("signals", [])
        if signals_list:
            md.append("## 7. Signal Quality Analysis (AUROC Table)")
            md.append("| Signal | n (Normal) | AUROC (Top-1 Correct) | Warning (Normal) | n (All) | AUROC (Normal vs NONE) | Warning (NONE) |")
            md.append("|---|---|---|---|---|---|---|")
            for item in signals_list:
                s_name = item.get("name", "")
                t1 = item.get("top1_correct", {})
                nv = item.get("normal_vs_none", {})
                auc1 = f"{t1.get('auroc'):.3f}" if t1.get("auroc") is not None else "N/A"
                auc2 = f"{nv.get('auroc'):.3f}" if nv.get("auroc") is not None else "N/A"
                w1 = t1.get("warning") or "-"
                w2 = nv.get("warning") or "-"
                md.append(f"| {s_name} | {t1.get('n', 0)} | **{auc1}** | {w1} | {nv.get('n', 0)} | **{auc2}** | {w2} |")
            md.append("")

        top_conf = s.get("top_confusions", [])
        if top_conf:
            md.append("## 8. Top Confusions (Expected -> Predicted)")
            for c in top_conf[:7]:
                md.append(f"- `{c['expected']}` → `{c['predicted']}`: {c['count']} row(s)")
            md.append("")

        calib = s.get("calibration", [])
        if calib:
            md.append("## 9. Calibration Offline Sweep (Temperature)")
            md.append("| Temperature | ECE (%) | Coverage (%) | Precision (%) |")
            md.append("|---|---|---|---|")
            for c in calib:
                md.append(f"| {c['temperature']} | {c['ece']}% | {c['coverage_pct']}% | {c['precision_pct']}% |")
            md.append("")

        warnings = s.get("warnings", [])
        if warnings:
            md.append("## 10. Data & Sample Size Warnings")
            for w in warnings:
                md.append(f"- ⚠️ {w}")
            md.append("")

        return "\n".join(md)


_job_manager: EvalJobManager | None = None


def get_eval_job_manager(settings: Settings | None = None) -> EvalJobManager:
    global _job_manager
    if _job_manager is None:
        from app.config import get_settings
        _job_manager = EvalJobManager(settings or get_settings())
    return _job_manager

