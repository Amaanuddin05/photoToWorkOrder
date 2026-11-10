"""
FastAPI router for dataset evaluation endpoints (/v1/eval/*).
All endpoints are gated behind ENABLE_EVAL and require standard API key authentication.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.eval.job_manager import EvalJobManager, get_eval_job_manager
from app.eval.validator import validate_ground_truth
from app.eval.zip_handler import ZipValidationError, inspect_and_extract_zip

logger = logging.getLogger("siglip.routers.eval")
router = APIRouter(prefix="/v1/eval", tags=["evaluation"])


def _eval_auth(request: Request, authorization: str | None = Header(None)) -> None:
    """Verifies ENABLE_EVAL is true and verifies Bearer token against API_KEYS."""
    from app.config import get_settings

    settings = getattr(request.app.state, "settings", None) or get_settings()
    if not getattr(settings, "ENABLE_EVAL", False):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Evaluation suite is disabled."},
        )

    key_list = settings.api_key_list
    if not key_list:
        return  # Dev mode: API_KEYS empty

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_AUTH", "message": "Authorization header required"},
        )
    provided = authorization[7:].strip()
    if not any(secrets.compare_digest(provided, k) for k in key_list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_API_KEY", "message": "Invalid API key"},
        )


class StartJobPayload(BaseModel):
    description_weight: float | None = None
    confidence_threshold: float | None = None
    gap_threshold: float | None = None
    no_action_floor: float | None = None
    enable_reranker: bool | None = None
    text_guard_enabled: bool | None = None
    text_no_action_floor: float | None = None
    text_auto_min: float | None = None
    agree_mode: str | None = None   # strict | balanced | relaxed (this run only)
    intent_enabled: bool | None = None   # message-intent gate on/off (this run only)


class LoadLocalPayload(BaseModel):
    path: str | None = None


class FusionRequestPayload(BaseModel):
    run_id: str


async def _process_zip_and_create_job(raw_zip: bytes, request: Request) -> JSONResponse:
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    try:
        dataset = inspect_and_extract_zip(raw_zip, settings)
    except ZipValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "ZIP_VALIDATION_ERROR", "message": str(exc)},
        ) from exc

    # Gather active service titles from state or DB
    active_services = []
    if hasattr(request.app.state, "service_profiles") and request.app.state.service_profiles:
        active_services = [
            p["title"] for p in request.app.state.service_profiles.values() if p.get("title")
        ]
    if not active_services and hasattr(request.app.state, "db_pool") and request.app.state.db_pool:
        from app.db import repository as repo
        try:
            db_services = await repo.get_all_services(request.app.state.db_pool, include_inactive=False)
            active_services = [s["title"] for s in db_services if s.get("title")]
        except Exception as exc:
            logger.warning("Could not query active services: %s", exc)

    validation_report = validate_ground_truth(dataset, active_services)

    job_id = f"eval_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    job = manager.create_job(job_id, dataset, validation_report)

    return JSONResponse(
        content={
            "job_id": job.job_id,
            "created_at": job.created_at,
            "total_images_in_zip": dataset.total_images,
            "uncompressed_bytes": dataset.uncompressed_bytes,
            "validation": validation_report.to_dict(),
        }
    )


@router.post("/upload")
async def upload_eval_zip(
    request: Request,
    file: UploadFile = File(...),
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """
    Accepts an uploaded ZIP file containing ground_truth.csv and images.
    Validates safety constraints (zip-slip, size, count), extracts to temp dir,
    and returns pre-run validation report with service and image diagnostics.
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_FILE_TYPE", "message": "Uploaded file must be a .zip archive."},
        )

    raw_zip = await file.read()
    if not raw_zip:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_FILE", "message": "Uploaded zip file is empty."},
        )

    return await _process_zip_and_create_job(raw_zip, request)


@router.post("/load-local")
async def load_local_eval_dataset(
    request: Request,
    payload: LoadLocalPayload = LoadLocalPayload(),
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """
    Loads an evaluation dataset directly from the local disk when running in dev mode
    (e.g. running via python on the host machine). Supports direct .zip paths or folder paths.
    """
    import io
    import os
    import urllib.parse
    import zipfile

    raw_path = (payload.path or "").strip()
    if raw_path.startswith("file:///"):
        raw_path = raw_path[8:]
    elif raw_path.startswith("file://"):
        raw_path = raw_path[7:]
    raw_path = urllib.parse.unquote(raw_path)

    # Normalize Windows drive letter if leading slash remains (e.g. /C:/path -> C:/path)
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]

    candidate_paths = []
    if raw_path:
        candidate_paths.append(os.path.abspath(raw_path))
    else:
        # Default local paths in dev
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate_paths.extend([
            os.path.join(base_dir, "eval_dataset.zip"),
            os.path.join(base_dir, "siglip", "eval_dataset.zip"),
            os.path.expanduser(r"~\Downloads\eval.zip"),
            base_dir,
        ])

    target_path = None
    for cp in candidate_paths:
        if os.path.exists(cp):
            target_path = cp
            break

    if not target_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "FILE_NOT_FOUND",
                "message": f"Local dataset not found at '{raw_path or 'default paths'}'. Please check the path.",
            },
        )

    if os.path.isdir(target_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(target_path):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, target_path).replace("\\", "/")
                    ext = os.path.splitext(f)[1].lower()
                    if ext in {".csv", ".jpg", ".jpeg", ".png", ".webp"}:
                        zf.write(full, arcname=rel)
        raw_zip = buf.getvalue()
    elif target_path.lower().endswith(".zip"):
        with open(target_path, "rb") as f:
            raw_zip = f.read()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_FORMAT", "message": f"Target '{target_path}' is neither a .zip file nor a dataset folder."},
        )

    return await _process_zip_and_create_job(raw_zip, request)


@router.post("/jobs/{job_id}/start")
async def start_eval_job(
    job_id: str,
    request: Request,
    payload: StartJobPayload = StartJobPayload(),
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """
    Starts sequential dry-run evaluation on the validated dataset with optional
    per-request override thresholds.
    """
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Evaluation job '{job_id}' not found.")

    overrides = {k: v for k, v in payload.model_dump().items() if v is not None}
    try:
        await manager.start_job(job_id, request.app.state, overrides=overrides)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse(
        content={
            "status": "started",
            "job_id": job_id,
            "effective_settings": job.effective_settings,
        }
    )


@router.post("/jobs/{job_id}/stop")
async def stop_eval_job(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """Stops an active evaluation run."""
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    stopped = manager.stop_job(job_id)
    if not stopped:
        job = manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return JSONResponse(content={"status": job.status, "message": "Job is not running."})

    return JSONResponse(content={"status": "stopping", "job_id": job_id})


@router.get("/jobs/{job_id}/stream")
async def stream_eval_job(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) stream for live per-row evaluation progress,
    metrics updates, and completion notifications.
    Supports reconnection to active jobs.
    """
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return StreamingResponse(
        manager.subscribe(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}")
async def get_eval_job_details(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """Returns complete state, rows, and summary for an evaluation run."""
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JSONResponse(content=job.to_dict(include_rows=True))


@router.get("/jobs/{job_id}/preview/{row_id}")
async def get_eval_row_preview(
    job_id: str,
    row_id: int,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> FileResponse:
    """
    Serves a larger preview JPEG (up to 1280px on long side) saved during evaluation.
    Enforces Bearer authentication. Frontend fetches via fetch() with headers and creates a blob URL.
    """
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    preview_path = manager.get_preview_path(job_id, row_id)
    if not preview_path or not preview_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Preview image for job '{job_id}' row '{row_id}' not found on disk.",
        )

    return FileResponse(
        path=str(preview_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/compare")
async def compare_eval_runs(
    request: Request,
    run_a: str = Query(..., description="Baseline Job ID (Run A)"),
    run_b: str = Query(..., description="Comparison Job ID (Run B)"),
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """
    Compares two evaluation runs matched by image filename (Requirement 3).
    Returns metric deltas (top-1/3/5, pool recall, auto precision, wrong-auto, NONE rates, latency),
    table of rows whose correctness changed (A wrong -> B right, A right -> B wrong),
    and dataset / catalog hash warnings.
    """
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    job_a = manager.get_job(run_a)
    if not job_a:
        raise HTTPException(status_code=404, detail=f"Run A '{run_a}' not found.")
    job_b = manager.get_job(run_b)
    if not job_b:
        raise HTTPException(status_code=404, detail=f"Run B '{run_b}' not found.")

    eff_a = job_a.effective_settings or {}
    eff_b = job_b.effective_settings or {}

    sum_a = job_a.summary or {}
    sum_b = job_b.summary or {}
    ov_a = sum_a.get("overall", {})
    ov_b = sum_b.get("overall", {})
    act_a = sum_a.get("actions", {})
    act_b = sum_b.get("actions", {})
    none_a = sum_a.get("none_panel", {})
    none_b = sum_b.get("none_panel", {})
    none_lvls_a = none_a.get("levels", {})
    none_lvls_b = none_b.get("levels", {})
    lat_a = sum_a.get("latency", {})
    lat_b = sum_b.get("latency", {})

    # 1. Match rows by image file name
    rows_a_map = {r.image: r for r in job_a.row_results if not r.is_warmup}
    rows_b_map = {r.image: r for r in job_b.row_results if not r.is_warmup}

    set_a = set(rows_a_map.keys())
    set_b = set(rows_b_map.keys())

    # 2. Warnings (datasets or catalog hashes differ)
    warnings = []
    dataset_mismatch = (set_a != set_b)
    if dataset_mismatch:
        diff_a = len(set_a - set_b)
        diff_b = len(set_b - set_a)
        warnings.append(
            f"Dataset Mismatch: Run A has {len(set_a)} images and Run B has {len(set_b)} images "
            f"({diff_a} only in Run A, {diff_b} only in Run B). Only matched images are compared."
        )

    cat_a = eff_a.get("catalog_hash")
    cat_b = eff_b.get("catalog_hash")
    catalog_mismatch = bool(cat_a and cat_b and cat_a != cat_b)
    if catalog_mismatch:
        warnings.append(
            f"Catalog Hash Mismatch: Run A catalog hash ({cat_a}) differs from Run B ({cat_b}). "
            f"Active service definitions, descriptions, or visual lines changed between runs."
        )

    # 3. Metric Deltas (B - A)
    def calc_delta(val_a: Any, val_b: Any, precision: int = 1) -> dict[str, Any]:
        v_a = float(val_a) if val_a is not None else 0.0
        v_b = float(val_b) if val_b is not None else 0.0
        d = round(v_b - v_a, precision)
        return {"a": round(v_a, precision), "b": round(v_b, precision), "delta": d}

    deltas = {
        # Overall Accuracy
        "top1_accuracy": calc_delta(ov_a.get("top1_accuracy"), ov_b.get("top1_accuracy")),
        "top3_accuracy": calc_delta(ov_a.get("top3_accuracy"), ov_b.get("top3_accuracy")),
        "top5_accuracy": calc_delta(ov_a.get("top5_accuracy"), ov_b.get("top5_accuracy")),
        "pool_recall": calc_delta(ov_a.get("pool_recall"), ov_b.get("pool_recall")),
        # Triage Actions & Safety
        "auto_precision": calc_delta(act_a.get("auto_precision"), act_b.get("auto_precision")),
        "wrong_auto_count": {
            "a": act_a.get("wrong_auto_count", 0),
            "b": act_b.get("wrong_auto_count", 0),
            "delta": act_b.get("wrong_auto_count", 0) - act_a.get("wrong_auto_count", 0),
        },
        "coverage_pct": calc_delta(act_a.get("coverage_pct"), act_b.get("coverage_pct")),
        # NONE Rates
        "none_correct_rate": calc_delta(none_a.get("correct_rate"), none_b.get("correct_rate")),
        "none_draft_rate": calc_delta(
            none_lvls_a.get("draft", {}).get("rate", 0.0) if none_lvls_a else 0.0,
            none_lvls_b.get("draft", {}).get("rate", 0.0) if none_lvls_b else 0.0,
        ),
        "none_wrong_auto_count": {
            "a": none_a.get("wrong_auto_count", 0),
            "b": none_b.get("wrong_auto_count", 0),
            "delta": none_b.get("wrong_auto_count", 0) - none_a.get("wrong_auto_count", 0),
        },
        "none_false_match_count": {
            "a": none_a.get("false_match_count", 0),
            "b": none_b.get("false_match_count", 0),
            "delta": none_b.get("false_match_count", 0) - none_a.get("false_match_count", 0),
        },
        # Latency
        "mean_latency_ms": calc_delta(lat_a.get("mean_ms"), lat_b.get("mean_ms")),
        "p50_latency_ms": calc_delta(lat_a.get("p50_ms"), lat_b.get("p50_ms")),
        "p95_latency_ms": calc_delta(lat_a.get("p95_ms"), lat_b.get("p95_ms")),
    }

    # 4. Changed rows (A wrong -> B right, A right -> B wrong)
    common_images = sorted(list(set_a & set_b))
    changed_rows = []

    for img in common_images:
        ra = rows_a_map[img]
        rb = rows_b_map[img]

        if ra.correct != rb.correct:
            if not ra.correct and rb.correct:
                change_type = "improved"
                change_label = "A wrong → B right"
            else:
                change_type = "regressed"
                change_label = "A right → B wrong"

            changed_rows.append({
                "image": img,
                "expected_service": ra.expected_service,
                "is_none": ra.is_none,
                "description": ra.description,
                "change_type": change_type,
                "change_label": change_label,
                "run_a": {
                    "prediction": ra.predicted_service,
                    "confidence": round(ra.confidence, 4) if ra.confidence is not None else None,
                    "gap": round(ra.gap, 4) if ra.gap is not None else None,
                    "action": ra.action,
                    "correct": ra.correct,
                    "raw_score": ra.top1_raw_score,
                },
                "run_b": {
                    "prediction": rb.predicted_service,
                    "confidence": round(rb.confidence, 4) if rb.confidence is not None else None,
                    "gap": round(rb.gap, 4) if rb.gap is not None else None,
                    "action": rb.action,
                    "correct": rb.correct,
                    "raw_score": rb.top1_raw_score,
                },
                "thumbnail_b64": ra.thumbnail_b64 or rb.thumbnail_b64,
            })

    return JSONResponse(
        content={
            "run_a": {
                "job_id": job_a.job_id,
                "created_at": job_a.created_at,
                "settings": eff_a,
                "total_rows": len(rows_a_map),
            },
            "run_b": {
                "job_id": job_b.job_id,
                "created_at": job_b.created_at,
                "settings": eff_b,
                "total_rows": len(rows_b_map),
            },
            "matched_rows_count": len(common_images),
            "warnings": warnings,
            "dataset_mismatch": dataset_mismatch,
            "catalog_mismatch": catalog_mismatch,
            "deltas": deltas,
            "counts": {
                "changed_count": len(changed_rows),
                "improved_count": len([r for r in changed_rows if r["change_type"] == "improved"]),
                "regressed_count": len([r for r in changed_rows if r["change_type"] == "regressed"]),
            },
            "changed_rows": changed_rows,
        }
    )


@router.get("/runs")
async def list_eval_runs(
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """Returns list of last N evaluation runs for comparison."""
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    runs = manager.list_runs()
    return JSONResponse(content={"runs": runs})


@router.get("/runs/{job_id}/export/csv")
async def export_run_csv(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> Response:
    """Download per-image results CSV."""
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    try:
        csv_data = manager.export_csv(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="eval_{job_id}_results.csv"'},
    )


@router.get("/runs/{job_id}/export/json")
async def export_run_json(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> Response:
    """Download complete report JSON."""
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    try:
        json_data = manager.export_json(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="eval_{job_id}_report.json"'},
    )


@router.get("/runs/{job_id}/export/md")
async def export_run_md(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> Response:
    """Download summary Markdown report."""
    from app.config import get_settings
    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    try:
        md_data = manager.export_markdown(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return Response(
        content=md_data,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="eval_{job_id}_report.md"'},
    )


@router.post("/fusion")
async def analyze_fusion(
    payload: FusionRequestPayload,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> JSONResponse:
    """
    Offline multimodal fusion exploration over saved evaluation run diagnostics.
    Runs fast offline computation in a worker thread. Never calls TEI or SigLIP.
    """
    import asyncio
    import json
    from app.config import get_settings
    from app.eval.fusion import run_fusion_analysis

    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    run_file = manager.runs_dir / f"{payload.run_id}.json"
    if not run_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{payload.run_id}' not found.",
        )

    try:
        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read evaluation run JSON: {exc}",
        )

    # Fast offline computation in worker thread
    report = await asyncio.to_thread(run_fusion_analysis, run_data)
    return JSONResponse(content=report)


@router.get("/runs/{job_id}/export/fusion-md")
async def export_fusion_markdown(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> Response:
    """Download offline Multimodal Fusion Explorer Markdown report."""
    import asyncio
    import json
    from app.config import get_settings
    from app.eval.fusion import run_fusion_analysis

    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    run_file = manager.runs_dir / f"{job_id}.json"
    if not run_file.exists():
        raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found.")

    with open(run_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    report = await asyncio.to_thread(run_fusion_analysis, run_data)
    md_content = report.get("markdown_report", "")
    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="fusion_report_{job_id}.md"'},
    )


@router.get("/runs/{job_id}/export/fusion-json")
async def export_fusion_json(
    job_id: str,
    request: Request,
    _auth: None = Depends(_eval_auth),
) -> Response:
    """Download complete offline Multimodal Fusion Explorer JSON report."""
    import asyncio
    import json
    from app.config import get_settings
    from app.eval.fusion import run_fusion_analysis

    settings = getattr(request.app.state, "settings", None) or get_settings()
    manager = get_eval_job_manager(settings)

    run_file = manager.runs_dir / f"{job_id}.json"
    if not run_file.exists():
        raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found.")

    with open(run_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    report = await asyncio.to_thread(run_fusion_analysis, run_data)
    json_bytes = json.dumps(report, indent=2, default=str).encode("utf-8")
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="fusion_report_{job_id}.json"'},
    )

