# Project Status: SigLIP Standalone Photo-to-Work-Order Service

## Current Task
- **Eval Suite Upgrade: Run Record Snapshot, Additive Diagnostic Storage, Run Comparison, 3-Level NONE Reporting, Signal Quality AUROC Table & Reranker Presets: COMPLETED** — 2026-10-31
  - Task 1 (Run Record): Captured comprehensive settings snapshot for every evaluation run (overrides, effective pipeline parameters, model IDs, active catalog SHA-256 hash). Surfaced in UI, exports, and added visible "RERANKER OFF" banner when reranker is disabled.
  - Task 2 (Store More Per Row): Stored all active service text scores and image similarities, candidate raw reranker scores, explicit top-1 raw score, top-1 image similarity, and top-1 text similarity without breaking pipeline or past run formats.
  - Task 3 (Compare Runs): Developed side-by-side run comparison view matching rows by image filename, computing metric deltas (accuracy, pool recall, auto precision, wrong-auto, NONE rates, latency), flagging dataset and catalog hash mismatches, and presenting a filtered flips table (A wrong -> B right, A right -> B wrong) with full image preview.
  - Task 4 (NONE Reporting in 3 Levels): Implemented 3-level evaluation for NONE ground truth: Level 1 No Action (correct suppression), Level 2 Draft (soft false match), and Level 3 Auto-Create (harmful false match / wrong-auto). Unified NONE correctness check in single shared helper `is_row_correct()` used across job manager and metrics.
  - Task 5 (Signal Quality Table): Implemented non-parametric rank-based AUROC analysis with sample size `n` evaluating 5 diagnostic signals as predictors of (a) "top-1 is correct" on normal rows and (b) "normal row vs NONE row", warning when any class has fewer than 10 instances.
  - Task 6 (Preset Buttons): Added "Reranker ON (production)" and "Reranker OFF (bi-encoder)" preset buttons in the evaluation start form that exclusively control the `enable_reranker` override.

## Completed Work
- **Eval Suite Advanced Upgrades (Run Record, Additive Storage, Run Comparison, 3-Level NONE, Signal Quality, Presets)** — 2026-10-31:
  - **Requirement 1: Run Record & Snapshot Persistence**:
    - Created `compute_catalog_hash()` in `app/eval/job_manager.py` computing SHA-256 fingerprint over `title + description + visual_line` across all active services in the database.
    - Snapshot captured with every run in `job.effective_settings`: per-run overrides, effective values of `enable_reranker`, `description_weight`, `text_top_n`, `image_top_n`, `candidate_pool_max`, `reranker_temperature`, `biencoder_temperature`, `agreement_boost`, `confidence_threshold`, `gap_threshold`, `no_action_floor`, `require_agree`, `always_draft_services`, `min_description_words`, `auto_create_enabled`, text/vision/reranker `model_ids`, and `catalog_hash`.
    - Added visible amber `⚠️ RERANKER OFF — Bi-Encoder Only Evaluation Mode` banner in `/ui/eval` when `enable_reranker` is false.
    - Rendered responsive Run Record Snapshot grid in `/ui/eval`, included snapshot in Section 1 of Markdown reports (`report.md`) and JSON exports.
  - **Requirement 2: Store More Per Row (Additive Diagnostic Storage)**:
    - Added fields `all_service_text_scores` and `all_service_image_scores` to `ProcessedPost` (`app/pipeline/decision.py`) and populated them in `engine.py` when `return_diagnostics=True`.
    - Extended `EvalRowResult` (`app/eval/metrics.py`) with `all_service_text_scores`, `all_service_image_scores`, `top1_raw_score`, `top1_image_similarity`, and `top1_text_similarity`.
    - Preserved raw reranker logits on all candidate pool entries (`raw_score`).
    - Added safe dataclass field filtering and legacy backfill in `job_manager.py` to ensure all existing saved evaluation runs load cleanly without schema migration errors.
    - Added UI details expansion showing explicit top-1 diagnostic card and collapsible full-catalog active service text and image similarity table.
  - **Requirement 3: Compare Runs View & Delta Analytics**:
    - Implemented `GET /v1/eval/compare?run_a={id}&run_b={id}` endpoint in `app/routers/eval.py`.
    - Matched evaluation instances by image file name across runs.
    - Warns prominently if datasets differ (missing images on either side) or if active catalog hashes differ between runs.
    - Computes metric deltas ($B - A$) for Top-1/3/5 accuracy, candidate pool recall, auto-create precision, harmful wrong-auto count, coverage, NONE rates (L1, L2, L3), and latencies (mean, p50, p95).
    - Identifies changed rows (`A wrong → B right` improved vs. `A right → B wrong` regressed) with expected service, predictions, confidences, actions, and thumbnails.
    - Added interactive Compare section in `/ui/eval` with run selector dropdowns, warning alerts, metric deltas table (color-coded for regressions vs improvements), filter pills, and image preview modal triggers.
  - **Requirement 4: 3-Level NONE Ground Truth Reporting**:
    - Defined `classify_none_action(action)` in `app/eval/metrics.py` categorizing NONE outcomes:
      - **Level 1**: `no_action` (correctly suppressed).
      - **Level 2**: `draft` (soft false match; requires operator review).
      - **Level 3**: `auto_create` (harmful false match; critical wrong-auto).
    - Implemented single unified helper `is_row_correct(*, is_none, action, predicted_service, expected_service)` shared between `job_manager.py` worker execution and `metrics.py` summary calculation.
    - Highlighted Level 3 WRONG-AUTO with danger borders and red alerts in `/ui/eval`, Markdown reports, CSV exports, and Compare deltas.
  - **Requirement 5: Signal Quality AUROC Predictor Table**:
    - Implemented non-parametric rank-based Mann-Whitney U AUROC algorithm (`compute_auroc`) handling ties via mid-ranks without requiring external libraries.
    - Evaluated 5 signals across two classification tasks:
      1. Predicting "Top-1 is correct" on normal ground-truth rows.
      2. Predicting "Normal row vs NONE row" across all evaluated rows.
      - Signals: Softmax Confidence, Top-1 vs Top-2 Margin (Gap), Raw Top-1 Reranker Score (available when reranker is ON), Top-1 Text Similarity, and Top-1 Image Similarity.
    - Added small-class sample warnings (`< 10` instances in any binary class).
    - Rendered in dedicated "Signal Quality (AUROC)" tab in `/ui/eval` and Section 7 of Markdown exports.
  - **Requirement 6: Preset Buttons in Start Form**:
    - Added "🟢 Reranker ON (production)" and "⚪ Reranker OFF (bi-encoder)" preset buttons above the settings sliders in `/ui/eval`.
    - Preset buttons exclusively configure the per-run `enable_reranker` override without mutating any other configuration parameters or defaults.
- **Eval Suite Upgrade (NONE Ground Truth, New Eval Button & Large Image Lightbox)** — 2026-10-31:
  - **Task 1: Ground Truth "NONE" Handling**:
    - **Validator**: Parsed `NONE / none / None` (trimmed, case-insensitive) as `NONE` class without triggering unknown service warnings. Displayed `none_count` in validation report.
    - **Accuracy Metrics**: Strictly excluded `NONE` rows from Top-1, Top-3, Top-5 accuracy, candidate pool recall, per-service accuracy (`service_stats`), and calibration reliability bins. Prevents out-of-scope rows from artificially dragging down service accuracy (e.g. Top-1 increased from 77.6% to true 83.9% on normal ground truth).
    - **Dedicated NONE Panel**: Surface `n` (out-of-scope count), `NONE-correct rate` (% returned as `no action`), `false-match count` (count returned as `auto` or `draft`), wrong-auto count, and interactive false matches table with full diagnostics and preview triggers.
    - **Confidence Distribution Histogram**: HTML5 Canvas dual-bar histogram comparing top-1 confidence distribution of `NONE` rows versus correctly-answered normal rows across 10 bins ($0-10\% \dots 90-100\%$) for diagnostic threshold separability inspection.
    - **Threshold Sensitivity Grid Recomputation**: Replaced duplicated heuristic with direct invocation of pipeline's own `make_decision` function. Recomputes triage actions and records `none_false_matches` at each setting. Penalizes auto precision when NONE rows produce `auto_create`.
    - **Findings & Warnings**: Automatically generates summary findings reporting false-match count and actions when NONE rows return as `draft` or `auto_create`.
    - **Confusion Matrix**: Added dedicated `NONE` row and column mapping `expected NONE -> predicted service` (when auto/draft) and `expected service -> returned no action`.
    - **Exports**: Updated CSV, JSON, and Markdown export formats with explicit NONE metrics, `is_none`, `false_match`, and findings.
  - **Task 2: "New Evaluation" Button**:
    - Added prominent `＋ New Evaluation` button in page header.
    - Implemented active job check: prompts user for confirmation and stops background run via `POST /v1/eval/jobs/{id}/stop` if job is running.
    - Resets complete page state: terminates SSE connection, revokes object URLs, clears file input, search query, validation report, results table, summary KPI cards, and canvas charts.
    - Preserves all past run history in the runs selector dropdown.
  - **Task 3: Large Image Lightbox Modal**:
    - **Preview Persistence**: During evaluation worker processing, downscales and saves high-resolution previews (max 1280px on long side, JPEG quality 80) to `<EVAL_RUNS_DIR>/<job_id>/previews/<row_id>.jpg`.
    - **Authenticated Endpoint**: Added `GET /v1/eval/jobs/{job_id}/preview/{row_id}` with Bearer token authentication and path traversal protection.
    - **Blob URL Loading**: Client loads previews via authenticated `fetch()` with `sessionStorage` token, creating blob URLs with in-memory caching and automatic revocation on reset.
    - **Lightbox Modal Experience**: Dark blurred backdrop (`rgba(0, 0, 0, 0.88)`), responsive image scaling up to 90vw x 85vh (`object-fit: contain`), Esc / click-outside closing, left/right arrow navigation (keyboard + on-screen buttons) across visible filtered rows, and an informative metadata sidebar showing row ID, filename, expected service, predicted service, action badge, result status, confidence, gap, and description.
    - **Automatic Cleanup**: Integrated preview directory deletion into `_cleanup_old_runs()` when runs exceed `EVAL_MAX_HISTORY_RUNS`.
- **Bi-Encoder Gap Fix (`app/pipeline/engine.py` + `app/config.py`)** — 2026-10-30:
  - **Root cause**: When `ENABLE_RERANKER=false`, bi-encoder scores from text (cosine ~0.35–0.55) and image (SigLIP dot-product ~0.04–0.10) were blended as `0.7*t_s + 0.3*i_s`. This dragged down candidates that had both modalities, causing the gap between candidate #1 and #2 to collapse to ≈ 0, triggering 60/65 rows as DRAFT (gap < `gap_threshold=0.1`).
  - **Fix in `engine.py`**: New bi-encoder scoring path:
    1. **Adjusted score**: Use text cosine similarity as primary when description present (not the blended value). Image only contributes via an **agreement boost** (`PIPELINE_BIENCODER_AGREE_BOOST=0.05`) when image also ranks the same candidate (full boost if image_rank=1, half boost if image_rank≤3).
    2. **Softmax amplification**: Apply `softmax_with_temperature(raw_adjusted, temperature=PIPELINE_BIENCODER_TEMP=0.06)` to amplify the discriminative gap between top-1 and top-2, matching the output range of the cross-encoder.
    3. **Sort & assign**: After softmax, sort by softmax probability descending and derive `confidence` + `gap` from the top two candidates.
  - **New config keys** in `app/config.py`:
    - `PIPELINE_BIENCODER_TEMP: float = 0.06` — softmax temperature; lower = sharper gaps.
    - `PIPELINE_BIENCODER_AGREE_BOOST: float = 0.05` — score boost when text & image both agree on same candidate.
  - **Calibration result** (65-row eval, 87.7% Top-1 accuracy):
    - Before: 5 AUTO (7.7% coverage), 100% precision, 60 DRAFT
    - After: **44 AUTO (67.7% coverage), 100% precision**, 19 DRAFT, 2 NO_ACTION
- **Evaluation Suite Implementation (`/ui/eval` & `/v1/eval/*`)**:
  - **Security & Configuration**:
    - Gated behind `ENABLE_EVAL` (default `false`, set to `true` in local `.env` for testing).
    - All `/v1/eval/*` endpoints enforce standard Bearer token authentication via `_eval_auth` (using user-entered API key in UI or frictionless dev when `API_KEYS` is empty).
    - Dry-run pipeline mode: `engine.process_post` supports `dry_run=True`, skipping all writes to `DecisionLog`, `SearchLog`, or any database table.
    - ZIP safety validation: protects against zip-slip (path traversal check with `os.path.commonpath`), limits total uncompressed size (`EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES=500MB`), file count (`EVAL_MAX_IMAGE_COUNT=1000`), per-image size (`EVAL_MAX_IMAGE_BYTES=10MB`), extracts to a temporary directory, and guarantees cleanup with `shutil.rmtree` after the run.
  - **Dataset Ingestion & Validation Report**:
    - Pre-run validation report inspecting `ground_truth.csv` and extracted image assets.
    - Checks missing images, corrupt/unreadable images via PIL, duplicate rows, empty-description (image-only) count, and NONE rows count.
    - Validates service names against active database `WoService` catalog and provides string distance suggestions (closest active service title) for typos.
  - **Sequential Execution & SSE Streaming**:
    - Background job worker with sequential processing for clean, reproducible latency measurements.
    - Excludes initial warm-up request from aggregate latency and throughput statistics.
    - Real-time progress broadcasting over Server-Sent Events (`GET /v1/eval/jobs/{job_id}/stream`), supporting start, stop/cancellation, and reconnecting to active runs.
    - Per-request override thresholds in UI (`description_weight`, `confidence_threshold`, `gap_threshold`, `no_action_floor`) without altering server config defaults.
    - In-memory job state and disk persistence to `app/eval/runs/{job_id}.json` with past runs selector.
  - **Per-Image Result Table & Additive Pipeline Changes**:
    - Live table with embedded low-res base64 thumbnails, sortable columns, and multi-mode filters (All, Wrong Only, NONE Expected, Empty Description, Auto-Create Only, Wrong-Auto Only).
    - Displays row ID, thumbnail, filename, description, expected service, predicted top-1 service, triage action badge, confidence, gap, and agree signal.
    - Exposes candidate pool ranks: Text top-10 rank, Image top-5 rank, Reranker rank, and full catalog text & image ranks across all services.
    - Preserves raw reranker cross-encoder logits for offline analysis.
    - Measures stage timings: image decode, SigLIP encode, text embed (TEI), pgvector search, rerank (TEI), decision, and total.
    - Expandable detail view showing complete candidate pool table, full catalog diagnostics, and stage latency breakdowns.
  - **Summary Panel & Analytics**:
    - Top KPI cards: Top-1, Top-3, Top-5 accuracy, Candidate Pool Recall, Auto-Create precision, and prominently highlighted WRONG-AUTO count.
    - Latency statistics (mean, p50 median, p95, max total) and mean per stage rendered as a responsive HTML5 Canvas bar chart.
    - Throughput calculation (images per second).
    - Per-service accuracy table with Wilson 95% score confidence intervals and `n < 5` small-sample warnings.
    - Top confusions list and confusion matrix heatmap.
    - Threshold sensitivity grid (Coverage vs Precision across confidence [0.3-0.9] and gap [0.0-0.2]).
    - Offline temperature calibration sweep ([0.5, 1, 2, 3, 5, 8]) calculating Expected Calibration Error (ECE), reliability diagram on Canvas, and bin tables with `n < 30` warnings.
  - **Data Exports**:
    - `results.csv`: Complete per-image evaluation results.
    - `report.json`: Full run configuration, summary metrics, and image details.
    - `report.md`: Markdown summary report with one-click clipboard copying.
  - **Local Dev Auth Removal**:
    - Cleared `API_KEYS=` in `siglip/.env` (backend automatically disables Bearer token enforcement when `API_KEYS` is empty).
    - Removed API key modal and button from `siglip/app/static/index.html` and cleaned up token handling in `siglip/app/static/app.js`.
    - Restarted service and verified all `/v1` endpoints and `/ui` respond without requiring any authentication headers.
- **Dataset Maintenance & Evaluation Package**:
  - Renumbered `image` column sequentially matching actual extensions in `siglip/photos/` (`img_001.jpg` - `img_067.jpg`, accounting for `img_019.webp` and `img_036.png`).
  - Updated [groundTruth.csv](file:///c:/Users/Win11/code/shamixo/siglip/groundTruth.csv) in-place and synchronized [updated_file.csv](file:///c:/Users/Win11/code/shamixo/siglip/updated_file.csv).
  - Built validated evaluation dataset archive [eval_dataset.zip](file:///c:/Users/Win11/code/shamixo/siglip/eval_dataset.zip) and updated `Downloads/eval.zip` (100% pass: 67/67 valid rows, 0 errors, 0 missing images).
- **Diagnostic Text Search Bug Fix (`app/db/repository.py`)**:
  - Fixed syntax bug in `RANKING_ALL_SQL` where `SELECT s.id, s.title, s.description` referenced alias `s` on an un-aliased `"WoService"` table.
  - This bug previously caused silent failures in text ranking during diagnostic runs (`return_diagnostics=True` in `/ui/eval`), resulting in 0.0 ms pgvector time, 0 text candidates, and 0.0% Text Top-1 Accuracy. Verified fixed with real embedding test (`AC Services` score 0.498).
- **Reranker Bypass & Evaluation Toggle**:
  - Added `ENABLE_RERANKER: bool = True` setting to `app/config.py`.
  - Added `enable_reranker` override support in `engine.py`, `eval.py` (`StartJobPayload`), and `job_manager.py`.
  - Added interactive "Cross-Encoder Reranker" checkbox toggle in `/ui/eval` under Run Configuration to allow comparing performance with vs. without the cross-encoder reranker directly.
  - Fixed confidence score calibration in bi-encoder mode: previously applying $T=1.0$ softmax over 10 cosine similarities in $[0.3, 0.5]$ flattened probabilities to $\sim 10\%$, falling below `no_action_floor` ($25\%$) and triggering 100% false `no_action`. Directly using the bi-encoder similarity match score restores accurate triage actions (yielding 87.7% Top-1 accuracy and 96.7% auto-create precision without reranker).
  - Moved `<input type="file" id="zipFileInput">` outside of `.dropzone` to eliminate DOM click bubbling recursion that caused browsers to suppress native file chooser dialogs.
  - Added `.dropzone * { pointer-events: none; }` to eliminate child element `dragleave` flicker during dragging.
  - Added window-level `dragover` and `drop` `preventDefault()` to prevent browsers from navigating away or rejecting drops.
  - Fixed detached input bug where `dropzone.innerHTML` previously destroyed the file input on upload state changes.
  - Added input value reset on change to allow re-uploading the same ZIP.
  - Added `POST /v1/eval/load-local` endpoint supporting direct file path loading (`.zip` or folder) from host machine without HTTP upload overhead.
  - Added "⚡ Quick Load Local Dataset" button and local path input to `/ui/eval`.
  - Added VS Code drag-and-drop handler in `eval.js` that converts dropped `file:///` text URIs into direct local load calls.
- **Part 1: Dependency Check on `photoToWorkOrder`**:
  - Full codebase audit across `siglip/`: zero runtime imports of `ptwo` or `photoToWorkOrder`, zero path-based requirements, zero hard-coded file paths into sibling folders.
- **Part 2: Static Operator Console Porting (`/ui`)**:
  - Ported static operator console (`index.html`, `styles.css`, `app.js`) to `siglip/app/static/`.

## Remaining Work (Deferred to Coolify Deployment Phase)
- Build and verify Linux x86_64 Docker container via GHCR.
- Apply production least-privilege PostgreSQL user grants.
- Configure Coolify Docker network attachment.
- Set production `API_KEYS` environment variable.

## Known Issues
- None.

## Next Recommended Task
- Start service (`python run.py --port 8083`) and run evaluation sweeps in `/ui/eval`.
- Use the new "Compare Runs" view to evaluate Reranker ON vs. Reranker OFF across candidate thresholds.
- Delete test post `decision_id: 1` from `DecisionLog` if not already cleaned up.
