# Photo-to-Work-Order Service (SigLIP)

Production-grade, standalone multimodal maintenance triage service replacing `photoToWorkOrder` as a single deployable unit.

It accepts photo uploads and/or maintenance descriptions, performs text ranking against a PostgreSQL `pgvector` service catalog, evaluates zero-shot visual similarity via Google's SigLIP 2 (`google/siglip2-base-patch16-224`), reranks candidate matches using an external Hugging Face Text Embeddings Inference (TEI) cross-encoder, and classifies requests into `auto_create`, `draft`, or `no_action`.

---

## 1. Architecture

```mermaid
flowchart TD
    Client["Client / Web UI / Mobile App"] -->|"POST /v1/posts/process"| SigLIP["SigLIP Standalone Service<br/>(FastAPI / PyTorch CPU)"]
    
    subgraph Storage["Existing PostgreSQL Database"]
        WoService["'WoService' (catalog + original 768d English text vectors)"]
        TextStore["'ServiceTextEmbedding' (text vectors per TEI model)"]
        ServiceImageProfile["'ServiceImageProfile' (768d SigLIP vectors)"]
        DecisionLog["'DecisionLog' (Triage audit & calibration)"]
    end

    subgraph TEI["External TEI Containers"]
        TEI_Embed["TEI Embedder<br/>(any model, e.g. BAAI/bge-m3 or BAAI/bge-base-en-v1.5 / Port 8081)"]
        TEI_Rerank["TEI Reranker<br/>(BAAI/bge-reranker-v2-m3 / Port 8082, disabled by default)"]
    end

    SigLIP -->|"1. Embed Query"| TEI_Embed
    SigLIP -->|"2. Text Search (pgvector cosine)"| WoService
    SigLIP -->|"3. Zero-shot Image Dot Product"| ServiceImageProfile
    SigLIP -->|"4. Cross-Encoder Logits (raw_scores=true)"| TEI_Rerank
    SigLIP -->|"5. Idempotent Audit Log"| DecisionLog
```

### Pipeline Execution Stages
1. **Text Ranking**: Query embedded via TEI `/embed` with whatever model TEI serves, as plain text with no instruction prefix. Cosine similarity computed against that model's catalog vectors in `"ServiceTextEmbedding"` (joined to active `"WoService"` rows) in PostgreSQL using `pgvector` (`w * desc_sim + (1-w) * title_sim`).
2. **Image Ranking**: Image encoded via in-process SigLIP 2 model, dot-producted against in-memory cached visual profiles. Non-visual services receive middle-rank and are excluded from the visual vote pool.
3. **Pool Merger**: Union of text top-N and image top-N, sorted by `min(text_rank, image_rank)`, capped at `pool_max` (default 10).
4. **Cross-Encoder Reranking**: Candidate pairs `(description, "{title}. {description}")` scored via TEI `/rerank` with `raw_scores: true`. Temperature-scaled softmax ($T=1.0$) is applied locally to preserve calibration against legacy thresholds.
5. **Decision Engine**:
   - Missing description: routed to `draft` (ranked by image only).
   - Word count $< 3$: routed to `draft`.
   - High-risk services (e.g. emergency response): mandatory `draft`.
   - Confidence $< 0.25$: routed to `no_action`.
   - Confidence $\ge 0.40$ and gap $\ge 0.10$: `auto_create` (or `draft` in shadow mode).
   - Default: `draft`.

---

## 2. API Specification

All application routes are versioned under `/v1` and documented via OpenAPI (`/docs`).

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1/posts/process` | Triage maintenance post (multipart/form-data) | Yes (Bearer) |
| `GET` | `/v1/services` | List service catalog entries | Yes (Bearer) |
| `PUT` | `/v1/services` | Bulk upsert services & regenerate embeddings | Yes (Bearer) |
| `DELETE` | `/v1/services/{id}` | Soft-delete a service (`isActive = false`) | Yes (Bearer) |
| `POST` | `/v1/decisions/{id}/feedback` | Record operator ground truth for calibration | Yes (Bearer) |
| `GET` | `/v1/decisions` | List recent triage decisions | Yes (Bearer) |
| `GET` | `/health/live` | Process liveness probe | No |
| `GET` | `/health/ready` | Component readiness check (DB, SigLIP, TEI) | No |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint | Optional |

### POST /v1/posts/process Example
```bash
curl -X POST http://localhost:8083/v1/posts/process \
  -H "Authorization: Bearer localdevkey" \
  -F "description=AC unit needs filter replacement urgently" \
  -F "location=Building A, Room 101" \
  -F "image=@photos/wo_13.jpg"
```

Response:
```json
{
  "decision_id": 70,
  "action": "draft",
  "would_be_action": "auto_create",
  "would_auto": true,
  "confidence": 0.4373,
  "gap": 0.1148,
  "agree": false,
  "reason": "Shadow mode: would be auto_create (conf=0.4373>=0.4, gap=0.1148>=0.1), routed to draft",
  "top_candidates": [
    {
      "service_id": 1,
      "title": "AC Services",
      "confidence": 0.4373,
      "confidence_pct": "43.7%",
      "text_rank": 1.0,
      "image_rank": null,
      "pool_rank": 1,
      "rerank_score": -3.134
    }
  ],
  "idempotent_replay": false,
  "shadow": true
}
```

---

## 3. Configuration Reference

All settings are managed via environment variables (or `.env` file):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | *(required)* | PostgreSQL connection string (`postgresql://user:pass@host:5432/db`) |
| `DB_SCHEMA` | `public` | Search path / schema for existing tables |
| `DB_POOL_MIN_SIZE` | `2` | Minimum async connection pool size |
| `DB_POOL_MAX_SIZE` | `10` | Maximum async connection pool size |
| `DB_TIMEOUT_SECONDS` | `10.0` | Connection timeout in seconds |
| `TEI_EMBED_URL` | `http://localhost:8081` | TEI text embedding endpoint |
| `TEI_RERANK_URL` | `http://localhost:8082` | TEI cross-encoder reranker endpoint |
| `TEI_API_KEY` | `None` | Optional Bearer token for TEI container auth |
| `TEI_TIMEOUT_SECONDS` | `10.0` | TEI request timeout |
| `TEI_MAX_RETRIES` | `2` | Maximum retry attempts for transient TEI connection drops |
| `TEXT_EMBED_MODEL_ID` | *(unset)* | Optional pin. Unset = use whatever model TEI serves (catalog re-embedded automatically). Set = TEI must serve exactly this model, else not-ready |
| `RERANK_MODEL_ID` | *(unset)* | Optional reranker label for logs; unset = read from the rerank container's `/info` (logged only when `ENABLE_RERANKER=true`) |
| `LEGACY_TEXT_MODEL_ID` | `BAAI/bge-base-en-v1.5` | Model of the original `WoService.titleEmb/descEmb` columns; they are only written when TEI serves this model |
| `SIGLIP_MODEL_ID` | `google/siglip2-base-patch16-224` | Hugging Face model identifier for vision encoder |
| `SIGLIP_TORCH_THREADS` | `2` | Number of PyTorch CPU inference threads |
| `SIGLIP_MAX_CONCURRENCY` | `2` | Semaphore limiting concurrent vision inference passes |
| `SIGLIP_LRU_CACHE_SIZE` | `256` | In-memory cache size for candidate text vectors |
| `PIPELINE_TEXT_DESC_WEIGHT` | `0.7` | Weight $w$ for description similarity ($1-w$ for title) |
| `PIPELINE_TEXT_TOP_N` | `10` | Candidates retrieved by text cosine similarity |
| `PIPELINE_IMAGE_TOP_N` | `5` | Candidates retrieved by zero-shot image similarity |
| `PIPELINE_POOL_MAX` | `10` | Maximum candidate pool size sent to reranker |
| `PIPELINE_RERANKER_TEMP` | `1.0` | Temperature for softmax score distribution |
| `DECISION_CONFIDENCE_THRESHOLD` | `0.40` | Minimum confidence $a$ for `auto_create` |
| `DECISION_GAP_THRESHOLD` | `0.10` | Minimum gap $g$ for `auto_create` |
| `DECISION_NO_ACTION_FLOOR` | `0.25` | Confidence floor below which requests are ignored |
| `DECISION_REQUIRE_AGREE` | `false` | If true, image rank #1 must match text rank #1 for auto |
| `AUTO_CREATE_ENABLED` | `false` | Enable automated creation (false = shadow mode) |
| `MIN_DESCRIPTION_WORDS` | `3` | Minimum words in description for auto eligibility |
| `ALWAYS_DRAFT_SERVICES` | `[...]` | JSON list of safety-critical services always routed to draft |
| `API_KEYS` | `""` | Comma-separated list of Bearer tokens (empty = dev mode / no auth) |
| `METRICS_API_KEY` | `None` | Optional Bearer token for `/metrics` scrape endpoint |
| `MAX_UPLOAD_BYTES` | `10485760` (10MB) | Maximum file size for image uploads |
| `STORE_DESCRIPTION` | `true` | Whether to persist text descriptions in `DecisionLog` |
| `ENABLE_DEBUG_ENDPOINTS` | `false` | Enable legacy `/embed-text`, `/embed-image`, `/rank` routes |
| `ENABLE_UI` | `false` | Enable web UI operator console at `/ui` |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `console` | `json` for structured log forwarding or `console` for dev |

---

## 4. Local Execution (PowerShell)

### Running the Service Locally
```powershell
cd c:\Users\Win11\code\shamixo\siglip
..\venv\Scripts\python.exe run.py --port 8083
```

### Probing Readiness
```powershell
Invoke-RestMethod http://localhost:8083/health/ready | ConvertTo-Json -Depth 5
```

### Running Test Suite
```powershell
cd c:\Users\Win11\code\shamixo
.\venv\Scripts\pytest.exe siglip/tests -v
```

### Text / Photo Agreement Strictness
With `DECISION_REQUIRE_AGREE=true`, a post is auto-created only when the description and the photo agree;
otherwise it becomes a draft. `DECISION_AGREE_MODE` sets how strict that is (all other gates — confidence,
gap, text guard, safety near-tie, always-draft — still apply):

| Level | Auto-create allowed when |
| :--- | :--- |
| `strict` (default) | photo #1 == text #1 |
| `balanced` | strict, OR text #1 is in the photo top-3 (`DECISION_AGREE_IMAGE_TOP_K`) and leads text #2 by >= 0.05 (`DECISION_AGREE_TEXT_MARGIN`) |
| `relaxed` | text #1 is in the photo top-3 |

- **Eval page** (`/ui/eval`): "Text / Photo Agreement Strictness" selector — applies to that eval run only and is
  recorded in the run's effective settings. CLI: `scripts\eval_model_compare.py run --agree-mode balanced`.
- **Operator console** (`/ui`): "Text / photo strictness (test only)" selector — sent as the `agree_mode` form
  field and honoured **only when `ENABLE_UI=true`**, for that request only. Production callers get the env default.
- The level used is returned as `agree_mode` and stored in `DecisionLog.models.agree_mode`.

66-row ground truth (reranker off), auto-created / wrong:
| Level | EN, bge-base-en-v1.5 | EN, bge-m3 | AR, bge-m3 |
| :--- | :--- | :--- | :--- |
| strict | 49 / 0 | 43 / 0 | 32 / 0 |
| balanced | 55 / 0 | 47 / 2 | 33 / 0 |
| relaxed | 55 / 0 | 52 / 4 | 41 / 3 |

---

## 5. Database Least-Privilege Role Setup

For production deployment, grant the service user minimal required privileges:

```sql
CREATE USER siglip_service WITH ENCRYPTED PASSWORD 'replace_with_strong_password';

GRANT USAGE ON SCHEMA public TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."WoService" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."ServiceImageProfile" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."DecisionLog" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."ServiceTextEmbedding" TO siglip_service;   -- per-model text vectors
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO siglip_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO siglip_service;

REVOKE CREATE ON SCHEMA public FROM siglip_service;
```

---

## 6. Text Embedding Models (multilingual branch)

This branch works with **whatever model the TEI embedder serves**. The catalog's text vectors are
stored per model in `"ServiceTextEmbedding"` (one row per service and model, like
`"ServiceImageProfile"` is keyed by `imageModel`):

| Column | Meaning |
| :--- | :--- |
| `serviceId`, `textModel` | primary key (`textModel` = TEI `/info` `model_id`) |
| `dim` | dimension measured from TEI (TEI `/info` has no dimension field; a probe `/embed` is used) |
| `titleEmb`, `descEmb` | unconstrained `vector` (any dimension) |
| `contentHash` | sha256 of the embedded texts; an edit to title/description/keywords triggers a re-embed |

Embedded texts (plain text, no prefix) follow the recipe of the original catalog vectors:
**title** and **`"{description}. {keywords}"`** (description alone when there are no keywords).

**Automatic re-embedding.** On startup the service embeds every service (active and inactive) that is
missing, stale or has another dimension for the current TEI model; a restart with the same model embeds
nothing. If the TEI model is switched while the service runs, `/health/ready` reports not-ready
("re-embedding catalog in the background"), starts one background re-embed, and is ready again when it
finishes (a few seconds for ~20 services). All writes happen in one transaction under an advisory lock.

**Original columns are kept.** `"WoService"."titleEmb"/"descEmb"` stay the original `vector(768)`
`BAAI/bge-base-en-v1.5` vectors used by the `main` branch and WoSemanticSearch. They are only written
when TEI serves `LEGACY_TEXT_MODEL_ID` with that dimension; with any other model, catalog writes leave
them unchanged and log a warning. `"ServiceImageProfile"` (SigLIP) is never touched.

### One-time setup (DB role allowed to CREATE TABLE)
```powershell
..\venv\Scripts\python.exe scripts\migrate_text_embeddings.py --dry-run
..\venv\Scripts\python.exe scripts\migrate_text_embeddings.py        # CREATE TABLE IF NOT EXISTS "ServiceTextEmbedding"
```

### Switching the embedding model
TEI compose files live in the `tei` folder: `docker-compose.yml` (English: bge-base-en-v1.5 +
bge-reranker-base) and `docker-compose.multilingual.yml` (bge-m3 + bge-reranker-v2-m3).
On a ~7.6 GB Docker VM bge-m3 (~5.5 GB resident) cannot run next to a second large model, so stop the
reranker first (it is disabled in the pipeline by default):
```powershell
docker compose stop tei-rerank
docker compose -f docker-compose.multilingual.yml up -d tei-embed   # or: docker compose up -d   (English)
```
Nothing else is needed: the running service re-embeds the catalog for the new model. To trigger it
by hand: `..\venv\Scripts\python.exe scripts\reembed_catalog.py` (`--force` re-embeds everything).

TEI settings for bge-m3 (`cpu-1.9`): `--pooling cls --max-batch-tokens 2048 --max-client-batch-size 32`.
`--max-batch-tokens 2048` is required on a ~7.6 GB Docker VM (8192 was OOM-killed during TEI warm-up);
with TEI's default auto-truncate this also caps inputs at 2048 tokens (descriptions are <= 1000 chars).

### Per-model text-guard thresholds
The text guard compares the top-1 text score (an absolute cosine value) with a no-action floor
and an auto-create minimum, so the values depend on the model. `TEXT_GUARD_PROFILES` (JSON) holds
per-model values; models without an entry use `DECISION_TEXT_NO_ACTION_FLOOR` / `DECISION_TEXT_AUTO_MIN`
(0.40 / 0.45, calibrated on bge-base). Eval runs record the values used (`text_guard_source`), and
explicit per-run overrides still win.

| Model | no_action_floor | auto_min | How it was chosen |
| :--- | :--- | :--- | :--- |
| `BAAI/bge-base-en-v1.5` | 0.40 | 0.45 | global defaults |
| `BAAI/bge-m3` | **0.35** | 0.45 | lowest real request across EN + AR ground truth and held-out sets scored 0.3604; 0.35 ignores none of the 250. auto_min 0.40-0.45 gave identical results; 0.48 would block the one Arabic wrong auto-create but turn 6 correct Arabic auto-creates into drafts |

Recalibrate after a model or catalog change: `..\venv\Scripts\python.exe scripts\calibrate_text_guard.py`
(read-only; uses `groundTruth.csv`, `groundTruth_ar.csv`, `eval/holdout_text_queries.csv`, `eval/calibration_ar.json`).

Note: posts **without a photo** are also subject to the confidence floor `DECISION_NO_ACTION_FLOOR` (0.25),
which was set on posts with photos. For text-only posts it ignores many real requests for **both** models
(bge-base: 9/62 test and 24/63 held-out English rows; bge-m3: 22/62 and 30/63) — see the calibration output.

### Results on the ground truth (reranker off, enriched catalog, per-model guard)
| Model | Language | text-only top-1 / top-3 | pipeline top-1 | auto precision | real requests ignored |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BAAI/bge-base-en-v1.5` | English | 95.16% / 98.39% | 95.16% | 97.56% | 0 |
| `BAAI/bge-base-en-v1.5` | Arabic | 8.06% / 19.35% | 32.26% | 100% (5 autos) | 4 |
| `BAAI/bge-m3` | English | 83.87% / 96.77% | 91.94% | 100% | 0 |
| `BAAI/bge-m3` | Arabic | 70.97% / 85.48% | 88.71% | 96.77% (1 wrong) | 1 (confidence floor) |

Compare runs with `scripts\eval_model_compare.py run --tag <name>` (outputs in
`app/eval/runs/model_compare/`) and `scripts\eval_model_compare.py compare <a.json> <b.json>`.
