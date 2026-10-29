# Photo-to-Work-Order Service (SigLIP)

Production-grade, standalone multimodal maintenance triage service replacing `photoToWorkOrder` as a single deployable unit.

It accepts photo uploads and/or maintenance descriptions, performs text ranking against a PostgreSQL `pgvector` service catalog, evaluates zero-shot visual similarity via Google's SigLIP 2 (`google/siglip2-base-patch16-224`), reranks candidate matches using an external Hugging Face Text Embeddings Inference (TEI) cross-encoder, and classifies requests into `auto_create`, `draft`, or `no_action`.

---

## 1. Architecture

```mermaid
flowchart TD
    Client["Client / Web UI / Mobile App"] -->|"POST /v1/posts/process"| SigLIP["SigLIP Standalone Service<br/>(FastAPI / PyTorch CPU)"]
    
    subgraph Storage["Existing PostgreSQL Database"]
        WoService["'WoService' (768d text vectors)"]
        ServiceImageProfile["'ServiceImageProfile' (768d SigLIP vectors)"]
        DecisionLog["'DecisionLog' (Triage audit & calibration)"]
    end

    subgraph TEI["External TEI Containers"]
        TEI_Embed["TEI Embedder<br/>(BAAI/bge-base-en-v1.5 / Port 8081)"]
        TEI_Rerank["TEI Reranker<br/>(BAAI/bge-reranker-base / Port 8082)"]
    end

    SigLIP -->|"1. Embed Query"| TEI_Embed
    SigLIP -->|"2. Text Search (pgvector cosine)"| WoService
    SigLIP -->|"3. Zero-shot Image Dot Product"| ServiceImageProfile
    SigLIP -->|"4. Cross-Encoder Logits (raw_scores=true)"| TEI_Rerank
    SigLIP -->|"5. Idempotent Audit Log"| DecisionLog
```

### Pipeline Execution Stages
1. **Text Ranking**: Query embedded via TEI `/embed` with instruction prefix (`Represent this sentence for searching relevant passages: `). Cosine similarity computed against `"WoService"` in PostgreSQL using `pgvector` (`w * desc_sim + (1-w) * title_sim`).
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

---

## 5. Database Least-Privilege Role Setup

For production deployment, grant the service user minimal required privileges:

```sql
CREATE USER siglip_service WITH ENCRYPTED PASSWORD 'replace_with_strong_password';

GRANT USAGE ON SCHEMA public TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."WoService" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."ServiceImageProfile" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."DecisionLog" TO siglip_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO siglip_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO siglip_service;

REVOKE CREATE ON SCHEMA public FROM siglip_service;
```
