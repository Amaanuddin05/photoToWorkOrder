# System Design & Schema Audit: Standalone Photo-to-Work-Order Service

**Document Version:** 1.0.0  
**Date:** 2026-10-30  
**Target Repository:** `siglip/` (replacing `photoToWorkOrder` as a deployable unit)  
**Deployment Target:** Linux x86_64, CPU-only, orchestrated via Coolify / Docker from GHCR  

---

## 1. Executive Summary & System Architecture

### 1.1 Objective
Transform `siglip/` into the complete, standalone, production-grade photo-to-work-order service. The service encapsulates all incoming post processing, multi-stage retrieval, reranking, decision logic, and database persistence into a single containerized microservice. It externalizes heavy language model inference to dedicated Text Embeddings Inference (TEI) containers over HTTP and integrates with an existing PostgreSQL database with `pgvector`.

### 1.2 Architecture Overview
The system follows a clean hexagonal (ports and adapters) architecture:
- **Core Pipeline Engine:** Pure, framework-agnostic pipeline stages (text ranking, zero-shot image ranking, pool merging, cross-encoder reranking, rule-based decision engine) with abstract interfaces.
- **Vision Inference Adapter:** Local CPU inference using Hugging Face `transformers` for `google/siglip2-base-patch16-224`, generating 768-dimensional normalized image embeddings and text embeddings (max 64 tokens).
- **Text Embedding & Reranker Adapters:** Asynchronous HTTP client (`TEIClient`) communicating with separate TEI containers:
  - `TEI Embedder`: `BAAI/bge-base-en-v1.5` for 768-dim text embeddings (`/embed`).
  - `TEI Reranker`: `BAAI/bge-reranker-base` (or `bge-reranker-v2-m3`) for cross-encoder pair scoring (`/rerank`).
- **Database Adapter:** Async PostgreSQL repository using `psycopg` connection pooling and `pgvector` for cosine distance calculations (`<=>`) against pre-embedded service catalog vectors.
- **HTTP Layer:** FastAPI with strict Pydantic v2 schemas, OpenAPI documentation, structured JSON logging, constant-time API key verification, Prometheus metrics, and Kubernetes/Coolify health probes.

### 1.3 Architecture Diagram

```mermaid
flowchart TD
    Client([Client / Frontend / Worker]) -->|POST /v1/posts/process| API[FastAPI /v1 API Layer]
    API --> Auth[API Key Auth & Rate Limiter]
    Auth --> Pipeline[Pipeline Engine]

    subgraph External TEI Services [TEI HTTP Microservices]
        TEI_E[TEI Embedder: bge-base-en-v1.5<br>:8081 /embed]
        TEI_R[TEI Reranker: bge-reranker-base<br>:8082 /rerank]
    end

    subgraph Local In-Process [SigLIP Microservice Container]
        SigLIP_M[SigLIP 2 Vision Model<br>CPU torch.inference_mode<br>google/siglip2-base-patch16-224]
        MemCache[(In-Memory Cache<br>Service Image Profiles)]
    end

    subgraph Database [Existing External PostgreSQL 18 + pgvector]
        T_S[WoService<br>19 rows, titleEmb/descEmb]
        T_IP[ServiceImageProfile<br>19 rows, textEmb]
        T_DL[DecisionLog<br>Telemetry & Feedback]
    end

    Pipeline -->|1. Text Query| TEI_E
    TEI_E -->|Query Vector 768d| Pipeline
    Pipeline -->|2. Cosine Distance <=>| T_S
    Pipeline -->|3. Encode Image| SigLIP_M
    SigLIP_M -->|4. Cosine Dot Product| MemCache
    Pipeline -->|5. Merge Pool <= 10| Pipeline
    Pipeline -->|6. Rerank Pairs| TEI_R
    TEI_R -->|Raw Logits -> Softmax T=1.0| Pipeline
    Pipeline -->|7. Business Decision Rules| Pipeline
    Pipeline -->|8. Persist Telemetry| T_DL
    Pipeline --> API
    API --> Client
```

---

## 2. Live Schema Audit & Drift Report

### 2.1 Inspection Methodology & Environment
A read-only live schema inspection was executed on 2026-10-30 against the active local database instance (`wo_search`).
- **PostgreSQL Version:** `PostgreSQL 18.6 on x86_64-windows, compiled by msvc-19.44.35229, 64-bit`
- **pgvector Extension Version:** `vector 0.8.7` installed in schema `public`
- **Database Schemas:** `public` (all tables reside in the `public` schema)
- **Active Tables Inspected:** `"WoService"`, `"ServiceImageProfile"`, `"DecisionLog"`, `"SearchLog"`

### 2.2 Table-by-Table Schema Inspection

#### Table 1: `"WoService"`
Stores the canonical service catalog with text bi-encoder embeddings (`BAAI/bge-base-en-v1.5`).
- **Row Count:** 19 rows (all `isActive = true`)
- **Columns:**
  | Column Name | Data Type | UDT | Nullable | Default | Notes |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | `id` | integer | `int4` | NO | `nextval('"WoService_id_seq"'::regclass)` | Primary Key |
  | `title` | text | `text` | NO | None | Unique service title |
  | `description` | text | `text` | NO | None | Catalog description |
  | `keywords` | text | `text` | YES | None | Comma-separated search keywords |
  | `"isActive"` | boolean | `bool` | NO | `true` | Quoted CamelCase identifier |
  | `"embeddingModel"` | text | `text` | YES | None | Quoted CamelCase (`'BAAI/bge-base-en-v1.5'`) |
  | `"updatedAt"` | timestamptz | `timestamptz` | NO | `now()` | Quoted CamelCase |
  | `"titleEmb"` | USER-DEFINED | `vector(768)` | YES | None | Quoted CamelCase; dimension = 768 |
  | `"descEmb"` | USER-DEFINED | `vector(768)` | YES | None | Quoted CamelCase; dimension = 768 |
- **Indexes:**
  - `"WoService_pkey"`: `UNIQUE INDEX ON "WoService" USING btree (id)`
  - `"WoService_title_key"`: `UNIQUE INDEX ON "WoService" USING btree (title)`
- **Constraints:**
  - `WoService_pkey` (PRIMARY KEY) on `id`
  - `WoService_title_key` (UNIQUE) on `title`

#### Table 2: `"ServiceImageProfile"`
Stores zero-shot visual line profiles with vision-model text encoder embeddings (`SigLIP 2`).
- **Row Count:** 19 rows (all for `imageModel = 'google/siglip2-base-patch16-224'`)
- **Columns:**
  | Column Name | Data Type | UDT | Nullable | Default | Notes |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | `"serviceId"` | integer | `int4` | NO | None | Quoted CamelCase; FK $\to$ `"WoService"(id)` |
  | `"imageModel"` | text | `text` | NO | None | Quoted CamelCase; Composite PK component |
  | `"visualLine"` | text | `text` | YES | None | Quoted CamelCase; NULL for non-visual services |
  | `"textUsed"` | text | `text` | NO | None | Quoted CamelCase; string fed to text encoder |
  | `truncated` | boolean | `bool` | NO | `false` | **Unquoted lowercase**; > 64 tokens flag |
  | `"textEmb"` | USER-DEFINED | `vector` | YES | None | Quoted CamelCase; unconstrained dim in DDL |
  | `"updatedAt"` | timestamptz | `timestamptz` | YES | `now()` | Quoted CamelCase |
- **Indexes:**
  - `"ServiceImageProfile_pkey"`: `UNIQUE INDEX ON "ServiceImageProfile" USING btree ("serviceId", "imageModel")`
- **Constraints:**
  - `ServiceImageProfile_pkey` (PRIMARY KEY) on `("serviceId", "imageModel")`
  - `ServiceImageProfile_serviceId_fkey` (FOREIGN KEY) on `"serviceId"` $\to$ `"WoService"(id)` `ON DELETE CASCADE`

#### Table 3: `"DecisionLog"`
Telemetry store capturing every incoming post inference run, candidate scores, and operator review feedback.
- **Row Count:** 65 rows
- **Columns:**
  | Column Name | Data Type | UDT | Nullable | Default | Notes |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | `id` | integer | `int4` | NO | `nextval('"DecisionLog_id_seq"'::regclass)` | Primary Key |
  | `"createdAt"` | timestamptz | `timestamptz` | YES | `now()` | Quoted CamelCase |
  | `description` | text | `text` | YES | None | Free-text description input |
  | `"photoRef"` | text | `text` | YES | None | Path to photo on disk (if stored) |
  | `"photoHash"` | text | `text` | YES | None | SHA-256 hex digest of image bytes |
  | `location` | text | `text` | YES | None | Optional facility location string |
  | `"textTop"` | jsonb | `jsonb` | YES | None | Top 10 text candidates JSON |
  | `"imageTop"` | jsonb | `jsonb` | YES | None | Top 5 image candidates JSON |
  | `pool` | jsonb | `jsonb` | YES | None | Merged candidate pool ($\le 10$) JSON |
  | `"rerankScores"` | jsonb | `jsonb` | YES | None | Reranked candidates with softmax probs |
  | `"topServiceId"` | integer | `int4` | YES | None | FK $\to$ `"WoService"(id)` |
  | `confidence` | double precision| `float8` | YES | None | Top-1 reranker softmax probability |
  | `gap` | double precision| `float8` | YES | None | Top-1 minus Top-2 confidence |
  | `agree` | boolean | `bool` | YES | None | Image top-1 == Reranker top-1 |
  | `action` | text | `text` | NO | None | Executed action: `auto_create`, `draft`, `no_action` |
  | `"wouldBeAction"` | text | `text` | NO | None | Action without shadow mode override |
  | `reason` | text | `text` | YES | None | Human-readable explanation of decision |
  | `models` | jsonb | `jsonb` | NO | None | Model identifiers dictionary |
  | `"configVersion"` | text | `text` | NO | None | SHA-256 version hash of config |
  | `timings` | jsonb | `jsonb` | YES | None | Per-stage latencies in milliseconds |
  | `"finalServiceId"` | integer | `int4` | YES | None | FK $\to$ `"WoService"(id)` (human ground truth) |
  | `"finalAt"` | timestamptz | `timestamptz` | YES | None | Timestamp of operator review feedback |
  | `"poolContainsFinal"` | boolean | `bool` | YES | None | Evaluation metric: was final in candidate pool |
- **Indexes:**
  - `"DecisionLog_pkey"`: `UNIQUE INDEX ON "DecisionLog" USING btree (id)`
- **Constraints:**
  - `DecisionLog_pkey` (PRIMARY KEY) on `id`
  - `DecisionLog_topServiceId_fkey` (FOREIGN KEY) on `"topServiceId"` $\to$ `"WoService"(id)` `ON DELETE NO ACTION`
  - `DecisionLog_finalServiceId_fkey` (FOREIGN KEY) on `"finalServiceId"` $\to$ `"WoService"(id)` `ON DELETE NO ACTION`

#### Table 4: `"SearchLog"`
Created and used exclusively by `WoSemanticSearch` for text search query analytics.
- **Row Count:** 5 rows
- **Columns:** `id` (int, PK), `query` (text), `"topServiceId"` (int, FK), `"chosenServiceId"` (int, FK), `"createdAt"` (timestamptz).
- **Status:** Maintained as read-only/unmodified by this service.

---

### 2.3 Comprehensive Drift & Inconsistency Analysis

1. **Missing Index (`idx_woservice_isactive`):**
   - In `WoSemanticSearch/sql/001_schema.sql`, an index `CREATE INDEX IF NOT EXISTS "idx_woservice_isactive" ON "WoService" ("isActive");` is defined.
   - **Finding:** This index does **not** exist in the live database.
   - **Impact:** At 19 rows, the query planner performs a sequential scan in < 0.1 ms regardless of index presence. However, schema verification must note this discrepancy.

2. **Absence of Approximate Nearest Neighbor (ANN) Vector Indexes:**
   - Neither `WoService` (`"titleEmb"`, `"descEmb"`) nor `ServiceImageProfile` (`"textEmb"`) have HNSW or IVFFlat vector indexes.
   - **Reason & Assessment:** The catalog contains 19 services. Exact nearest neighbor calculation using pgvector's `<=>` operator computes 19 dot products in memory in under 0.2 ms with 100% precision. Constructing an IVFFlat index on < 1000 rows is discouraged by pgvector docs because lists would be empty or underfilled. HNSW is also unnecessary overhead at this scale.

3. **Identifier Casing & Mixed Quoting Inconsistency:**
   - PostgreSQL treats unquoted identifiers as lowercase.
   - In `WoService`: `id`, `title`, `description`, `keywords` are lowercase; `"isActive"`, `"embeddingModel"`, `"updatedAt"`, `"titleEmb"`, `"descEmb"` are CamelCase and **require double quotes**.
   - In `ServiceImageProfile`: `"serviceId"`, `"imageModel"`, `"visualLine"`, `"textUsed"`, `"textEmb"`, `"updatedAt"` are CamelCase and require double quotes; however, `truncated` is **lowercase unquoted**.
   - In `DecisionLog`: Mixed casing across 23 columns (e.g. `description` vs `"photoRef"`).
   - **Resolution:** The new service repository must explicitly double-quote all table names (`"WoService"`, `"ServiceImageProfile"`, `"DecisionLog"`) and column names in all SQL queries to prevent lowercase folding errors.

4. **Vector Dimension Constraints in DDL:**
   - In `WoService`: `"titleEmb"` and `"descEmb"` are defined as `vector(768)` (strict dimension enforced by PostgreSQL).
   - In `ServiceImageProfile`: `"textEmb"` is defined as `vector` without an explicit dimension parameter (`vector` vs `vector(768)`).
   - **Finding:** While the column is unconstrained in DDL, `SELECT DISTINCT vector_dims("textEmb") FROM "ServiceImageProfile"` confirms that all 19 rows contain exactly 768 dimensions matching `google/siglip2-base-patch16-224`.

5. **Foreign Key Deletion Constraint (`ON DELETE NO ACTION`):**
   - `"ServiceImageProfile"."serviceId"` has `ON DELETE CASCADE` $\to$ deleting a service cascades to its image profile.
   - `"DecisionLog"."topServiceId"` and `"finalServiceId"` have `ON DELETE NO ACTION` (default restrict) $\to$ hard deletion (`DELETE FROM "WoService" WHERE id = %s`) will **fail** if the service has ever been referenced in historical telemetry.
   - **Resolution:** Catalog deletion via `DELETE /v1/services/{id}` must execute a **soft-delete** (`UPDATE "WoService" SET "isActive" = FALSE WHERE id = %s`).

6. **Absence of Dedicated `idempotencyKey` Column in `DecisionLog`:**
   - The user request specifies: `POST /v1/posts/process: post_id / Idempotency-Key. Idempotent: the same key returns the stored decision and does not create a duplicate.`
   - **Finding:** `DecisionLog` has no `idempotencyKey` or `post_id` column.
   - **Zero-DDL Production Solution:** The `DecisionLog` table includes a `models` column of type `jsonb`. We can store `{"text": ..., "image": ..., "reranker": ..., "idempotency_key": "<key>"}` inside `models`. Idempotency lookup is performed via `SELECT ... FROM "DecisionLog" WHERE models->>'idempotency_key' = %s ORDER BY id DESC LIMIT 1`.
   - **Proposed Reviewed Migration:** For optimal query performance and a unique constraint, a separate migration script `002_decision_log_idempotency.sql` is provided below for user review (NOT applied automatically).

---

## 3. What to Port 1:1 vs What to Change & Why

### 3.1 Ported 1:1 (Preserving Proven Pipeline & Calibration)
1. **Text Ranking Logic:**
   - Direct cosine similarity computation in PostgreSQL using pgvector `<=>`.
   - Composite scoring formula: `score = w * desc_sim + (1 - w) * title_sim` with default `desc_weight = 0.7`.
   - Top 10 candidate retrieval.
2. **Zero-Shot Image Ranking:**
   - Pre-computed `ServiceImageProfile` text vectors (`f"{title}. {visualLine}"` or `{title}`).
   - Dot product of normalized SigLIP image embedding with normalized text vectors.
   - Middle-rank rule: Services with no `visualLine` receive neutral rank `(N + 1) / 2.0 = 10.0`, score `0.0`, and are **never** added to the candidate pool by the image ranker.
   - Top 5 candidate retrieval.
3. **Pool Formation & Tie Breaking:**
   - Union of text top 10 and image top 5.
   - Sort key: `(min(text_rank, image_rank), text_rank, image_rank, service_title)`.
   - Strict cap at `pool_max = 10`.
4. **Cross-Encoder Reranking:**
   - Query pairs: `(description, f"{candidate.title}. {candidate.description}")`.
   - Numerically stable temperature-scaled softmax over pool logits: $P(c_i) = \frac{e^{s_i / T}}{\sum_j e^{s_j / T}}$ with $T = 1.0$.
5. **Operational Decision Policy (Sequential Business Rules):**
   - **Rule 1 (Missing description):** `action = "draft"`, `would_be_action = "draft"`, ranked by image only, `confidence = None`, `gap = None`.
   - **Rule 2 (Short description):** Words $< 3 \to \text{"draft"}$.
   - **Rule 3 (Always draft services):** `"Emergency Response & Area Securing"`, `"Emergency Reactive Cleaning (Spills, Vomit, Incidents)"` $\to \text{"draft"}$.
   - **Rule 4 (Confidence floor):** $\text{confidence} < 0.25 \to \text{"no\_action"}$.
   - **Rule 5 (Auto-create qualification):** $\text{confidence} \ge a (0.40) \land \text{gap} \ge g (0.10) \land (\text{agree} \lor \neg \text{require\_agree}) \to \text{would\_be\_action} = \text{"auto\_create"}$.
     - If `AUTO_CREATE_ENABLED=true` $\to \text{action} = \text{"auto\_create"}$.
     - If `AUTO_CREATE_ENABLED=false` (shadow mode default) $\to \text{action} = \text{"draft"}$, reason notes shadow routing.
   - **Rule 6 (Default):** Otherwise $\to \text{"draft"}$.
6. **Telemetry & Feedback Recording:**
   - Full serialization of `textTop`, `imageTop`, `pool`, `rerankScores`, `timings`, and models to `DecisionLog`.
   - Operator feedback endpoint updating `finalServiceId`, `finalAt`, and `poolContainsFinal`.

### 3.2 What to Change & Why (Engineering Rationale)

| Aspect | `photoToWorkOrder` (Legacy) | New `siglip/` Service | Rationale |
| :--- | :--- | :--- | :--- |
| **Model Hosting** | In-process PyTorch for BGE bi-encoder, SigLIP, and BGE reranker (~6 GB RAM). | SigLIP in-process; TEI Embedder & Reranker called via HTTP over internal Docker network. | Drastically reduces memory footprint to ~2.1 GB RSS. Allows independent scaling and caching of TEI text models. |
| **Database Driver** | Synchronous `psycopg.connect()` on every HTTP request. | Pooled async `psycopg` (`AsyncConnectionPool`) with timeout and max pool size from env. | Eliminates connection establishment latency; prevents connection exhaustion under burst traffic. |
| **API Versioning** | Root paths (`/process-post`, `/decisions/{id}/final`). | Versioned `/v1` endpoints (`/v1/posts/process`, `/v1/decisions/{id}/feedback`). | Follows enterprise REST API standards; enables non-breaking future migrations. |
| **Catalog Management**| Offline Python script `build_service_profiles.py` requiring direct DB script execution. | RESTful `/v1/services` (GET, PUT bulk upsert, DELETE) with automatic idempotent content hashing. | Allows Coolify/web UI to manage services dynamically via HTTP without bash/CLI access. |
| **Idempotency** | None (duplicate requests created duplicate rows in `DecisionLog`). | Header `Idempotency-Key` or body `post_id`. Duplicate returns cached decision without re-inference or duplicate logging. | Prevents duplicate work orders and race conditions from retrying network clients. |
| **Configuration** | Static YAML file (`config/pipeline.yaml`). | Strict `pydantic-settings` reading from `.env` with validation at startup. | Twelve-Factor App standard for 12-factor cloud deployment (Coolify / Docker). |
| **Observability** | Ad-hoc Python logging. | Structured JSON logs with correlation IDs (`X-Request-ID`), Prometheus `/metrics`, and separated liveness/readiness probes. | Production-grade observability for Coolify and Grafana. |
| **Security & Auth** | Single API key, or no auth. | Comma-separated `API_KEYS` supporting seamless key rotation; constant-time digest comparison. | Zero-downtime key rotation without modifying code. |
| **Image Protection** | Unchecked Pillow loading. | `Image.MAX_IMAGE_PIXELS` set, content size limits, format validation (JPEG/PNG/WebP), memory stream safety. | Protects against decompression-bomb and DoS exploits. |

---

## 4. Final API Specification & Existing Table Mapping

All endpoints (except probes and `/metrics`) require the `Authorization: Bearer <API_KEY>` header.

### 4.1 `POST /v1/posts/process`
Ingests a post (photo and/or description) and returns the operational work order triage decision.

- **Content-Type:** `multipart/form-data`
- **Headers:** `Idempotency-Key` (string, optional)
- **Form Fields:**
  - `image`: File (optional, max 10MB, JPEG/PNG/WebP)
  - `description`: String (optional, max 2000 chars)
  - `post_id`: String (optional; fallback if `Idempotency-Key` header omitted)
  - `location`: String (optional; stored in `DecisionLog`, not used in ranking)
  *Validation Rule:* At least one of `image` or `description` must be provided.
- **Response Model (`200 OK`):**
  ```json
  {
    "decision_id": 128,
    "action": "draft",
    "would_auto": true,
    "confidence": 0.8421,
    "gap": 0.4512,
    "agree": true,
    "reason": "Shadow mode: would be auto_create (conf=0.8421>=0.40, gap=0.4512>=0.10), routed to draft",
    "top_candidates": [
      {
        "service_id": 1,
        "title": "AC Services",
        "confidence": 0.8421,
        "confidence_pct": "84.2%",
        "text_rank": 1,
        "image_rank": 1,
        "pool_rank": 1,
        "rerank_score": 3.84
      },
      {
        "service_id": 4,
        "title": "HVAC Filter Maintenance",
        "confidence": 0.0821,
        "confidence_pct": "8.2%",
        "text_rank": 2,
        "image_rank": 3,
        "pool_rank": 2,
        "rerank_score": -0.12
      },
      {
        "service_id": 7,
        "title": "Electrical Maintenance",
        "confidence": 0.0315,
        "confidence_pct": "3.2%",
        "text_rank": 4,
        "image_rank": null,
        "pool_rank": 3,
        "rerank_score": -1.05
      }
    ],
    "idempotent_replay": false
  }
  ```
- **Mapping to Existing Tables:**
  - Inserts 1 row into `"DecisionLog"`.
  - `description` $\to$ `"DecisionLog".description`
  - `photoHash` (SHA-256) $\to$ `"DecisionLog"."photoHash"`
  - `location` $\to$ `"DecisionLog".location`
  - `textTop` $\to$ `"DecisionLog"."textTop"`
  - `imageTop` $\to$ `"DecisionLog"."imageTop"`
  - `pool` $\to$ `"DecisionLog".pool`
  - `rerankScores` $\to$ `"DecisionLog"."rerankScores"`
  - `confidence`, `gap`, `agree` $\to$ `"DecisionLog"` columns
  - `action`, `wouldBeAction` $\to$ `"DecisionLog"` columns
  - `models` JSONB $\to$ `{"text": "...", "image": "...", "reranker": "...", "idempotency_key": "<key>"}`

### 4.2 `PUT /v1/services`
Bulk upserts service catalog definitions and synchronizes both text and vision profiles.

- **Request Body (`application/json`):**
  ```json
  [
    {
      "id": 1,
      "title": "AC Services",
      "description": "Comprehensive maintenance, repair, filter replacement, and temperature calibration for air conditioning and cooling systems.",
      "visual_line": "Air conditioning unit mounted on wall or ceiling, cooling vents, thermostat controls, condensate drip trays.",
      "keywords": "ac, air conditioner, hvac, cooling, chiller, ventilation",
      "is_active": true
    }
  ]
  ```
- **Behavior & Content Hashing:**
  1. For each service, computes content hashes:
     - Text hash: `SHA256(f"{title}\n{description}")`
     - Image text hash: `SHA256(f"{title}. {visual_line}")`
  2. If `WoService` row exists with matching title/description and `"embeddingModel"` equals active TEI model, text re-embedding is **skipped**.
  3. If changed or new, calls TEI `/embed` with un-prefixed passage for `title` and `description` and updates `"WoService"` (`"titleEmb"`, `"descEmb"`).
  4. If `visual_line` or `title` changed in `ServiceImageProfile`, encodes `textUsed` via local SigLIP model and updates `"ServiceImageProfile"` (`"textEmb"`, `truncated`).
  5. Refreshes in-memory image profile cache immediately.

### 4.3 `GET /v1/services`
Retrieves the catalog of active services.
- **Query Params:** `include_inactive=false` (boolean)
- **Response Model:** List of services with metadata (`id`, `title`, `description`, `visual_line`, `is_active`, `updated_at`).

### 4.4 `DELETE /v1/services/{id}`
Soft-deactivates a service.
- **Behavior:** Executes `UPDATE "WoService" SET "isActive" = FALSE, "updatedAt" = now() WHERE id = %s`.
- *FK Safety:* Prevents foreign key constraint violations on historical `"DecisionLog"` rows. If `hard=true` query parameter is explicitly provided, verifies no rows in `"DecisionLog"` or `"SearchLog"` reference the ID; otherwise returns `409 Conflict`.

### 4.5 `POST /v1/decisions/{id}/feedback`
Records operator ground truth on an existing decision (mirroring `photoToWorkOrder/ptwo/api.py`).
- **Request Body:** `{"service_id": 1}`
- **Behavior:**
  1. Queries `"DecisionLog"` for decision `id`. Returns `404 Not Found` if missing.
  2. Validates `service_id` exists in `"WoService"`.
  3. Evaluates if `service_id` was present in the logged `pool` JSON (`poolContainsFinal`).
  4. Updates `"DecisionLog"`:
     ```sql
     UPDATE "DecisionLog"
     SET "finalServiceId" = %s,
         "finalAt" = now(),
         "poolContainsFinal" = %s
     WHERE id = %s
     RETURNING id, "finalServiceId", "finalAt", "poolContainsFinal";
     ```

### 4.6 Health & Observability Endpoints
- `GET /health/live`: HTTP 200 `{"status": "alive"}` (process liveness probe; unauthenticated).
- `GET /health/ready`: HTTP 200 if all components ready; HTTP 503 if any required component fails:
  ```json
  {
    "status": "ready",
    "database": {"connected": true, "pgvector_version": "0.8.7", "schema_verified": true},
    "siglip_model": {"loaded": true, "model_id": "google/siglip2-base-patch16-224", "dim": 768},
    "tei_embed": {"reachable": true, "url": "http://tei-embed:80", "model_id": "BAAI/bge-base-en-v1.5", "dim": 768},
    "tei_rerank": {"reachable": true, "url": "http://tei-rerank:80", "model_id": "BAAI/bge-reranker-base"}
  }
  ```
- `GET /metrics`: Standard Prometheus metrics exposition (counters, histograms, gauges).
- Internal debug endpoints (`/embed-text`, `/embed-image`, `/rank`): Disabled by default, returning 404 unless `ENABLE_DEBUG_ENDPOINTS=true`.

---

## 5. Environment Variable Configuration Table

All configurations are strictly managed via `pydantic-settings` with environment variable loading:

| Category | Variable Name | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pipeline Weights** | `PIPELINE_TEXT_DESC_WEIGHT` | `float` | `0.7` | No | Weight assigned to description embedding vs title (0.0 to 1.0) |
| | `PIPELINE_TEXT_TOP_N` | `int` | `10` | No | Number of candidates retrieved from pgvector text search |
| | `PIPELINE_IMAGE_TOP_N` | `int` | `5` | No | Number of candidates retrieved from SigLIP image search |
| | `PIPELINE_POOL_MAX` | `int` | `10` | No | Maximum candidate capacity for the merged pool |
| | `PIPELINE_RERANKER_TEMP` | `float` | `1.0` | No | Softmax temperature for reranker logits |
| **Thresholds & Safety** | `DECISION_CONFIDENCE_THRESHOLD` | `float` | `0.40` | No | Minimum confidence ($a$) for auto-create qualification |
| | `DECISION_GAP_THRESHOLD` | `float` | `0.10` | No | Minimum margin ($g$) between #1 and #2 candidate |
| | `DECISION_NO_ACTION_FLOOR` | `float` | `0.25` | No | Confidence threshold below which action becomes `no_action` |
| | `DECISION_REQUIRE_AGREE` | `bool` | `false` | No | Require image and text #1 agreement for auto-create |
| | `AUTO_CREATE_ENABLED` | `bool` | `false` | No | Master auto-create toggle. If false, returns draft with `would_auto: true` |
| | `ALWAYS_DRAFT_SERVICES` | `list[str]`| `["Emergency Response & Area Securing", "Emergency Reactive Cleaning (Spills, Vomit, Incidents)"]` | No | JSON list of service titles requiring mandatory human review |
| | `MIN_DESCRIPTION_WORDS` | `int` | `3` | No | Minimum word count in description to allow auto-create |
| **Database Connection**| `DATABASE_URL` | `str` | None | **Yes** | PostgreSQL connection URI (`postgresql://user:pass@host:5432/dbname`) |
| | `DB_SCHEMA` | `str` | `public` | No | PostgreSQL search_path schema |
| | `DB_POOL_MIN_SIZE` | `int` | `2` | No | Minimum persistent connections in connection pool |
| | `DB_POOL_MAX_SIZE` | `int` | `10` | No | Maximum connections in connection pool |
| | `DB_TIMEOUT_SECONDS` | `float` | `10.0` | No | Database query and connection timeout |
| **TEI Dependencies** | `TEI_EMBED_URL` | `str` | `http://tei-embed:80` | No | HTTP URI of Text Embeddings Inference embedder container |
| | `TEI_RERANK_URL` | `str` | `http://tei-rerank:80` | No | HTTP URI of Text Embeddings Inference reranker container |
| | `TEI_API_KEY` | `str` | None | No | Bearer token for TEI containers (if configured with `--api-key`) |
| | `TEI_TIMEOUT_SECONDS` | `float` | `10.0` | No | HTTP request timeout for TEI queries |
| | `TEI_MAX_RETRIES` | `int` | `2` | No | Maximum connection retry attempts with exponential backoff |
| **SigLIP Model** | `SIGLIP_MODEL_ID` | `str` | `google/siglip2-base-patch16-224` | No | Hugging Face model identifier for vision-language encoder |
| | `SIGLIP_TORCH_THREADS` | `int` | `2` | No | Number of CPU threads dedicated to PyTorch inference |
| | `SIGLIP_MAX_CONCURRENCY`| `int` | `2` | No | Semaphore bound for simultaneous PyTorch inference passes |
| | `SIGLIP_LRU_CACHE_SIZE` | `int` | `256` | No | In-memory LRU cache capacity for candidate text vectors |
| **Security & Auth** | `API_KEYS` | `str` | None | No | Comma-separated list of valid client Bearer API keys |
| | `METRICS_API_KEY` | `str` | None | No | Optional dedicated API key for `/metrics` endpoint |
| **Operational & Limits**| `MAX_UPLOAD_BYTES` | `int` | `10485760` (10MB) | No | Maximum allowed payload size for uploaded images |
| | `STORE_DESCRIPTION` | `bool` | `true` | No | Whether to persist raw description text in `DecisionLog` |
| | `ENABLE_DEBUG_ENDPOINTS`| `bool` | `false` | No | Enables legacy `/embed-text`, `/embed-image`, `/rank` endpoints |
| | `LOG_LEVEL` | `str` | `INFO` | No | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| | `LOG_FORMAT` | `str` | `json` | No | Log output format (`json` or `console`) |

---

## 6. Embedding & Model Validation at Startup (Fail Fast)

To prevent catastrophic metric degradation from silent model mixing, the application executes strict startup validation before accepting traffic:

```mermaid
flowchart TD
    Start([Application Lifespan Startup]) --> Step1[1. Verify PostgreSQL Connectivity & Extensions]
    Step1 -->|Fail| Abort1[Fatal Error: DB unreachable or pgvector missing]
    Step1 -->|Pass| Step2[2. Verify Database Schema Tables & Columns]
    Step2 -->|Fail| Abort2[Fatal Error: Missing required table/column or type drift]
    Step2 -->|Pass| Step3[3. Probe TEI Embedder /info and /embed]
    Step3 -->|Fail| Abort3[Fatal Error: TEI Embedder unreachable or error]
    Step3 -->|Pass| Step4[4. Compare Real TEI Vector Dim vs WoService Column]
    Step4 -->|Mismatch != 768| Abort4[Fatal Error: TEI dim does not match vector 768]
    Step4 -->|Match 768| Step5[5. Probe TEI Reranker /info and /rerank]
    Step5 -->|Fail| Abort5[Fatal Error: TEI Reranker unreachable or error]
    Step5 -->|Pass| Step6[6. Load SigLIP Model & Verify Hidden Dim == 768]
    Step6 -->|Match 768| Step7[7. Load & Cache ServiceImageProfile in Memory]
    Step7 --> Ready([Mark Service Ready: HTTP 200 /health/ready])
```

### 6.1 Validation Checks
1. **Database Schema Verification:**
   - Validates existence of `"WoService"`, `"ServiceImageProfile"`, `"DecisionLog"`.
   - Checks presence of required columns, types (`vector`, `jsonb`, `timestamptz`), and constraints.
2. **TEI Embedding Model & Dimension Verification:**
   - Sends test input to `POST {TEI_EMBED_URL}/embed` (`"Represent this sentence for searching relevant passages: probe"`).
   - Measures output vector dimension $D_{\text{real}}$.
   - Queries `atttypmod` on `"WoService"."titleEmb"` and `"descEmb"`. Asserts $D_{\text{real}} == 768$. On mismatch, fails immediately with:
     `ConfigurationError: TEI Embedder dimension (X) does not match WoService embedding column dimension (768).`
   - Inspects `GET {TEI_EMBED_URL}/info` to log model ID (`BAAI/bge-base-en-v1.5`).
3. **SigLIP Image Model Verification:**
   - Validates `embedding_dim == 768`.
   - Validates `vector_dims("textEmb")` in `"ServiceImageProfile"`.
4. **Reindexing CLI Tooling:**
   - A dedicated CLI command `python -m app.cli reindex` is provided. It:
     - Queries all active services from `"WoService"`.
     - Re-embeds title and description via TEI `/embed` and updates `"WoService"`.
     - Re-embeds `textUsed` via SigLIP text encoder and updates `"ServiceImageProfile"`.
     - Runs idempotently using transaction blocks.

---

## 7. Least-Privilege Database Role & Permissions

In production, the microservice must not connect with `postgres` superuser privileges. The following SQL creates a dedicated, least-privilege user `siglip_service` granting only required runtime operations:

```sql
-- 1. Create dedicated application user
CREATE USER siglip_service WITH PASSWORD '<SECURE_RANDOM_PASSWORD>';

-- 2. Grant connection and usage on schema
GRANT CONNECT ON DATABASE wo_search TO siglip_service;
GRANT USAGE ON SCHEMA public TO siglip_service;

-- 3. Grant table permissions (NO DDL, NO DROP, NO TRUNCATE)
-- WoService: Read catalog and update embeddings/isActive
GRANT SELECT, INSERT, UPDATE ON TABLE public."WoService" TO siglip_service;

-- ServiceImageProfile: Read profiles and update text embeddings
GRANT SELECT, INSERT, UPDATE ON TABLE public."ServiceImageProfile" TO siglip_service;

-- DecisionLog: Insert inference telemetry and update human feedback
GRANT SELECT, INSERT, UPDATE ON TABLE public."DecisionLog" TO siglip_service;

-- SearchLog: Read-only access if needed for audit
GRANT SELECT ON TABLE public."SearchLog" TO siglip_service;

-- 4. Grant sequence usage for autoincrement IDs
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO siglip_service;

-- Revoke all public DDL permissions
REVOKE CREATE ON SCHEMA public FROM siglip_service;
```

---

## 8. Proposed Optional Migration Script (Reviewed, Not Auto-Applied)

To provide index-accelerated idempotency without relying on JSONB table scans, the following migration script `siglip/sql/002_decision_log_idempotency.sql` is prepared:

```sql
-- Migration: 002_decision_log_idempotency.sql
-- Purpose: Add dedicated idempotencyKey column and unique index to DecisionLog.
-- Safe, non-breaking, idempotent migration.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'DecisionLog' 
          AND column_name = 'idempotencyKey'
    ) THEN
        ALTER TABLE public."DecisionLog" ADD COLUMN "idempotencyKey" TEXT NULL;
    END IF;
END $$;

-- Partial unique index ensures idempotency keys are unique while allowing multiple NULL entries
CREATE UNIQUE INDEX IF NOT EXISTS "idx_decisionlog_idempotency" 
ON public."DecisionLog" ("idempotencyKey") 
WHERE "idempotencyKey" IS NOT NULL;
```
*Note:* In Phase 0 and runtime Phase 1, the service operates seamlessly on the existing schema without applying this migration by utilizing the `models->>'idempotency_key'` JSONB index path.

---

## 9. Tenancy Analysis, Open Questions & Risk Register

### 9.1 Multi-Tenancy Architecture Recommendation
- **User Default:** One deployment per company / organization (no `tenant` column in schema).
- **Architectural Assessment:**
  1. *Data Isolation & Compliance:* Maintenance and work order requests contain internal facility maps, photos, and access details. Dedicated per-company databases eliminate multi-tenant cross-talk and accidental query leakage (`WHERE tenant_id = ...` omission bugs).
  2. *Custom Service Catalogs & Visual Lines:* Different companies operate vastly different facilities (e.g. school campus vs commercial high-rise vs industrial manufacturing). Vector spaces, visual lines, and RRF weights diverge per enterprise catalog.
  3. *Coolify Deployment Topology:* Coolify excels at managing container stacks per client/environment. A dedicated container + DB stack per company provides clear resource isolation and billing attribution.
- **Recommendation:** **Agree completely with the default.** We maintain the existing zero-tenant schema and deploy one isolated container stack per company.

### 9.2 Open Questions & Risk Register

| Risk / Question | Severity | Mitigation Strategy |
| :--- | :---: | :--- |
| **1. Memory Footprint on CPU** | Medium | SigLIP 2 occupies ~2.1 GB RSS on CPU. In `docker-compose.dev.yml`, memory limits are pinned to 4 GB for `siglip` and 2 GB each for `tei-embed` and `tei-rerank`. Semaphores restrict concurrent PyTorch passes to 2. |
| **2. TEI Reranker Model Divergence** | Low | Live TEI reranker container runs `BAAI/bge-reranker-base`, while earlier scripts mentioned `BAAI/bge-reranker-v2-m3`. Both return pairwise cross-encoder logits compatible with temperature softmax. The client adapts to either model. |
| **3. Foreign Key Deletion Failure** | Low | Hard deletion of a service referenced in `"DecisionLog"` will trigger `ON DELETE NO ACTION`. Deletion endpoint defaults to soft-delete (`"isActive" = false`). |
| **4. External Docker Network Connection** | Low | When deployed on Coolify, the `siglip` container joins the user's existing Docker bridge network (e.g. `coolify` or `tei_default`) to reach PostgreSQL and TEI without exposing ports publicly. |

---

## 10. Before Coolify Deployment Checklist

The following items are deferred during local development and must be addressed prior to deploying the service into Coolify:

### 10.1 Fresh Database Bootstrap SQL
If deploying against a brand-new PostgreSQL instance (rather than an existing database), the database and vector extension must be initialized before starting the application:

```sql
-- 1. Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. WoService table (service catalog with text embeddings)
CREATE TABLE IF NOT EXISTS public."WoService" (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    keywords TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "embeddingModel" TEXT NOT NULL,
    "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT now(),
    "titleEmb" vector(768) NOT NULL,
    "descEmb" vector(768) NOT NULL
);

-- 3. ServiceImageProfile table (visual lines and SigLIP text embeddings)
CREATE TABLE IF NOT EXISTS public."ServiceImageProfile" (
    "serviceId" INTEGER NOT NULL REFERENCES public."WoService"(id) ON DELETE CASCADE,
    "imageModel" TEXT NOT NULL,
    "visualLine" TEXT,
    "textUsed" TEXT NOT NULL,
    "truncated" BOOLEAN NOT NULL DEFAULT FALSE,
    "textEmb" vector(768) NOT NULL,
    "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY ("serviceId", "imageModel")
);

-- 4. DecisionLog table (triage audits and operator calibration logs)
CREATE TABLE IF NOT EXISTS public."DecisionLog" (
    id SERIAL PRIMARY KEY,
    "createdAt" TIMESTAMPTZ NOT NULL DEFAULT now(),
    description TEXT,
    "photoRef" TEXT,
    "photoHash" VARCHAR(64),
    location TEXT,
    "textTop" JSONB,
    "imageTop" JSONB,
    pool JSONB,
    "rerankScores" JSONB,
    "topServiceId" INTEGER REFERENCES public."WoService"(id) ON DELETE NO ACTION,
    confidence DOUBLE PRECISION,
    gap DOUBLE PRECISION,
    agree BOOLEAN,
    action VARCHAR(32) NOT NULL,
    "wouldBeAction" VARCHAR(32) NOT NULL,
    reason TEXT,
    models JSONB NOT NULL,
    "configVersion" VARCHAR(64) NOT NULL,
    timings JSONB,
    "finalServiceId" INTEGER REFERENCES public."WoService"(id) ON DELETE NO ACTION,
    "finalAt" TIMESTAMPTZ,
    "poolContainsFinal" BOOLEAN
);

-- 5. Indexes
CREATE INDEX IF NOT EXISTS "idx_woservice_isactive" ON public."WoService"("isActive");
CREATE INDEX IF NOT EXISTS "idx_decisionlog_createdat" ON public."DecisionLog"("createdAt" DESC);
CREATE INDEX IF NOT EXISTS "idx_decisionlog_photohash" ON public."DecisionLog"("photoHash");
CREATE INDEX IF NOT EXISTS "idx_decisionlog_models" ON public."DecisionLog" USING gin (models);
```

### 10.2 Least-Privilege Role & Sequence Grants
The operational role must have sequence usage to perform `SERIAL` / `BIGSERIAL` inserts into `DecisionLog` and `WoService`:

```sql
CREATE USER siglip_service WITH ENCRYPTED PASSWORD 'replace_with_strong_password';

-- Schema usage
GRANT USAGE ON SCHEMA public TO siglip_service;

-- Table permissions (Least privilege: SELECT, INSERT, UPDATE)
GRANT SELECT, INSERT, UPDATE ON TABLE public."WoService" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."ServiceImageProfile" TO siglip_service;
GRANT SELECT, INSERT, UPDATE ON TABLE public."DecisionLog" TO siglip_service;

-- Sequence permissions (CRITICAL: required for SERIAL RETURNING id)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO siglip_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO siglip_service;

-- Revoke all DDL / DROP privileges
REVOKE CREATE ON SCHEMA public FROM siglip_service;
```

### 10.3 DELETE Privilege & Hard-Delete Policy
In `DELETE /v1/services/{id}`, the default behavior is **soft-delete** (`isActive = false`), which requires only `UPDATE` privilege and safely preserves historical references in `"DecisionLog"`.
If hard-delete (`DELETE FROM "WoService"`) is ever needed via a flag, the `siglip_service` user would require `DELETE` privileges, which will fail with foreign key violation errors on any service referenced by historical `"DecisionLog"` entries due to `ON DELETE NO ACTION`.
*Production rule:* Keep all deletions soft (`isActive = false`). Do not grant `DELETE` privilege to the application user.

### 10.4 Mandatory Production API Keys
In local development, `API_KEYS` can be left empty for convenient unauthenticated testing. In production Coolify:
- `API_KEYS` **MUST** be non-empty (comma-separated tokens, e.g. `primary-key-1,rotated-key-2`).
- A startup check in production should fail fast if `API_KEYS` is empty while running in production mode.
- Probes (`/health/live`, `/health/ready`) remain unauthenticated for Coolify health checking.
- Metrics (`/metrics`) should either be protected by `METRICS_API_KEY` or restricted to the internal network.

### 10.5 Coolify Docker Network Attachment
The `siglip` service container must attach to the external Docker network that connects to the PostgreSQL container and TEI containers (e.g. `coolify` or `tei_default`). Set `network_mode` or configure Coolify's attached network in the environment interface so hostnames like `tei-embed:80`, `tei-rerank:80`, and the PostgreSQL host resolve internally without routing through public DNS or internet.


