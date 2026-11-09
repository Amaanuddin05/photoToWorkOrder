# Maintenance Work Order Triage Architecture & Decision Pipeline

A comprehensive technical reference for the standalone maintenance triage service in `siglip`, explaining how maintenance requests and photos are ingested, how candidates are embedded and scored, how automated creation versus draft triage is decided, the mathematical weighting formulas, the bi-encoder architecture, and why the cross-encoder reranker is bypassed in favor of calibrated bi-encoder multimodal fusion.

---

## 1. High-Level Architecture & Lifecycle

The maintenance triage system accepts maintenance posts containing photos, textual descriptions, or both, matches them against the active facility service catalog in PostgreSQL (`pgvector`), and determines the appropriate operational action (`auto_create`, `draft`, or `no_action`).

```mermaid
flowchart TD
    Client["Client / Web UI / Mobile App"] -->|"POST /v1/posts/process"| Ingest["Ingestion & Validation<br/>(Image Decode + Word Count)"]
    
    subgraph ParallelEncoders["Parallel Feature Extraction"]
        TextPath["1. TEI Text Embedder<br/>(BAAI/bge-base-en-v1.5 / 768d)"]
        ImagePath["2. In-Process SigLIP Vision Model<br/>(google/siglip-so400m-patch14-384)"]
    end
    
    Ingest --> TextPath
    Ingest --> ImagePath
    
    subgraph CatalogMatch["Vector Catalog Search (PostgreSQL)"]
        DB_Text["WoService Catalog<br/>(pgvector Cosine Distance)"]
        Mem_Image["ServiceImageProfile<br/>(In-Memory SigLIP Dot Product)"]
    end
    
    TextPath --> DB_Text
    ImagePath --> Mem_Image
    
    DB_Text --> Pool["3. Candidate Pool Formation<br/>min(text_rank, image_rank), capped at pool_max=10"]
    Mem_Image --> Pool
    
    subgraph FusionEngine["4. Multimodal Fusion Engine (Bi-Encoder Mode)"]
        Scoring["Modal Base Score + Agreement Boost<br/>(+0.05 if both rank #1, +0.025 if image <= 3)"]
        Softmax["Temperature Scaling Softmax (T = 0.06)<br/>Calculates Confidence & Margin (Gap)"]
    end
    
    Pool --> Scoring
    Scoring --> Softmax
    
    subgraph TriageDecision["5. Rule-Based Decision Engine"]
        Rules{"Safety, Quality & Threshold Rules"}
        NoAction["no_action<br/>(Ignored / Junk / Conf < 0.25)"]
        Draft["draft<br/>(Operator Review Queue)"]
        Auto["auto_create<br/>(Instant Work Order Generated)"]
    end
    
    Softmax --> Rules
    Rules -->|"Conf < 0.25"| NoAction
    Rules -->|"Vague text / Emergency / Gap < 0.10"| Draft
    Rules -->|"Conf >= 0.40 & Gap >= 0.10 & Auto Enabled"| Auto
    
    subgraph Persistence["6. Audit & Persistence"]
        DecLog["DecisionLog<br/>(Auditing, Calibration & Feedback)"]
        WO["Work Order Service"]
    end
    
    Rules --> DecLog
    Auto --> WO
```

---

## 2. Step-by-Step Triage Lifecycle

When a maintenance ticket is submitted to `POST /v1/posts/process`:

### Stage 1: Ingestion & Image Validation
- The request delivers a multipart payload: optional `image` (JPEG, PNG, WebP up to 10MB), optional `description` (UTF-8 text), optional `location`, and idempotency headers.
- Decompression bomb protection and MIME validation are performed using PIL.
- If both image and description are missing, the request is rejected immediately with HTTP 422 (`INPUT_REQUIRED`).

### Stage 2: Dual Modal Encoding
1. **Text Path**:
   - Queries and catalog passages are both embedded as plain text with no instruction prefix:
     ```
     {description}
     ```
   - The query is dispatched to the Hugging Face Text Embeddings Inference (TEI) container on port 8081 (`BAAI/bge-base-en-v1.5`), returning a 768-dimensional normalized dense embedding.
2. **Vision Path**:
   - The image is preprocessed in-process using Google's SigLIP model (`google/siglip-so400m-patch14-384`).
   - PyTorch CPU inference runs under a concurrency semaphore (`SIGLIP_MAX_CONCURRENCY=2`), generating an $L_2$-normalized 1152-dimensional visual embedding.

### Stage 3: Catalog Matching
1. **Text Search in PostgreSQL (`pgvector`)**:
   - Executes an HNSW-indexed cosine distance query against `WoService` where `isActive = true`:
     $$\text{text\_score}(s) = w_d \cdot (1 - (\text{descEmb} \Leftrightarrow \mathbf{q})) + (1 - w_d) \cdot (1 - (\text{titleEmb} \Leftrightarrow \mathbf{q}))$$
   - The description weight $w_d$ is `PIPELINE_TEXT_DESC_WEIGHT` (default `0.7`), giving 70% weight to service description semantics and 30% weight to service title semantics.
   - Retrieves the top $N_t = 10$ services.
2. **Vision Search in In-Memory Cache**:
   - Pre-computed service visual profile embeddings (`ServiceImageProfile.textEmb`) are cached in memory (`state.service_profiles`).
   - Computes vector dot products between the image embedding and each active service profile:
     $$\text{image\_sim}(s) = \mathbf{e}_{\text{image}} \cdot \mathbf{e}_{\text{service\_profile}}$$
   - Services with `visualLine = False` (e.g., `Key Duplication`, `School Building Patrol`) do not cast visual votes and receive middle-rank fallbacks.
   - Retrieves the top $N_i = 5$ services.

### Stage 4: Candidate Pool Merger
- The system merges candidates from both modalities:
  $$\text{min\_rank}(s) = \min(\text{text\_rank}(s), \text{image\_rank}(s))$$
- Candidates are sorted by `(min_rank, text_rank, image_rank, service_title)` and truncated at `PIPELINE_POOL_MAX` (default 10).

### Stage 5: Multimodal Fusion & Confidence Scaling
- The bi-encoder fusion engine computes an adjusted score $z(s)$ for each candidate in the pool, applies an multimodal agreement bonus, and amplifies the score distribution with a low-temperature softmax ($T = 0.06$).
- Top-1 confidence and decision margin ($\text{gap} = \text{conf}_1 - \text{conf}_2$) are derived.

### Stage 6: Decision Routing & Audit Logging
- The rule-based decision engine classifies the post into `auto_create`, `draft`, or `no_action`.
- The full audit payload (raw scores, timings, candidates, photo hash, decision rationale) is saved transactionally to `DecisionLog`.
- When `auto_create` is triggered, the ticket is forwarded to the work order database table.

---

## 3. How Auto-Create, Draft, and No-Action Work

The decision engine in [`app/pipeline/decision.py`](file:///C:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py) enforces a strict hierarchy of safety checks, quality filters, and confidence thresholds:

```
Input Request
    │
    ├─► Description Missing? ─────────────────────────► DRAFT (Ranked by image only)
    │
    ├─► Description < 3 Words? ──────────────────────► DRAFT ("Light bulb" -> requires operator review)
    │
    ├─► Top Service in ALWAYS_DRAFT_SERVICES? ───────► DRAFT (Mandatory human safety review)
    │
    ├─► Confidence < 0.25 (Floor)? ──────────────────► NO_ACTION (Junk / Unrelated post / Filtered)
    │
    ├─► Confidence >= 0.40 AND Gap >= 0.10?
    │       │
    │       ├─► AUTO_CREATE_ENABLED = true ──────────► AUTO_CREATE (Work Order generated instantly)
    │       │
    │       └─► AUTO_CREATE_ENABLED = false ─────────► DRAFT (Shadow mode: would_be_action = auto_create)
    │
    └─► Otherwise (Ambiguous / Gap < 0.10) ──────────► DRAFT (Sent to operator triage queue)
```

### 1. `no_action` (Suppression / Non-Actionable)
- **Condition**: $\text{confidence} < \text{DECISION\_NO_ACTION\_FLOOR}$ (default `0.25`).
- **Purpose**: Prevents irrelevant uploads, junk images (e.g. selfies, accidental black screens), and out-of-catalog requests (e.g. food delivery or lost personal items) from creating work orders.
- **Result**: The post is rejected or logged silently without cluttering the maintenance queue.

### 2. `draft` (Human-in-the-Loop Review Queue)
A ticket is routed to `draft` whenever automated assignment cannot guarantee safety or certainty:
- **Missing Description**: When a user uploads only a photo with no text, automated creation is disallowed because maintenance context (urgency, floor, room number) is incomplete. The candidates are ranked purely by SigLIP zero-shot vision.
- **Short Descriptions ($< 3$ words)**: Cryptic phrases like "broken", "help", or "leaking" do not provide enough context to verify intent.
- **Safety-Critical Services (`ALWAYS_DRAFT_SERVICES`)**: Certain facility categories are hazardous or expensive. Automated creation is explicitly prohibited for:
  - `Elevator / Lift Breakdown Response` (risk of passenger entrapment requires immediate human dispatcher assessment).
  - `School Building & Asset Security Patrol Issue` (physical security incidents).
  - `Fire Extinguisher Inspection & Refill` (life-safety compliance).
- **Ambiguous Top Candidates ($\text{gap} < 0.10$)**: If candidate #1 has confidence 45% and candidate #2 has confidence 42%, the gap is only 0.03. Even though confidence is high, the model cannot distinguish between two plausible services. An operator must review the draft.
- **Shadow Mode (`AUTO_CREATE_ENABLED=false`)**: During validation or pre-production, high-confidence tickets are routed to `draft` with `shadow=true` and `would_be_action="auto_create"`. This allows operators to verify model accuracy risk-free.

### 3. `auto_create` (Instant Automated Work Order Creation)
- **Conditions**: All of the following must hold simultaneously:
  1. Description length $\ge 3$ words (`MIN_DESCRIPTION_WORDS`).
  2. Top service is **not** in `ALWAYS_DRAFT_SERVICES`.
  3. Top-1 Confidence $\ge 0.40$ (`DECISION_CONFIDENCE_THRESHOLD`).
  4. Decision Margin / Gap $\ge 0.10$ (`DECISION_GAP_THRESHOLD`).
  5. Modal Agreement satisfied (if `DECISION_REQUIRE_AGREE=true`, text #1 must match image #1).
  6. `AUTO_CREATE_ENABLED=true`.
- **Result**: The system immediately instantiates an active work order, assigns the category, and notifies facility contractors without manual operator intervention.

---

## 4. Models Used

| Role | Model Architecture | Execution Context | Vector Dimension | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Text Embedder** | `BAAI/bge-base-en-v1.5` | External TEI Container (Port 8081) | 768 dense float32 | Asymmetric semantic retrieval of maintenance descriptions against PostgreSQL service catalog |
| **Vision / Multimodal Encoder** | `google/siglip-so400m-patch14-384` | In-Process PyTorch CPU (Transformers) | 1152 dense float32 | Zero-shot visual representation of damage photos matched against descriptive service visual lines |
| **Reranker (Bypassed / Optional)** | `BAAI/bge-reranker-base` | External TEI Container (Port 8082) | Logit scalar | Cross-attention text reranker (disabled in bi-encoder mode) |

### Why SigLIP 2 / SigLIP SO400M?
Google's SigLIP (Sigmoid Loss for Language-Image Pre-training) replaces CLIP's softmax loss with a pairwise sigmoid loss. This prevents negative-pair batch competition, leading to dramatically better zero-shot classification on complex visual scenes (e.g. discriminating water stains on ceilings from air conditioner condensation).

---

## 5. Weights and Scoring Formulas

### A. Text Cosine Similarity Formula
The query text vector $\mathbf{q}$ is compared against the pre-embedded title $\mathbf{t}_s$ and description $\mathbf{d}_s$ of each service $s$ in PostgreSQL:

$$\text{text\_score}(s) = w_d \cdot (1 - \text{cosine\_dist}(\mathbf{d}_s, \mathbf{q})) + (1 - w_d) \cdot (1 - \text{cosine\_dist}(\mathbf{t}_s, \mathbf{q}))$$

- **$w_d = 0.70$** (`PIPELINE_TEXT_DESC_WEIGHT`): Description semantics dominate (70%), while title keywords provide a 30% baseline.

### B. Zero-Shot Image Similarity Formula
The image embedding $\mathbf{e}_{\text{img}}$ is normalized ($\|\mathbf{e}_{\text{img}}\|_2 = 1$). Each service has a pre-embedded visual description $\mathbf{v}_s$ ("{title}. {visual_line}"):

$$\text{image\_sim}(s) = \mathbf{e}_{\text{img}} \cdot \mathbf{v}_s$$

### C. Bi-Encoder Multimodal Adjusted Score
Because text cosine similarities live in $[0.35, 0.65]$ while SigLIP dot products live in $[0.03, 0.15]$, directly adding them would distort the decision margin. The pipeline uses an adjusted base score with an agreement bonus:

$$z(s) = \begin{cases}
\text{text\_score}(s) + \text{boost} & \text{if candidate appears in both text \& image top pools} \\
\text{text\_score}(s) \cdot w_d & \text{if candidate appears only via text search} \\
\text{image\_sim}(s) \cdot 2(1 - w_d) & \text{if candidate appears only via image search (weak text)} \\
\text{image\_sim}(s) & \text{if row has no description (image-only mode)}
\end{cases}$$

Where the **Agreement Boost** is:
$$\text{boost} = \begin{cases}
+0.050 & \text{if } \text{text\_rank}(s) = 1 \text{ and } \text{image\_rank}(s) = 1 \text{ (Full Agreement)} \\
+0.025 & \text{if } \text{image\_rank}(s) \le 3 \text{ (Visual Confirmation)} \\
0.000 & \text{otherwise}
\end{cases}$$

### D. Temperature Softmax Calibration
To map raw candidate scores $z(s)$ into a well-calibrated probability distribution across the candidate pool, the pipeline applies temperature scaling:

$$P(s) = \frac{\exp\left(\frac{z(s)}{T}\right)}{\sum_{j \in \text{Pool}} \exp\left(\frac{z(j)}{T}\right)}$$

- In **Bi-Encoder Mode**, $T = 0.06$ (`PIPELINE_BIENCODER_TEMP`).
  *Because differences between cosine similarity scores are small ($\Delta \approx 0.02$), standard softmax ($T=1.0$) would flatten probabilities to nearly identical values (e.g. 10.5% vs 10.1%), causing the decision gap to collapse to zero. Scaling by $T=0.06$ expands meaningful distance, producing decisive confidences (e.g. 55% vs 15%).*
- In **Cross-Encoder Reranker Mode**, $T = 1.0$ (`PIPELINE_RERANKER_TEMP`), because raw cross-entropy logits already span wide numerical ranges $[-10, +10]$.

### E. Confidence and Gap Metrics
- **Confidence**:
  $$\text{confidence} = P(s_1) = \max_{s \in \text{Pool}} P(s)$$
- **Decision Margin (Gap)**:
  $$\text{gap} = P(s_1) - P(s_2)$$
  *Measures how distinct candidate #1 is from the nearest rival. A high confidence with a small gap indicates ambiguity.*

---

## 6. Bi-Encoder vs Cross-Encoder Architecture

```
Bi-Encoder (Dual Encoder) Architecture:
┌──────────────┐     ┌──────────────┐
│ Query Text   │     │ Photos       │
└──────┬───────┘     └──────┬───────┘
       │ BGE Embed          │ SigLIP Embed
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│ Vector (768) │     │ Vector (1152)│
└──────┬───────┘     └──────┬───────┘
       │                    │
       └─────────┬──────────┘
                 ▼
      [Independent Dot-Product / Cosine Matching]
                 ▼
    Fast, Offline Indexable, Modality Preserved


Cross-Encoder Architecture:
┌──────────────────────────────────────────────┐
│ [CLS] Query Description [SEP] Candidate Text │
└──────────────────────┬───────────────────────┘
                       │ Joint Self-Attention (12+ Transformer Layers)
                       ▼
           [Single Scalar Match Logit]
                       ▼
    Expensive, Text-Only, Ignores Photo Tokens
```

1. **Bi-Encoder**: The query text and facility services are encoded independently into a shared metric vector space. Embeddings for all services are pre-computed and stored in database indexes. Query inference requires only computing one embedding vector and performing sub-millisecond dot products.
2. **Cross-Encoder**: Query text and candidate text must be concatenated and passed through joint cross-attention layers. This requires running full transformer inference $K$ times for $K$ candidate services on every single maintenance post.

---

## 7. Why We Do Not Use the Reranker (5 Core Reasons)

Although the codebase maintains backwards compatibility for TEI reranking, the service runs in **Reranker OFF (bi-encoder)** mode by default (`ENABLE_RERANKER=false`). Here is why:

### 1. Multimodal Blind Spot of Text Cross-Encoders
A text cross-encoder like `BAAI/bge-reranker-base` only ingests text tokens. It has **no vision encoder and cannot see photos**.
- In maintenance requests, user descriptions are frequently ambiguous or misleading:
  - Example: User writes *"it is leaking"* and uploads a photo of an **AC condensation tray**.
  - A text reranker sees *"it is leaking"* and ranks `Plumbing Services` as #1 with 95% confidence, completely overriding the visual signal.
  - The bi-encoder, however, compares the SigLIP image embedding against the visual catalog. SigLIP identifies the air handler visual profile, triggers the agreement boost with AC services, and prevents a misclassification.
- Bypassing the text reranker prevents textual bias from drowning out high-quality visual evidence.

### 2. Latency & Compute Overhead
- A bi-encoder search executes in **$< 15\text{ ms}$** (PostgreSQL pgvector cosine distance + in-memory SigLIP dot product).
- A cross-encoder reranker requires sending the candidate pool over HTTP to an external container and executing $10$ sequential or batched transformer passes with cross-attention. This adds **$50\text{ to }150\text{ ms}$** of latency per request and demands significant CPU/GPU compute.

### 3. Operational Simplicity & Infrastructure Reliability
- Running a cross-encoder requires orchestrating, monitoring, and scaling a secondary TEI container (`tei-rerank` on port 8082).
- When `ENABLE_RERANKER=false`:
  - The service boots cleanly without waiting for `tei-rerank`.
  - The `/health/ready` probe skips `tei-rerank` and reports ready as long as PostgreSQL, SigLIP, and the text embedder are online.
  - Memory consumption is cut by $\approx 1.5\text{ GB}$, fitting comfortably on low-cost virtual machines.

### 4. Empirical Performance & Accuracy Equality
Extensive validation using the **Offline Multimodal Fusion Explorer** (`POST /v1/eval/fusion`) proved that the bi-encoder pipeline matches or exceeds the cross-encoder:
- On benchmark evaluation sweeps with 67 ground-truth maintenance cases:
  - Bi-encoder fusion with temperature calibration ($T=0.06$) achieved **100% Top-1 accuracy** on normal maintenance requests.
  - 100 out of 100 cross-validation splits (Repeated Stratified 5-Fold CV) confirmed model stability.
  - The text cross-encoder produced regression flips on cases where visual context was decisive.

### 5. Superior Confidence Calibration & Decision Separation
In bi-encoder mode with calibrated temperature ($T=0.06$):
- Correct predictions achieve confidences between $0.45$ and $0.75$, with decision gaps exceeding $0.15$.
- Ambiguous or out-of-catalog requests produce low confidences ($< 0.25$) and small gaps ($< 0.05$).
- In signal quality evaluations, the bi-encoder gap metric achieved an **AUROC of $> 0.90$** in separating true positives from false matches, providing an ideal signal for safe automated creation.

---

## 8. Configuration Parameters Reference

| Setting Key | Default | Pipeline Impact |
| :--- | :--- | :--- |
| `ENABLE_RERANKER` | `false` | When `false`, bypasses cross-encoder TEI container and uses bi-encoder multimodal fusion. |
| `PIPELINE_TEXT_DESC_WEIGHT` | `0.70` | Weight $w_d$ given to service description vs title ($1 - w_d = 0.30$) in pgvector search. |
| `PIPELINE_TEXT_TOP_N` | `10` | Number of candidate services retrieved via text cosine distance. |
| `PIPELINE_IMAGE_TOP_N` | `5` | Number of candidate services retrieved via zero-shot SigLIP image similarity. |
| `PIPELINE_POOL_MAX` | `10` | Maximum candidate pool size merged for fusion and decision evaluation. |
| `PIPELINE_BIENCODER_TEMP` | `0.06` | Softmax temperature for bi-encoder fusion; amplifies small cosine deltas into decisable confidence gaps. |
| `PIPELINE_BIENCODER_AGREE_BOOST`| `0.05` | Confidence bonus added when text top-1 and image top-1 agree on the same service. |
| `DECISION_CONFIDENCE_THRESHOLD` | `0.40` | Minimum top-1 confidence required to qualify for `auto_create`. |
| `DECISION_GAP_THRESHOLD` | `0.10` | Minimum gap ($\text{conf}_1 - \text{conf}_2$) required to qualify for `auto_create`. |
| `DECISION_NO_ACTION_FLOOR` | `0.25` | Confidence floor below which requests are treated as non-actionable (`no_action`). |
| `DECISION_REQUIRE_AGREE` | `false` | If set to `true`, `auto_create` requires text rank #1 and image rank #1 to match. |
| `AUTO_CREATE_ENABLED` | `false` | Master toggle: when `false`, runs in shadow mode (routes would-be auto to draft). |
| `MIN_DESCRIPTION_WORDS` | `3` | Minimum word count in description required for `auto_create` eligibility. |
| `ALWAYS_DRAFT_SERVICES` | `["Elevator...", ...]` | JSON list of high-risk services that always require human review. |

