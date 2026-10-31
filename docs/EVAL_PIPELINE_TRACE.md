# Evaluation and Processing Pipeline Trace

This document describes the end-to-end execution of a triage request and the evaluation scoring suite in the `siglip` service (`C:\Users\Win11\code\shamixo\siglip`). Every statement, threshold, formula, and rule is cited from the codebase.

---

## 1. Overview

The lifecycle of an incoming post triage request follows a strict sequence of 15 steps from reception to response:

1. **HTTP Request Ingestion**: The client issues `POST /v1/posts/process` with multipart form data (`image`, `description`, `location`, `post_id`, or `Idempotency-Key` header) received by `process_post()` in [`app/routers/posts.py:100-225`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L100-L225).
2. **Authentication Verification**: `_require_auth()` performs constant-time Bearer token verification against `settings.api_key_list` in [`app/routers/posts.py:28-48`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L28-L48).
3. **Idempotency Replay Check**: If an idempotency key is present, `repo.get_decision_log_by_idempotency_key()` queries PostgreSQL. If a prior record exists, a reconstructed `ProcessedPost` is returned immediately without running the pipeline in [`app/routers/posts.py:125-159`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L125-L159) and [`app/db/repository.py:223-261`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L223-L261).
4. **Image Format & Safety Validation**: Bytes are inspected; PIL loads the image to verify format (`JPEG`, `PNG`, `WEBP`) and enforce decompression bomb limits (`MAX_IMAGE_PIXELS = 178_956_970`) in [`app/routers/posts.py:161-192`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L161-L192).
5. **Pipeline Engine Dispatch**: `engine.process_post()` is invoked with image bytes, description, DB pool, and service instances in [`app/routers/posts.py:204-215`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L204-L215) and [`app/pipeline/engine.py:36-350`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L36-L350).
6. **Image Decoding & Hashing**: PIL converts raw image bytes to RGB (`Image.open(io.BytesIO(image_bytes)).convert("RGB")`) and computes the SHA-256 hex digest (`hashlib.sha256(image_bytes).hexdigest()`) in [`app/pipeline/engine.py:79-82`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L79-L82).
7. **Text Query Embedding**: If a description is provided, it is prepended with `BGE_QUERY_PREFIX` and sent to the TEI embedder via `tei_client.embed()` to produce a 768-dimensional query vector in [`app/pipeline/engine.py:92-98`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L92-L98) and [`app/tei_client.py:107-126`](file:///c:/Users/Win11/code/shamixo/siglip/app/tei_client.py#L107-L126).
8. **pgvector Text Search**: `repo.text_search()` executes SQL cosine similarity against active `WoService` records, combining description and title embedding distances into a single weighted score in [`app/pipeline/engine.py:101-114`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L101-L114) and [`app/db/repository.py:43-76`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L43-L76).
9. **Zero-Shot Image Ranking**: In an off-thread worker (`asyncio.to_thread`), `rank_image_against_profiles()` encodes the image using `siglip_service.encode_images()` and computes dot products against cached `ServiceImageProfile` vectors in [`app/pipeline/engine.py:128-145`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L128-L145), [`app/pipeline/image_rank.py:21-99`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/image_rank.py#L21-L99), and [`app/model.py:134-150`](file:///c:/Users/Win11/code/shamixo/siglip/app/model.py#L134-L150).
10. **Candidate Pool Formation**: `build_candidate_pool()` unifies text top-$N$ and image top-$N$, deduplicating services by name and sorting by minimum modal rank up to `PIPELINE_POOL_MAX` in [`app/pipeline/engine.py:153-158`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L153-L158) and [`app/pipeline/pool.py:11-77`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/pool.py#L11-L77).
11. **Cross-Encoder Reranking / Scoring**:
    - *Cross-encoder enabled (`ENABLE_RERANKER=True`)*: `tei_client.rerank()` scores the raw description text against each candidate passage with `raw_scores=True`; raw logits are mapped to probabilities via `apply_reranker_softmax()` in [`app/pipeline/engine.py:166-191`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L166-L191) and [`app/pipeline/decision.py:70-93`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L70-L93).
    - *Cross-encoder disabled (`ENABLE_RERANKER=False`)*: An adjusted bi-encoder score with multimodal agreement boost is passed to `softmax_with_temperature(temperature=0.06)` in [`app/pipeline/engine.py:194-262`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L194-L262).
12. **Agreement Evaluation**: The top-1 prediction from the reranked stage is compared with the top-1 prediction from the image search to evaluate multimodal agreement (`agree = bool(image_top1_name == reranked[0]["service"])`) in [`app/pipeline/engine.py:266-275`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L266-L275).
13. **Rule-Based Decision Making**: `make_decision()` evaluates business safety checks, word counts, always-draft service lists, confidence floors, and auto-create thresholds in [`app/pipeline/engine.py:278-295`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L278-L295) and [`app/pipeline/decision.py:98-270`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L98-L270).
14. **Decision Persistence**: When `dry_run=False`, the execution metadata, timing stages, and candidate ranks are inserted into `DecisionLog` in PostgreSQL inside a serialized transaction in [`app/pipeline/engine.py:315-349`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L315-L349) and [`app/db/repository.py:135-207`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L135-L207).
15. **Response Serialization**: `_build_response()` converts the `ProcessedPost` object into JSON schema containing decision ID, action, confidence, gap, agreement, and candidate breakdowns in [`app/routers/posts.py:51-96`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L51-L96) and [`app/routers/posts.py:225`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L225).

---

## 2. Inputs and Preprocessing

### Request Parameters
The endpoint `POST /v1/posts/process` accepts multipart form parameters defined in [`app/routers/posts.py:100-108`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L100-L108):
- `image`: Optional uploaded image binary (`UploadFile | None = File(None)`).
- `description`: Optional text string (`str | None = Form(None)`).
- `post_id`: Optional identifier used for idempotency fallback (`str | None = Form(None)`).
- `location`: Optional string metadata (`str | None = Form(None)`).
- `Idempotency-Key`: Optional header taking precedence over `post_id` (`Header(None, alias="Idempotency-Key")`).

At least one modal input is required by [`app/routers/posts.py:194-201`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L194-L201):
```python
if image_bytes is None and not (description and description.strip()):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "code": "INPUT_REQUIRED",
            "message": "At least one of 'image' or 'description' must be provided",
        },
    )
```

### Image Decoding and Safety Checks
- **Upload Size Limit**: Checked against `settings.MAX_UPLOAD_BYTES` (default: `10_485_760` bytes = 10 MB, defined in [`app/config.py:37`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L37), env `MAX_UPLOAD_BYTES`). Exceeding this raises `HTTP 413 REQUEST_ENTITY_TOO_LARGE` in [`app/routers/posts.py:165-172`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L165-L172).
- **Decompression Bomb Guard**: Set globally via `Image.MAX_IMAGE_PIXELS = 178_956_970` in [`app/routers/posts.py:25`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L25).
- **Format Validation**: Decoded via `Image.open(io.BytesIO(raw)); pil.load()`. Formats strictly permitted are `"JPEG"`, `"PNG"`, and `"WEBP"` in [`app/routers/posts.py:174-185`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L174-L185).
- **Color Conversion**: `pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")` in [`app/pipeline/engine.py:80`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L80).
- **Resizing**: The application code does **not** manually resize image inputs before embedding. Instead, resizing and tensor normalization to 224x224 are handled natively inside HuggingFace's `AutoProcessor` call (`self.processor(images=images, return_tensors="pt")`) in [`app/model.py:142`](file:///c:/Users/Win11/code/shamixo/siglip/app/model.py#L142).

### Empty or Short Descriptions
- **Empty / Missing Description**: If `description` is missing or contains only whitespace (`not trimmed_desc`), `make_decision()` bypasses text ranking and cross-encoder reranking, immediately returning `action="draft"` with `reason="Description missing: manual triage required (ranked by image only)"` in [`app/pipeline/decision.py:125-154`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L125-L154).
- **Short Description**: If trimmed word count is below `min_description_words` (`len(trimmed_desc.split()) < min_description_words`), `make_decision()` immediately routes the post to `action="draft"` with `reason="Description has X word(s) (< Y words required)"` in [`app/pipeline/decision.py:181-193`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L181-L193). The default threshold is `MIN_DESCRIPTION_WORDS = 3` in [`app/config.py:77`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L77) (env `MIN_DESCRIPTION_WORDS`).

---

## 3. Text Retrieval

### Model and Embedding Dimension
- **Model**: `BAAI/bge-base-en-v1.5` served via Text Embeddings Inference (TEI) at `TEI_EMBED_URL` (default: `"http://tei-embed:80"`, in `.env`: `"http://localhost:8081"`, cited from [`app/config.py:46`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L46) and [`app/pipeline/engine.py:338`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L338)).
- **Dimensions**: 768 dimensions (`_EXPECTED_TEXT_DIM = 768` in [`app/db/schema_verify.py:40`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/schema_verify.py#L40); verified at startup in [`app/main.py:186-196`](file:///c:/Users/Win11/code/shamixo/siglip/app/main.py#L186-L196)).

### Embedded Text
- **Query**: The user description is prepended with the BGE asymmetric query instruction prefix defined in [`app/pipeline/engine.py:33, 93`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L33):
  `prefixed = "Represent this sentence for searching relevant passages: " + description.strip()`
- **Catalog Passages**: Passages in `WoService` are embedded **without** the prefix during catalog sync: `vecs = await tei.embed([title, desc])` generates `titleEmb` from the service title and `descEmb` from the service description in [`app/routers/services.py:101-118`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/services.py#L101-L118). Keywords are not included in the embedding vector.

### Database Query and Scoring Formula
The query executes via `pgvector` in [`app/db/repository.py:18-40`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L18-L40):
```sql
SELECT id, title, description,
       1 - ("titleEmb" <=> %(q)s::vector) AS title_sim,
       1 - ("descEmb"  <=> %(q)s::vector) AS desc_sim,
       %(w)s * (1 - ("descEmb"  <=> %(q)s::vector))
         + (1 - %(w)s) * (1 - ("titleEmb" <=> %(q)s::vector)) AS score
FROM "WoService"
WHERE "isActive"
ORDER BY score DESC
LIMIT %(top_n)s;
```
- **Distance Metric**: Cosine distance using the `<=>` operator. Cosine similarity is computed as `1 - (vector <=> query)`.
- **Top N**: Controlled by `PIPELINE_TEXT_TOP_N` (default: `10`, in [`app/config.py:64`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L64), env `PIPELINE_TEXT_TOP_N`).
- **Description Weight**: Controlled by `PIPELINE_TEXT_DESC_WEIGHT` (default: `0.7`, in [`app/config.py:63`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L63), env `PIPELINE_TEXT_DESC_WEIGHT`).
- **Scoring Equation**:
  $$\text{score} = 0.7 \times \text{desc\_sim} + 0.3 \times \text{title\_sim}$$

---

## 4. Image Retrieval

### Model and Embedding Computation
- **Model**: `google/siglip2-base-patch16-224` (`SIGLIP_MODEL_ID`, default: `"google/siglip2-base-patch16-224"` in [`app/config.py:53`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L53), `.env`: `"google/siglip2-base-patch16-224"`).
- **Execution**: Run on CPU in inference mode within an isolated thread pool using `asyncio.to_thread` in [`app/pipeline/engine.py:129-144`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L129-L144).
- **Encoding Code** ([`app/model.py:134-150`](file:///c:/Users/Win11/code/shamixo/siglip/app/model.py#L134-L150)):
```python
def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
    if not self._is_loaded:
        raise RuntimeError("Model is not loaded.")
    if not images:
        return []
    inputs = self.processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        out = self.model.get_image_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else (out[0] if isinstance(out, tuple) else out)
        feat = F.normalize(feat, p=2, dim=-1)
        embeddings = feat.cpu().tolist()
    return embeddings
```

### Service Profiles and Cached Vectors
- **Profile Loading**: Loaded from PostgreSQL table `ServiceImageProfile` into memory at startup via `get_all_service_profiles()` in [`app/main.py:216`](file:///c:/Users/Win11/code/shamixo/siglip/app/main.py#L216) and [`app/db/repository.py:78-132`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L78-L132).
- **Embedded Fields**: Embedded during service catalog sync as:
  `image_text = f"{title}. {visual_line}" if visual_line else title` in [`app/routers/services.py:90`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/services.py#L90).
  It encodes the service `title` concatenated with `visual_line` (a descriptive prompt of visual cues).
- **Vector Normalization**: Vectors are L2-normalized upon text encoding (`F.normalize(feat, p=2, dim=-1)` in [`app/model.py:129`](file:///c:/Users/Win11/code/shamixo/siglip/app/model.py#L129)) and re-normalized when loaded from the database in [`app/db/repository.py:120-122`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L120-L122):
  ```python
  norm = torch.norm(tensor_vec, p=2)
  if norm > 0:
      tensor_vec = tensor_vec / norm
  ```

### Similarity Computation and Middle-Rank Rule
- **Cosine Dot Product**: Because both image tensor and profile text vector are unit L2-normalized, similarity is computed as a raw dot product in [`app/pipeline/image_rank.py:51`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/image_rank.py#L51):
  ```python
  sim = float(torch.dot(img_tensor, vec).item())
  ```
  No temperature or logit scaling is applied.
- **Middle-Rank Rule**: Services without a defined `visual_line` are placed at `middle_rank = (total_services + 1) / 2.0` (rank `10.0` for 19 services) with score `0.0` and `has_visual_vote = False` in [`app/pipeline/image_rank.py:40-82`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/image_rank.py#L40-L82). Non-visual services are filtered out of the top results via `valid = [item for item in ranked_all if item["has_visual_vote"]]` in [`app/pipeline/image_rank.py:96`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/image_rank.py#L96).
- **Top N**: Capped at `PIPELINE_IMAGE_TOP_N` (default: `5` in [`app/config.py:65`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L65), env `PIPELINE_IMAGE_TOP_N`).

---

## 5. Candidate Pool

Merging is handled by `build_candidate_pool()` in [`app/pipeline/pool.py:11-78`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/pool.py#L11-L78):

```python
def build_candidate_pool(
    text_top: list[dict[str, Any]] | None,
    image_top: list[dict[str, Any]] | None,
    pool_max: int = 10,
) -> list[dict[str, Any]]:
```

### Merging and Deduplication
1. **Text Addition**: All candidates from `text_top` (up to 10) are inserted into `candidates_by_name` indexed by service name (`svc`). Each entry stores `text_rank = float(item["rank"])`, `image_rank = float("inf")`, and `text_score`.
2. **Image Addition & Deduplication**: Candidates from `image_top` (up to 5) are processed. If `svc` already exists in `candidates_by_name`, `image_rank` and `image_score` are updated on the existing record. If it is new, it is inserted with `text_rank = float("inf")`, `image_rank = float(item["rank"])`, and `text_score = 0.0`.
3. **Minimum Rank Determination**: For each candidate, `min_rank = min(text_rank, image_rank)` in [`app/pipeline/pool.py:62`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/pool.py#L62).
4. **Tie Breaking and Sorting**: In [`app/pipeline/pool.py:65`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/pool.py#L65), candidates are sorted lexicographically by:
   ```python
   pool.sort(key=lambda x: (x["min_rank"], x["text_rank"], x["image_rank"], x["service"]))
   ```
5. **Capacity Capping**: Capped at `PIPELINE_POOL_MAX` (default: `10` in [`app/config.py:66`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L66), env `PIPELINE_POOL_MAX`). Infinite ranks are converted back to `None`.

### One-Sided Pool Behavior
- **Text Only (No Image)**: `image_top` is `None` or `[]`. Every candidate has `image_rank = None` and `min_rank = text_rank`. The pool is ordered by `text_rank` up to 10 candidates.
- **Image Only (No Description)**: `text_top` is `None` or `[]`. Every candidate has `text_rank = None` and `min_rank = image_rank`. The pool is ordered by `image_rank` up to 5 candidates.

---

## 6. Reranker

### Cross-Encoder Query and Document Formulation
When `ENABLE_RERANKER=True` (default: `True` in [`app/config.py:68`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L68), env `ENABLE_RERANKER`), the candidate texts are constructed in [`app/pipeline/engine.py:168-178`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L168-L178):

```python
candidate_texts = [
    f"{item['service']}. {item.get('description', '')}".strip()
    for item in candidate_pool
]
raw_results = await tei_client.rerank(
    description.strip(),
    candidate_texts,
    raw_scores=True,
)
```

- **Query**: `description.strip()`
- **Candidate Documents**: `f"{item['service']}. {item.get('description', '')}".strip()` (service title and catalog description).
- **Image Contribution**: The image contributes **nothing** to the cross-encoder reranker. The reranker operates exclusively as a cross-attention transformer over the raw user text description and candidate text descriptions.

### TEI Endpoint Call and Raw Logits
- **Endpoint**: `POST {rerank_url}/rerank` via `tei_client.rerank()` in [`app/tei_client.py:128-143`](file:///c:/Users/Win11/code/shamixo/siglip/app/tei_client.py#L128-L143).
- **Payload**: `{"query": description.strip(), "texts": candidate_texts, "raw_scores": True}`.
- **Model**: `BAAI/bge-reranker-base` (logged in [`app/pipeline/engine.py:340`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L340)).
- **Output**: Returns raw unbound transformer logits: `[{"index": 0, "score": float}, ...]`.

### Softmax, Temperature, and Confidence Calculation
Raw logits are normalized using `apply_reranker_softmax()` in [`app/pipeline/decision.py:70-93`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L70-L93) and `softmax_with_temperature()` in [`app/pipeline/decision.py:16-29`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L16-L29):

```python
def softmax_with_temperature(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    if temperature <= 0.0:
        raise ValueError(f"Softmax temperature must be strictly positive, got {temperature}")
    arr = np.array(scores, dtype=np.float64) / temperature
    arr_shifted = arr - np.max(arr)
    exp_vals = np.exp(arr_shifted)
    return (exp_vals / np.sum(exp_vals)).tolist()
```

- **Config Key**: `PIPELINE_RERANKER_TEMP: float = 1.0` in [`app/config.py:67`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L67) (env `PIPELINE_RERANKER_TEMP`).
- **Confidence & Gap Calculation** ([`app/pipeline/engine.py:188-191`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L188-L191)):
```python
if reranked:
    confidence = float(reranked[0]["confidence"])
    gap = float(confidence - reranked[1]["confidence"]) if len(reranked) > 1 else confidence
```
Confidence is the top-1 softmax probability. Gap is the arithmetic difference between the rank-1 and rank-2 softmax probabilities.

### Bi-Encoder Fallback (When Reranker is Disabled)
When `ENABLE_RERANKER=False`, [`app/pipeline/engine.py:209-262`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L209-L262) computes raw adjusted scores:
- Primary signal: Text cosine similarity `t_s`.
- Multimodal agreement boost: If both text and image rank the candidate at rank 1 (`t_r == 1.0 and i_r == 1.0`), add `PIPELINE_BIENCODER_AGREE_BOOST` (default `0.05` in [`app/config.py:69`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L69)). If image rank is $\le 3.0$, add $0.5 \times 0.05 = 0.025$.
- Softmax scaling: Passed to `softmax_with_temperature(raw_adjusted, temperature=PIPELINE_BIENCODER_TEMP)` where temperature defaults to `0.06` in [`app/config.py:70`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L70).

---

## 7. Decision

### Full `make_decision` Function
Verbatim code from [`app/pipeline/decision.py:98-270`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/decision.py#L98-L270):

```python
def make_decision(
    *,
    description: str | None,
    has_image: bool,
    text_top: list[dict[str, Any]],
    image_top: list[dict[str, Any]] | None,
    pool: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    confidence: float | None,
    gap: float | None,
    agree: bool | None,
    # config
    auto_create_enabled: bool,
    confidence_threshold: float,
    gap_threshold: float,
    no_action_floor: float,
    require_agree: bool,
    always_draft_services: list[str],
    min_description_words: int,
    top_candidates_count: int = 3,
) -> ProcessedPost:
    trimmed_desc = description.strip() if description else ""

    # Case 1: Description missing → draft, rank by image only
    if not trimmed_desc:
        candidates: list[CandidateOut] = []
        all_candidates: list[CandidateOut] = []
        if image_top:
            for idx, item in enumerate(image_top, start=1):
                score = item.get("score")
                c_out = CandidateOut(
                    service_id=item.get("service_id"),
                    title=item["service"],
                    confidence=score,
                    confidence_pct=f"{score * 100:.1f}%" if score is not None else None,
                    image_rank=float(item.get("rank", 0)),
                    pool_rank=idx,
                    rerank_rank=idx,
                )
                all_candidates.append(c_out)
                if len(candidates) < top_candidates_count:
                    candidates.append(c_out)
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=None,
            gap=None,
            agree=agree,
            reason="Description missing: manual triage required (ranked by image only)",
        )

    # Build candidate list from reranked (or pool fallback)
    candidate_source = reranked if reranked else pool
    candidates = []
    all_candidates = []
    for idx, item in enumerate(candidate_source, start=1):
        conf = item.get("confidence")
        c_out = CandidateOut(
            service_id=item.get("service_id"),
            title=item["service"],
            confidence=conf,
            confidence_pct=f"{conf * 100:.1f}%" if conf is not None else None,
            text_rank=item.get("text_rank"),
            image_rank=item.get("image_rank"),
            pool_rank=item.get("pool_rank"),
            rerank_score=item.get("rerank_score"),
            rerank_rank=item.get("rerank_rank", idx),
            raw_score=item.get("rerank_score"),
        )
        all_candidates.append(c_out)
        if len(candidates) < top_candidates_count:
            candidates.append(c_out)

    top_service = candidates[0].title if candidates else ""

    # Case 2: Short description → draft
    words = trimmed_desc.split()
    if len(words) < min_description_words:
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Description has {len(words)} word(s) (< {min_description_words} words required)",
        )

    # Case 3: Always-draft services
    if top_service in always_draft_services:
        return ProcessedPost(
            decision_id=None,
            action="draft",
            would_be_action="draft",
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Service '{top_service}' requires mandatory human draft review for safety",
        )

    # Case 4: Below confidence floor → no_action
    if confidence is not None and confidence < no_action_floor:
        return ProcessedPost(
            decision_id=None,
            action="no_action",
            would_be_action="no_action",
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=f"Confidence {confidence:.4f} is below floor threshold ({no_action_floor})",
        )

    # Case 5: Meets auto_create criteria
    meets_conf = (confidence is not None) and (confidence >= confidence_threshold)
    meets_gap = (gap is not None) and (gap >= gap_threshold)
    meets_agree = (agree is True) or (agree is None) or (not require_agree)

    if meets_conf and meets_gap and meets_agree:
        would_be = "auto_create"
        if auto_create_enabled:
            action = "auto_create"
            reason = (
                f"High confidence ({confidence:.4f} >= {confidence_threshold}) "
                f"and gap ({gap:.4f} >= {gap_threshold})"
            )
            shadow = False
        else:
            action = "draft"
            reason = (
                f"Shadow mode: would be auto_create "
                f"(conf={confidence:.4f}>={confidence_threshold}, gap={gap:.4f}>={gap_threshold}), "
                f"routed to draft"
            )
            shadow = True
        return ProcessedPost(
            decision_id=None,
            action=action,
            would_be_action=would_be,
            top_candidates=candidates,
            all_candidates=all_candidates,
            confidence=confidence,
            gap=gap,
            agree=agree,
            reason=reason,
            shadow=shadow,
        )

    # Case 6: Default → draft
    return ProcessedPost(
        decision_id=None,
        action="draft",
        would_be_action="draft",
        top_candidates=candidates,
        all_candidates=all_candidates,
        confidence=confidence,
        gap=gap,
        agree=agree,
        reason=f"Criteria for auto_create not met (conf={confidence}, gap={gap}, agree={agree})",
    )
```

### Detailed Evaluation Rules in Order of Execution
1. **Rule 1 (Missing Description)**:
   - Evaluated: `not trimmed_desc`
   - Compared against: Empty string `""`
   - Comparison: Truthiness / string length `== 0`
   - Action: `draft` (`would_be_action = "draft"`, `confidence = None`, `gap = None`)
2. **Rule 2 (Short Description)**:
   - Evaluated: `len(words)`
   - Compared against: `min_description_words` (default: `3` in [`app/config.py:77`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L77), env `MIN_DESCRIPTION_WORDS`)
   - Comparison operator: `<`
   - Action: `draft` (`would_be_action = "draft"`)
3. **Rule 3 (Always-Draft Services)**:
   - Evaluated: `top_service` (title of rank-1 candidate)
   - Compared against: `always_draft_services` (default: `["Emergency Response & Area Securing", "Emergency Reactive Cleaning (Spills, Vomit, Incidents)"]` in [`app/config.py:78`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L78), env `ALWAYS_DRAFT_SERVICES`)
   - Comparison operator: `in`
   - Action: `draft` (`would_be_action = "draft"`)
4. **Rule 4 (Confidence Below Floor)**:
   - Evaluated: `confidence` (**Check Value: Top-1 post-softmax probability** `reranked[0]["confidence"]`, NOT raw cross-encoder score)
   - Compared against: `no_action_floor` (default: `0.25` in [`app/config.py:75`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L75), env `DECISION_NO_ACTION_FLOOR`)
   - Comparison operator: `<`
   - Action: `no_action` (`would_be_action = "no_action"`)
5. **Rule 5 (Meets Auto-Create Criteria)**:
   - Sub-rule 5a: `confidence >= confidence_threshold` (default: `0.40` in [`app/config.py:73`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L73), env `DECISION_CONFIDENCE_THRESHOLD`, operator `>=`)
   - Sub-rule 5b: `gap >= gap_threshold` (default: `0.10` in [`app/config.py:74`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L74), env `DECISION_GAP_THRESHOLD`, operator `>=`)
   - Sub-rule 5c: `(agree is True) or (agree is None) or (not require_agree)` (default: `DECISION_REQUIRE_AGREE = False` in [`app/config.py:76`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L76), set to `true` in `.env`)
   - If 5a, 5b, and 5c are all met:
     - `would_be_action = "auto_create"`
     - If `auto_create_enabled == True` (default: `False` in [`app/config.py:72`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L72), set to `true` in `.env`): `action = "auto_create"`, `shadow = False`
     - If `auto_create_enabled == False`: `action = "draft"`, `shadow = True`
6. **Rule 6 (Default Fallback)**:
   - Action: `draft` (`would_be_action = "draft"`)

### Decision Flow Logic
```
IF description is missing THEN
    RETURN action = draft, reason = "Description missing"
ELSE IF word_count(description) < min_description_words THEN
    RETURN action = draft, reason = "Description has X words (< Y required)"
ELSE IF top_candidate.title IN always_draft_services THEN
    RETURN action = draft, reason = "Service requires mandatory human review"
ELSE IF top1_softmax_confidence < no_action_floor THEN
    RETURN action = no_action, reason = "Confidence below floor threshold"
ELSE IF (top1_softmax_confidence >= confidence_threshold)
    AND (softmax_gap >= gap_threshold)
    AND (agree == True OR agree == None OR require_agree == False) THEN
    IF auto_create_enabled == True THEN
        RETURN action = auto_create, shadow = False
    ELSE
        RETURN action = draft, shadow = True (Shadow mode)
ELSE
    RETURN action = draft, reason = "Criteria for auto_create not met"
END IF
```

---

## 8. Logging and Dry-Run

### Database Writes in Normal Requests
When `dry_run=False` (default in standard HTTP triage requests), [`app/pipeline/engine.py:315-349`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L315-L349) invokes `repo.insert_decision_log()` in [`app/db/repository.py:135-207`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/repository.py#L135-L207):
- Acquires an advisory transaction lock if an idempotency key is supplied: `SELECT pg_advisory_xact_lock(hashtext(%s));`.
- Writes one row to table `"DecisionLog"` containing:
  - `description` (censored if `STORE_DESCRIPTION=False`)
  - `photoRef` (`None`)
  - `photoHash` (SHA-256 string)
  - `location`
  - `textTop` (JSON array of text rankings)
  - `imageTop` (JSON array of image rankings)
  - `pool` (JSON array of merged candidate pool)
  - `rerankScores` (JSON array of rerank scores and confidences)
  - `topServiceId` (integer service ID)
  - `confidence` (float)
  - `gap` (float)
  - `agree` (boolean)
  - `action` (`"auto_create"`, `"draft"`, or `"no_action"`)
  - `wouldBeAction`
  - `reason`
  - `models` (JSON object storing text, image, and reranker model IDs and idempotency key)
  - `configVersion`
  - `timings` (JSON object with stage latencies in ms)

### SearchLog Table
**NOT FOUND**. The `SearchLog` table does **not** exist in the `siglip` service codebase. The database schema verification in [`app/db/schema_verify.py:20-37`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/schema_verify.py#L20-L37) checks only three tables: `"WoService"`, `"ServiceImageProfile"`, and `"DecisionLog"`. No repository function writes to `SearchLog`.

### Confirmation: Zero Database Writes During Evaluation Runs
During an evaluation run, **nothing is written to the database**.
- Verification of write paths: The sole database write path in the pipeline is `repo.insert_decision_log()`, which is guarded in [`app/pipeline/engine.py:315`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L315) by:
  ```python
  if not dry_run:
      ...
      decision_id = await repo.insert_decision_log(pool, log_data)
  ```
- Both the evaluation warm-up pass ([`app/eval/job_manager.py:372`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L372)) and the evaluation sequential worker loop ([`app/eval/job_manager.py:418`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L418)) pass `dry_run=True`.
- The only database operations executed during evaluation are read-only:
  1. `repo.get_all_services(..., include_inactive=False)` (`SELECT` in [`app/routers/eval.py:88`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/eval.py#L88)).
  2. `repo.text_search(..., return_all=True)` (`SELECT` cosine similarity search in [`app/pipeline/engine.py:102`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L102)).

---

## 9. Evaluation Flow

```
[ZIP Upload / Local Path]
         │
         ▼
[inspect_and_extract_zip]  ──► Enforce zip-slip, byte/count limits, extract to tempdir
         │
         ▼
[validate_ground_truth]    ──► Parse CSV, check image files, normalize NONE, fuzzy-match services
         │
         ▼
[create_job]               ──► Allocate job_id, store report, register EvalJob
         │
         ▼
[start_eval_job]           ──► Background task launch (_run_evaluation_worker)
         │
         ├─► [Warm-Up Pass] ──────────► Process row 1 with dry_run=True (excluded from metrics)
         │
         ├─► [Sequential Eval Loop] ──► For each row:
         │                                - Save 1280px preview image to disk
         │                                - Create 96px base64 thumbnail
         │                                - engine.process_post(dry_run=True, return_diagnostics=True)
         │                                - Classify correctness (NONE vs Normal)
         │                                - Recompute cumulative summary metrics
         │                                - Broadcast SSE 'row' event
         │
         ├─► [Final Summary] ─────────► compute_metrics() over all non-warmup rows
         ├─► [Save to Disk] ──────────► Write <job_id>.json to EVAL_RUNS_DIR; prune old runs
         ├─► [Cleanup] ───────────────► rmtree temp extraction directory
         └─► [Broadcast SSE] ─────────► Emit 'complete' event to browser clients
```

### File and Line Citations
- **Upload & ZIP Validation**: Handled in [`app/routers/eval.py:109-134`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/eval.py#L109-L134) (`POST /v1/eval/upload`) and `inspect_and_extract_zip()` in [`app/eval/zip_handler.py:39-173`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/zip_handler.py#L39-L173). Validates `EVAL_MAX_ZIP_BYTES` (default: `104_857_600` = 100 MB), rejects path traversal (`..`, `:`, leading `/`), limits entries to `EVAL_MAX_IMAGE_COUNT` (default: `1000`), checks image sizes against `EVAL_MAX_IMAGE_BYTES` (default: `10_485_760` = 10 MB), and enforces total uncompressed size limit `EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES` (default: `524_288_000` = 500 MB). Extracts to a secure directory created via `tempfile.mkdtemp(prefix="siglip_eval_")`.
- **CSV Parsing & Ground Truth Validation**: `validate_ground_truth()` in [`app/eval/validator.py:58-219`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/validator.py#L58-L219). Parses `ground_truth.csv`, validates mandatory columns `image` and `service`, identifies duplicates, parses `NONE` case-insensitively, normalizes valid service titles, and invokes `im.verify()` on each image.
- **Job Creation**: `EvalJobManager.create_job()` in [`app/eval/job_manager.py:202-218`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L202-L218) creates an `EvalJob` with status `"validated"`.
- **Job Start Endpoint**: `POST /v1/eval/jobs/{job_id}/start` in [`app/routers/eval.py:213-245`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/eval.py#L213-L245) accepts optional overrides (`description_weight`, `confidence_threshold`, `gap_threshold`, `no_action_floor`, `enable_reranker`) and launches `_run_evaluation_worker` via `asyncio.create_task()`.
- **Warm-Up Execution**: The first valid row is executed once before the timer starts in [`app/eval/job_manager.py:352-376`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L352-L376) to warm caches and JIT compilers, tagged with `is_warmup=True`.
- **Per-Row Engine Call**: Invoked in [`app/eval/job_manager.py:407-420`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L407-L420):
  ```python
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
  ```
- **Diagnostics Collected**: `return_diagnostics=True` causes `repo.text_search(..., return_all=True)` and `rank_image_against_profiles(..., return_all_ranks=True)` to attach complete 19-service catalog rankings to `result.full_text_ranks` and `result.full_image_ranks` in [`app/pipeline/engine.py:306-312`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L306-L312).
- **Timing Stages Collected**: Recorded per-row in `result.timings`: `image_decode_ms`, `siglip_encode_ms`, `text_embed_ms`, `pgvector_search_ms`, `text_ms`, `image_ms`, `pool_ms`, `rerank_ms`, `decision_ms`, `total_ms` in [`app/pipeline/engine.py:60-72, 298`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L60-L72).
- **Thumbnail and Preview Creation**:
  - Table thumbnail: `_encode_thumbnail_b64()` scales down to max 96x96 pixels and encodes a base64 JPEG data URI in [`app/eval/job_manager.py:35-46, 398`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L35-L46).
  - Modal high-resolution preview: `_save_preview_image()` scales the image to max 1280px on its longest side (JPEG quality ~80) and saves it to `<EVAL_RUNS_DIR>/<job_id>/previews/<row_id>.jpg` in [`app/eval/job_manager.py:49-61, 401-402`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L49-L61).
- **SSE Events Broadcast**: Broadcasted by `_broadcast()` in [`app/eval/job_manager.py:316-323`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L316-L323) to clients connected to `GET /v1/eval/jobs/{job_id}/stream`:
  - `init`: catch-up snapshot upon connection in [`app/eval/job_manager.py:292-301`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L292-L301).
  - `status`: status announcements (e.g. `"warmup"`) in [`app/eval/job_manager.py:353`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L353).
  - `row`: streamed live for each completed evaluation row in [`app/eval/job_manager.py:526-531`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L526-L531).
  - `stopped`: emitted if aborted early in [`app/eval/job_manager.py:383`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L383).
  - `complete`: final summary report payload in [`app/eval/job_manager.py:555-560`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L555-L560).
- **Run Persistence & Pruning**: `_save_run_to_disk()` saves `<job_id>.json` to `EVAL_RUNS_DIR` (default: `"app/eval/runs"` in [`app/config.py:83`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L83)) and deletes older runs beyond `EVAL_MAX_HISTORY_RUNS` (default: `50` in [`app/config.py:84`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L84)) including their `previews/` folders in [`app/eval/job_manager.py:182-200, 574-584`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L182-L200). Temp dataset directories are deleted in the `finally:` block via `cleanup_temp_dir()` in [`app/eval/job_manager.py:569-573`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L569-L573).

---

## 10. Metrics Definitions

### Metrics Formulas and Denominators
All metrics are computed in `compute_metrics()` in [`app/eval/metrics.py:80-615`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L80-L615). Warmup rows are strictly excluded (`eval_rows = [r for r in rows if not r.is_warmup]`).

```python
# Subsets
non_none_rows = [r for r in eval_rows if not r.is_none]
none_rows = [r for r in eval_rows if r.is_none]
non_none_n = len(non_none_rows)
total_n = len(eval_rows)
```

1. **Top-1 Accuracy**:
   - Code ([`app/eval/metrics.py:142`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L142)): `top1_acc = (non_none_correct / non_none_n * 100.0) if non_none_n else 0.0`
   - Denominator: `non_none_n` (number of normal rows with a true service). NONE rows are **strictly excluded**.
   - Condition: `r.correct` where `r.predicted_service == r.expected_service`.
2. **Top-3 Accuracy**:
   - Code ([`app/eval/metrics.py:135-143`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L135-L143)):
     ```python
     if r.reranker_rank is not None and r.reranker_rank <= 3:
         top3_correct += 1
     top3_acc = (top3_correct / non_none_n * 100.0) if non_none_n else 0.0
     ```
   - Denominator: `non_none_n` (NONE rows excluded).
3. **Top-5 Accuracy**:
   - Code ([`app/eval/metrics.py:138-144`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L138-L144)):
     ```python
     if r.reranker_rank is not None and r.reranker_rank <= 5:
         top5_correct += 1
     top5_acc = (top5_correct / non_none_n * 100.0) if non_none_n else 0.0
     ```
   - Denominator: `non_none_n` (NONE rows excluded).
4. **Pool Recall**:
   - Code ([`app/eval/metrics.py:133-145`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L133-L145)):
     ```python
     if r.entered_pool:
         entered_pool_count += 1
     pool_recall = (entered_pool_count / non_none_n * 100.0) if non_none_n else 0.0
     ```
   - Denominator: `non_none_n` (NONE rows excluded). Represents the upper bound for the cross-encoder reranker.
5. **Text Top-1 Accuracy**:
   - Code ([`app/eval/metrics.py:148-150`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L148-L150)):
     ```python
     text_eligible = [r for r in non_none_rows if r.text_top1 is not None]
     text_correct = sum(1 for r in text_eligible if r.text_top1 == r.expected_service)
     text_top1_acc = (text_correct / len(text_eligible) * 100.0) if text_eligible else 0.0
     ```
   - Denominator: `len(text_eligible)` (non-NONE rows where `text_top1` was produced).
6. **Image Top-1 Accuracy**:
   - Code ([`app/eval/metrics.py:152-154`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L152-L154)):
     ```python
     image_eligible = [r for r in non_none_rows if r.image_top1 is not None]
     image_correct = sum(1 for r in image_eligible if r.image_top1 == r.expected_service)
     image_top1_acc = (image_correct / len(image_eligible) * 100.0) if image_eligible else 0.0
     ```
   - Denominator: `len(image_eligible)` (non-NONE rows where `image_top1` was produced).
7. **Agreement Rate & Conditional Accuracies**:
   - Code ([`app/eval/metrics.py:156-161`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L156-L161)):
     ```python
     agree_rows = [r for r in non_none_rows if r.agree is True]
     disagree_rows = [r for r in non_none_rows if r.agree is False]
     agree_rate = (len(agree_rows) / non_none_n * 100.0) if non_none_n else 0.0
     acc_agree = (sum(1 for r in agree_rows if r.correct) / len(agree_rows) * 100.0) if agree_rows else 0.0
     acc_disagree = (sum(1 for r in disagree_rows if r.correct) / len(disagree_rows) * 100.0) if disagree_rows else 0.0
     ```
   - Denominators: `non_none_n` for `agree_rate`; `len(agree_rows)` for `acc_agree`; `len(disagree_rows)` for `acc_disagree`.
8. **Auto Precision and Wrong-Auto Count**:
   - Code ([`app/eval/metrics.py:164-174`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L164-L174)):
     ```python
     auto_rows = [r for r in eval_rows if r.action == "auto_create"]
     auto_correct = sum(1 for r in auto_rows if r.correct)
     wrong_auto_rows = [r for r in auto_rows if not r.correct]
     wrong_auto_count = len(wrong_auto_rows)
     auto_precision = (auto_correct / len(auto_rows) * 100.0) if auto_rows else 0.0
     coverage = (len(auto_rows) / total_n * 100.0) if total_n else 0.0
     ```
   - Denominator: `len(auto_rows)` (total decisions returning `auto_create`).
   - Inclusions: **Includes NONE rows**. A NONE row returning `auto_create` has `correct = False`, and therefore is counted in `wrong_auto_count` and degrades `auto_precision`.
9. **Per-Service Accuracy and Wilson Interval**:
   - Code ([`app/eval/metrics.py:205-220`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L205-L220)):
     ```python
     svc_rows = [r for r in eval_rows if r.expected_service == svc and not r.is_none]
     n_svc = len(svc_rows)
     k_svc = sum(1 for r in svc_rows if r.correct)
     acc = round(k_svc / n_svc * 100.0, 1)
     ci_low, ci_high = wilson_score_interval(k_svc, n_svc, confidence=0.95)
     ```
   - Wilson Score Interval Code ([`app/eval/metrics.py:17-36`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L17-L36)):
     ```python
     z = 1.95996 if confidence == 0.95 else 1.64485
     p_hat = k / n
     z2 = z * z
     denom = 1.0 + z2 / n
     center = p_hat + z2 / (2.0 * n)
     spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
     ci_low = max(0.0, (center - spread) / denom)
     ci_high = min(1.0, (center + spread) / denom)
     ```
10. **Latency Percentiles and Throughput**:
    - Code ([`app/eval/metrics.py:177-181, 199`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L177-L181)):
      ```python
      total_latencies = [r.timings.get("total_ms", 0.0) for r in eval_rows]
      mean_latency = float(np.mean(total_latencies))
      p50_latency = float(np.percentile(total_latencies, 50))
      p95_latency = float(np.percentile(total_latencies, 95))
      max_latency = float(np.max(total_latencies))
      throughput = round(total_n / max(total_elapsed_sec, 0.001), 2)
      ```

### Definition of Ranks Shown Per Row
Computed in [`app/eval/job_manager.py:443-456`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L443-L456):
- `"text pool rank"`: The rank ($1 \dots 10$) of the ground-truth service in the text search results (`text_top`), provided it entered the candidate pool. If not present in the candidate pool, it is `None` (rendered in UI and exports as `"not in pool"`).
- `"image pool rank"`: The rank ($1 \dots 5$) of the ground-truth service in the visual search results (`image_top`), provided it entered the candidate pool. If not present, it is `None` (rendered as `"not in pool"`).
- `"reranker rank"`: The final rank ($1 \dots 10$) assigned to the ground-truth service after cross-encoder reranking. If the ground-truth service was not in the candidate pool, it is `None` (`"not in pool"`).
- `"full text rank"`: The absolute rank ($1 \dots N$, where $N$ is total active catalog services) across the entire catalog computed via pgvector query without `LIMIT`.
- `"full image rank"`: The absolute rank ($1 \dots N$) across the entire catalog computed against all cached service profiles.
- **"Not in Pool" Meaning**: In code, `"not in pool"` means `c.title == exp_svc` was never found among the entries of `result.all_candidates` (the merged pool). Consequently, `entered_pool = False`, and `reranker_rank = None`.

---

## 11. NONE Rows

### Parsing and Validation
- **Validator Parsing**: Handled in [`app/eval/validator.py:115-119`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/validator.py#L115-L119):
  ```python
  is_none = (raw_svc.strip().upper() == "NONE")
  if is_none:
      report.none_count += 1
      effective_service = "NONE"
  ```
  Any case variation (`NONE`, `none`, `None`) is mapped to the `NONE` class and is **not** flagged as an unknown service.
- **Row Input Tagging**: Tagged with `is_none=True` in `EvalRowInput` in [`app/eval/validator.py:179`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/validator.py#L179).

### Correctness Definition for NONE Rows
In [`app/eval/job_manager.py:430-436`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L430-L436):
```python
if r_in.is_none:
    # NONE row rule: correct if action is "no action" (or "no_action")
    correct = (action in ("no_action", "no action"))
else:
    # Non-NONE row: correct if reranked top-1 equals expected
    correct = (predicted_top1 == exp_svc)
```
A NONE row is marked `correct = True` **if and only if** the decision engine returns `"no_action"` or `"no action"`. If it returns `"auto_create"` or `"draft"`, it is marked `correct = False` (a **false match**).

### Exclusions and Inclusions
- **Excluded From**:
  - Top-1, Top-3, and Top-5 accuracy ([`app/eval/metrics.py:112, 142-144`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L112)).
  - Candidate pool recall ([`app/eval/metrics.py:145`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L145)).
  - Per-service accuracy and Wilson intervals ([`app/eval/metrics.py:203, 206`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L203)).
  - Offline temperature calibration reliability bins and ECE ([`app/eval/metrics.py:398`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L398)).
- **Included In**:
  - `actions.auto_create_count`, `actions.draft_count`, and `actions.no_action_count` ([`app/eval/metrics.py:164-166`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L164-L166)).
  - `actions.wrong_auto_count` and `actions.auto_precision` ([`app/eval/metrics.py:170-173`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L170-L173)).
  - Latency and throughput stats ([`app/eval/metrics.py:177-199`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L177-L199)).
  - The confusion matrix (represented as a dedicated `"NONE"` row and column in [`app/eval/metrics.py:221-245`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L221-L245)).
  - The dedicated NONE Panel.

### The NONE Panel and Histogram
The dedicated panel is assembled in [`app/eval/metrics.py:256-314`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L256-L314):
```python
none_correct_rows = [r for r in none_rows if r.action in ("no_action", "no action")]
none_correct_count = len(none_correct_rows)
none_correct_rate = round(none_correct_count / n_none * 100.0, 1) if n_none else 0.0

none_false_match_rows = [r for r in none_rows if r.action not in ("no_action", "no action")]
false_match_count = len(none_false_match_rows)
```
- **False Matches List**: Iterates through `none_false_match_rows` collecting `row_id`, `image`, `description`, `predicted_service`, `action`, `confidence`, `gap`, and `thumbnail_b64`.
- **Confidence Histogram**: Divides $[0.0, 1.0]$ into 10 bins (`bin_edges = np.linspace(0.0, 1.0, 11)`). Compares the distribution of top-1 confidence for all NONE rows against correctly-answered normal rows to determine whether a decision threshold cleanly separates out-of-scope posts.

---

## 12. Threshold Grid and Temperature Sweep

### Threshold Sensitivity Grid
Implemented in [`app/eval/metrics.py:317-388`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L317-L388):
- **Grid Axes**:
  - `conf_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]`
  - `gap_grid = [0.00, 0.05, 0.10, 0.20]`
- **Pipeline Function Reuse**: Calls the pipeline's own `make_decision()` function verbatim:
```python
dec = make_decision(
    description=r.description if r.description else None,
    has_image=bool(r.image),
    text_top=text_top_dicts,
    image_top=image_top_dicts,
    pool=cand_dicts,
    reranked=cand_dicts,
    confidence=r.confidence,
    gap=r.gap,
    agree=r.agree,
    auto_create_enabled=True,
    confidence_threshold=conf_th,
    gap_threshold=gap_th,
    no_action_floor=current_no_action_floor,
    require_agree=require_agree,
    always_draft_services=always_draft_list,
    min_description_words=min_description_words,
)
```
- **Shared State Mutation**: Does **not** modify any shared `Settings` object. All configurations are passed directly as local arguments to `make_decision()`.
- **NONE Row Accounting in Grid**: For each cell, if `r.is_none` and `dec.action in ("auto_create", "draft")`, `none_false_matches_cnt` is incremented. If `dec.action == "auto_create"`, it increments `auto_cnt` but **not** `correct_auto_cnt`.

### Temperature Sweep and Expected Calibration Error (ECE)
Implemented in [`app/eval/metrics.py:391-506`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L391-L506):
- **Temperatures**: `temps = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]`.
- **Confidence Recomputation**: Re-applies softmax to stored `raw_score` values for rows having descriptions and raw reranker scores:
  ```python
  probs = softmax_with_temperature(raw_scores, temperature=temp)
  ```
- **Row Inclusions / Exclusions**: Evaluated strictly on rows with raw scores, excluding empty descriptions and **excluding NONE rows** (`rows_with_raw = [r for r in eval_rows if r.raw_reranker_scores and not r.is_empty_desc and not r.is_none]`).
- **Reliability Bins & ECE Formula**:
  - Bin edges: 10 bins across $[0.0, 1.0]$: `bin_edges = np.linspace(0.0, 1.0, 11)` (`0-10%`, `10-20%`, ..., `90-100%`).
  - Expected Calibration Error (ECE) calculation ([`app/eval/metrics.py:437-439`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L437-L439)):
    $$ECE = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
    ```python
    avg_conf = float(np.mean([confidences[i] for i in indices_in_bin]))
    avg_acc = float(np.mean([1.0 if is_corrects[i] else 0.0 for i in indices_in_bin]))
    ece += (bin_n / max(n_scored, 1)) * abs(avg_acc - avg_conf)
    ```

---

## 13. Worked Example

Source run artifact: [`app/eval/runs/eval_1793442297_4323b3.json`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/runs/eval_1793442297_4323b3.json). Run settings: `description_weight=0.7`, `confidence_threshold=0.4`, `gap_threshold=0.1`, `no_action_floor=0.25`, `enable_reranker=false`.

### Case A: Normal Row (Row 1)
- **Image**: `img_001.jpg`
- **Description**: `"AC in room 204 is blowing warm air"`
- **Expected Service**: `"AC Services"`
- **Text Rank List**: `text_top1 = "AC Services"`, `full_text_rank = 1`
- **Image Rank List**: `image_top1 = "AC Services"`, `full_image_rank = 1`
- **Agreement**: `agree = true`
- **Candidate Pool (Top Candidates with Raw Scores & Softmax Confidences)**:
  1. `AC Services` (ID: 1, `text_rank=1.0`, `image_rank=1.0`, `pool_rank=1`, `rerank_rank=1`, `raw_score=0.548`, `confidence=0.9307`)
  2. `Emergency Reactive Cleaning (Spills, Vomit, Incidents)` (ID: 4, `text_rank=5.0`, `image_rank=4.0`, `pool_rank=7`, `rerank_rank=2`, `raw_score=0.357`, `confidence=0.0386`)
  3. `General Areas Cleaning (Classrooms, Offices, Toilets)` (ID: 8, `text_rank=2.0`, `image_rank=None`, `pool_rank=2`, `rerank_rank=3`, `raw_score=0.2572`, `confidence=0.0073`)
  4. `Waste Management Service` (ID: 18, `text_rank=3.0`, `image_rank=None`, `pool_rank=4`, `rerank_rank=4`, `raw_score=0.2535`, `confidence=0.0069`)
  5. `Office / Administration Areas Cleaning` (ID: 11, `text_rank=4.0`, `image_rank=None`, `pool_rank=6`, `rerank_rank=5`, `raw_score=0.2532`, `confidence=0.0068`)
- **Top Metrics**: `confidence = 0.9307`, `gap = 0.8921` ($0.9307 - 0.0386$)
- **Step-by-Step Decision Rule Trace**:
  - *Case 1 (Missing Desc)*: Description is `"AC in room 204 is blowing warm air"` $\rightarrow$ False.
  - *Case 2 (Short Desc)*: Word count is 8 $\ge 3$ (`min_description_words`) $\rightarrow$ False.
  - *Case 3 (Always-Draft)*: Top service is `"AC Services"`, not in `always_draft_services` $\rightarrow$ False.
  - *Case 4 (Floor)*: Top-1 confidence is $0.9307 \ge 0.25$ (`no_action_floor`) $\rightarrow$ False.
  - *Case 5 (Auto-Create Criteria)*:
    - $0.9307 \ge 0.40$ (`confidence_threshold`) $\rightarrow$ True.
    - $0.8921 \ge 0.10$ (`gap_threshold`) $\rightarrow$ True.
    - `agree == True` $\rightarrow$ True.
    - `auto_create_enabled == True` in evaluation worker $\rightarrow$ True.
  - **Resulting Action**: `action = "auto_create"`, `correct = true` (since `"AC Services" == "AC Services"`).

---

### Case B: NONE Row (Row 63)
- **Image**: `img_063.jpg`
- **Description**: `"Projector in room 12 won't turn on"`
- **Expected Service**: `"NONE"` (`is_none = true`)
- **Text Rank List**: `text_top1 = "Door / Locker Lock Assistance"`, `full_text_rank = null`
- **Image Rank List**: `image_top1 = "School Event Setup"`, `full_image_rank = null`
- **Agreement**: `agree = false`
- **Candidate Pool (Top Candidates with Raw Scores & Softmax Confidences)**:
  1. `School Event Setup` (ID: 14, `text_rank=3.0`, `image_rank=1.0`, `pool_rank=2`, `rerank_rank=1`, `raw_score=0.3747`, `confidence=0.2798`)
  2. `Handyman Service (Minor / Urgent Repairs)` (ID: 9, `text_rank=2.0`, `image_rank=4.0`, `pool_rank=3`, `rerank_rank=2`, `raw_score=0.3644`, `confidence=0.2357`)
  3. `AC Services` (ID: 1, `text_rank=6.0`, `image_rank=3.0`, `pool_rank=5`, `rerank_rank=3`, `raw_score=0.3574`, `confidence=0.2097`)
  4. `Office / Administration Areas Cleaning` (ID: 11, `text_rank=8.0`, `image_rank=5.0`, `pool_rank=8`, `rerank_rank=4`, `raw_score=0.3193`, `confidence=0.1111`)
  5. `Door / Locker Lock Assistance` (ID: 3, `text_rank=1.0`, `image_rank=None`, `pool_rank=1`, `rerank_rank=5`, `raw_score=0.2786`, `confidence=0.0564`)
- **Top Metrics**: `confidence = 0.2798`, `gap = 0.0441` ($0.2798 - 0.2357$)
- **Step-by-Step Decision Rule Trace**:
  - *Case 1 (Missing Desc)*: Description is `"Projector in room 12 won't turn on"` $\rightarrow$ False.
  - *Case 2 (Short Desc)*: Word count is 7 $\ge 3$ $\rightarrow$ False.
  - *Case 3 (Always-Draft)*: Top service is `"School Event Setup"`, not in `always_draft_services` $\rightarrow$ False.
  - *Case 4 (Floor)*: Top-1 confidence is $0.2798 \ge 0.25$ (`no_action_floor`) $\rightarrow$ False (Did **not** drop below floor).
  - *Case 5 (Auto-Create Criteria)*:
    - $0.2798 < 0.40$ (`confidence_threshold`) $\rightarrow$ False.
    - $0.0441 < 0.10$ (`gap_threshold`) $\rightarrow$ False.
    - `agree == False` $\rightarrow$ False.
  - *Case 6 (Default Fallback)*: Falls through all criteria $\rightarrow$ `action = "draft"`.
  - **Resulting Action**: `action = "draft"`.
  - **Correctness**: Because `is_none = True`, correctness requires `action in ("no_action", "no action")`. Since action is `"draft"`, `correct = false`. This row is flagged as a **false match** and highlighted in the NONE Panel.

---

## 14. Known Gaps

### NOT FOUND / UNSURE Items
1. `SearchLog`: **NOT FOUND**. The `SearchLog` table does not exist anywhere in the `siglip` service codebase. Database schema verification in [`app/db/schema_verify.py:20-37`](file:///c:/Users/Win11/code/shamixo/siglip/app/db/schema_verify.py#L20-L37) verifies only `WoService`, `ServiceImageProfile`, and `DecisionLog`. No repository method or query writes to a `SearchLog` table.

### Separately Duplicated Logic
1. **Idempotency Replay State Construction**: In [`app/routers/posts.py:129-158`](file:///c:/Users/Win11/code/shamixo/siglip/app/routers/posts.py#L129-L158), the replay response reconstructs a `ProcessedPost` and computes the shadow flag via `existing["would_be_action"] == "auto_create" and existing["action"] != "auto_create"` directly from database columns instead of delegating to `make_decision()`.
2. **Bi-Encoder vs Cross-Encoder Scoring Bifurcation**: When `ENABLE_RERANKER=True`, scoring is delegated to `tei_client.rerank()` and `apply_reranker_softmax()` ([`app/pipeline/engine.py:174-186`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L174-L186)), but when `ENABLE_RERANKER=False`, an inline adjusted scoring formula with agree boosts and temperature softmax is implemented separately in [`app/pipeline/engine.py:209-262`](file:///c:/Users/Win11/code/shamixo/siglip/app/pipeline/engine.py#L209-L262).
3. **NONE Row Correctness Evaluation**: The check for whether a NONE row was correctly answered as no-action is coded in [`app/eval/job_manager.py:432`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/job_manager.py#L432) (`correct = (action in ("no_action", "no action"))`) and re-evaluated independently in [`app/eval/metrics.py:256`](file:///c:/Users/Win11/code/shamixo/siglip/app/eval/metrics.py#L256) (`none_correct_rows = [r for r in none_rows if r.action in ("no_action", "no action")]`).

---

## 15. Config Table

All settings affecting processing and evaluation results:

| Setting Name | Environment Variable | Default Value | Code Definition | Value in `.env` |
|---|---|---|---|---|
| `PIPELINE_TEXT_DESC_WEIGHT` | `PIPELINE_TEXT_DESC_WEIGHT` | `0.7` | [`app/config.py:63`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L63) | *Not defined (uses 0.7)* |
| `PIPELINE_TEXT_TOP_N` | `PIPELINE_TEXT_TOP_N` | `10` | [`app/config.py:64`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L64) | *Not defined (uses 10)* |
| `PIPELINE_IMAGE_TOP_N` | `PIPELINE_IMAGE_TOP_N` | `5` | [`app/config.py:65`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L65) | *Not defined (uses 5)* |
| `PIPELINE_POOL_MAX` | `PIPELINE_POOL_MAX` | `10` | [`app/config.py:66`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L66) | *Not defined (uses 10)* |
| `PIPELINE_RERANKER_TEMP` | `PIPELINE_RERANKER_TEMP` | `1.0` | [`app/config.py:67`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L67) | *Not defined (uses 1.0)* |
| `ENABLE_RERANKER` | `ENABLE_RERANKER` | `True` | [`app/config.py:68`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L68) | *Not defined (uses True)* |
| `PIPELINE_BIENCODER_AGREE_BOOST` | `PIPELINE_BIENCODER_AGREE_BOOST` | `0.05` | [`app/config.py:69`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L69) | *Not defined (uses 0.05)* |
| `PIPELINE_BIENCODER_TEMP` | `PIPELINE_BIENCODER_TEMP` | `0.06` | [`app/config.py:70`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L70) | *Not defined (uses 0.06)* |
| `AUTO_CREATE_ENABLED` | `AUTO_CREATE_ENABLED` | `False` | [`app/config.py:72`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L72) | `true` |
| `DECISION_CONFIDENCE_THRESHOLD` | `DECISION_CONFIDENCE_THRESHOLD` | `0.40` | [`app/config.py:73`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L73) | *Not defined (uses 0.40)* |
| `DECISION_GAP_THRESHOLD` | `DECISION_GAP_THRESHOLD` | `0.10` | [`app/config.py:74`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L74) | *Not defined (uses 0.10)* |
| `DECISION_NO_ACTION_FLOOR` | `DECISION_NO_ACTION_FLOOR` | `0.25` | [`app/config.py:75`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L75) | *Not defined (uses 0.25)* |
| `DECISION_REQUIRE_AGREE` | `DECISION_REQUIRE_AGREE` | `False` | [`app/config.py:76`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L76) | `true` |
| `MIN_DESCRIPTION_WORDS` | `MIN_DESCRIPTION_WORDS` | `3` | [`app/config.py:77`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L77) | *Not defined (uses 3)* |
| `ALWAYS_DRAFT_SERVICES` | `ALWAYS_DRAFT_SERVICES` | `'["Emergency Response & Area Securing", "Emergency Reactive Cleaning (Spills, Vomit, Incidents)"]'` | [`app/config.py:78`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L78) | *Not defined* |
| `SIGLIP_MODEL_ID` | `SIGLIP_MODEL_ID` | `"google/siglip2-base-patch16-224"` | [`app/config.py:53`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L53) | `"google/siglip2-base-patch16-224"` |
| `SIGLIP_TORCH_THREADS` | `SIGLIP_TORCH_THREADS` | `None` | [`app/config.py:54`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L54) | *Not defined* |
| `SIGLIP_LRU_CACHE_SIZE` | `SIGLIP_LRU_CACHE_SIZE` | `256` | [`app/config.py:55`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L55) | *Not defined (uses 256)* |
| `TEI_EMBED_URL` | `TEI_EMBED_URL` | `"http://tei-embed:80"` | [`app/config.py:46`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L46) | `"http://localhost:8081"` |
| `TEI_RERANK_URL` | `TEI_RERANK_URL` | `"http://tei-rerank:80"` | [`app/config.py:47`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L47) | `"http://localhost:8082"` |
| `TEI_API_KEY` | `TEI_API_KEY` | `None` | [`app/config.py:48`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L48) | `"test"` |
| `MAX_UPLOAD_BYTES` | `MAX_UPLOAD_BYTES` | `10485760` (10 MB) | [`app/config.py:37`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L37) | *Not defined* |
| `ENABLE_EVAL` | `ENABLE_EVAL` | `False` | [`app/config.py:82`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L82) | `true` |
| `EVAL_RUNS_DIR` | `EVAL_RUNS_DIR` | `"app/eval/runs"` | [`app/config.py:83`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L83) | *Not defined* |
| `EVAL_MAX_HISTORY_RUNS` | `EVAL_MAX_HISTORY_RUNS` | `50` | [`app/config.py:84`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L84) | *Not defined (uses 50)* |
| `EVAL_MAX_ZIP_BYTES` | `EVAL_MAX_ZIP_BYTES` | `104857600` (100 MB) | [`app/config.py:85`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L85) | *Not defined* |
| `EVAL_MAX_IMAGE_COUNT` | `EVAL_MAX_IMAGE_COUNT` | `1000` | [`app/config.py:86`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L86) | *Not defined* |
| `EVAL_MAX_IMAGE_BYTES` | `EVAL_MAX_IMAGE_BYTES` | `10485760` (10 MB) | [`app/config.py:87`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L87) | *Not defined* |
| `EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES` | `EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES` | `524288000` (500 MB) | [`app/config.py:88`](file:///c:/Users/Win11/code/shamixo/siglip/app/config.py#L88) | *Not defined* |

