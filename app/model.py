"""
SigLIP 2 Model wrapper and LRU text embedding cache.
Uses transformers AutoModel and AutoProcessor on CPU in inference mode.
"""

from collections import OrderedDict
from contextlib import contextmanager
import gc
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("siglip.model")


class TextEmbeddingLRUCache:
    """Thread-safe LRU cache for candidate text embeddings."""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = max(1, maxsize)
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: List[float]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class SigLIPService:
    """SigLIP 2 model service wrapper handling image/text embeddings and zero-shot ranking."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        torch_threads: Optional[int] = None,
        cache_size: int = 256,
        load_now: bool = True,
    ) -> None:
        self.model_id = model_id
        self.torch_threads = torch_threads
        self.lru_cache = TextEmbeddingLRUCache(maxsize=cache_size)
        self.model = None
        self.processor = None
        self.embedding_dim: Optional[int] = None
        self._is_loaded = False
        # On-demand loading: the weights are (re)loaded by the first call that needs them and
        # dropped by unload_if_idle(). The RLock serialises load/unload; _inflight blocks an
        # unload while any encode/rank call is running.
        self._lock = threading.RLock()
        self._inflight = 0
        self._last_used = time.monotonic()
        self._loading = False
        self.last_load_error: Optional[str] = None
        self.load_count = 0

        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(torch_threads)
            logger.info("Configured PyTorch CPU threads: %d", torch_threads)

        if load_now:
            self.load()

    def load(self) -> None:
        """Loads model and processor on CPU (no-op when already loaded)."""
        with self._lock:
            if self._is_loaded:
                return
            self._loading = True
            t0 = time.perf_counter()
            try:
                self._load_weights()
                self.last_load_error = None
            except Exception as exc:
                self.last_load_error = str(exc)
                raise
            finally:
                self._loading = False
            self.load_count += 1
            self._last_used = time.monotonic()
            logger.info("SigLIP model loaded in %.1fs (load #%d)", time.perf_counter() - t0, self.load_count)

    def unload(self) -> bool:
        """Drop the weights from RAM. Refuses (returns False) while a call is running."""
        with self._lock:
            if not self._is_loaded or self._inflight > 0:
                return False
            self.model = None
            self.processor = None
            self.lru_cache.clear()
            self._is_loaded = False
        gc.collect()
        logger.info("SigLIP model unloaded (idle)")
        return True

    def unload_if_idle(self, idle_seconds: float) -> bool:
        """Unload when no call has used the model for idle_seconds (0 or less = never)."""
        if idle_seconds <= 0:
            return False
        with self._lock:
            if not self._is_loaded or self._inflight > 0:
                return False
            if time.monotonic() - self._last_used < idle_seconds:
                return False
        return self.unload()

    @property
    def state(self) -> str:
        if self._loading:
            return "loading"
        return "loaded" if self._is_loaded else "unloaded"

    @property
    def idle_seconds(self) -> float:
        return round(time.monotonic() - self._last_used, 1)

    @contextmanager
    def _use(self):
        """Mark a call in flight and make sure the weights are loaded for it."""
        with self._lock:
            self._inflight += 1
            try:
                if not self._is_loaded:
                    self.load()
            except Exception:
                self._inflight -= 1
                raise
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1
                self._last_used = time.monotonic()

    def _load_weights(self) -> None:
        logger.info("Loading SigLIP model from %s on CPU...", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to("cpu").eval()

        # Determine embedding dimension
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            self.embedding_dim = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, "hidden_size"):
            self.embedding_dim = self.model.config.hidden_size
        else:
            self.embedding_dim = 768

        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def encode_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[bool]]:
        """
        Encodes list of texts into L2-normalized float embeddings with max 64 tokens.
        Returns: (embeddings, truncated_flags)
        """
        with self._use():
            return self._encode_texts(texts)

    def _encode_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[bool]]:
        if not texts:
            return [], []

        # Check truncation for each text
        truncated_flags: List[bool] = []
        for text in texts:
            tokens = self.processor.tokenizer(text, truncation=False)["input_ids"]
            truncated_flags.append(len(tokens) > 64)

        inputs = self.processor(
            text=texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            out = self.model.get_text_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else (out[0] if isinstance(out, tuple) else out)
            feat = F.normalize(feat, p=2, dim=-1)
            embeddings = feat.cpu().tolist()

        return embeddings, truncated_flags

    def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        """Encodes list of PIL images into L2-normalized float embeddings."""
        with self._use():
            return self._encode_images(images)

    def _encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        if not images:
            return []

        inputs = self.processor(images=images, return_tensors="pt")

        with torch.inference_mode():
            out = self.model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else (out[0] if isinstance(out, tuple) else out)
            feat = F.normalize(feat, p=2, dim=-1)
            embeddings = feat.cpu().tolist()

        return embeddings

    def rank(self, image: Image.Image, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks candidates by cosine similarity against the image embedding.
        Uses LRU cache for candidate text embeddings.
        Returns candidates sorted by score descending: [{"id": id, "score": float, "rank": int}]
        """
        with self._use():
            return self._rank(image, candidates)

    def _rank(self, image: Image.Image, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # 1. Fetch cached text embeddings or compute missing ones
        missing_texts: List[str] = []
        for cand in candidates:
            txt = cand["text"]
            if self.lru_cache.get(txt) is None and txt not in missing_texts:
                missing_texts.append(txt)

        if missing_texts:
            new_embs, _ = self.encode_texts(missing_texts)
            for txt, emb in zip(missing_texts, new_embs):
                self.lru_cache.put(txt, emb)

        # 2. Encode image
        img_embs = self.encode_images([image])
        img_vec = np.array(img_embs[0], dtype=np.float32)

        # 3. Compute cosine similarity (dot product of L2-normalized vectors)
        results: List[Dict[str, Any]] = []
        for cand in candidates:
            txt_emb = self.lru_cache.get(cand["text"])
            txt_vec = np.array(txt_emb, dtype=np.float32)
            score = float(np.dot(img_vec, txt_vec))
            results.append({"id": cand["id"], "score": score})

        # 4. Sort descending by raw cosine score
        results.sort(key=lambda x: x["score"], reverse=True)

        # 5. Assign rank 1..N
        for idx, item in enumerate(results, start=1):
            item["rank"] = idx

        return results

