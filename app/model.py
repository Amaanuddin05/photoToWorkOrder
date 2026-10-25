"""
SigLIP 2 Model wrapper and LRU text embedding cache.
Uses transformers AutoModel and AutoProcessor on CPU in inference mode.
"""

from collections import OrderedDict
import logging
import threading
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

        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(torch_threads)
            logger.info("Configured PyTorch CPU threads: %d", torch_threads)

        if load_now:
            self.load()

    def load(self) -> None:
        """Loads model and processor once on CPU."""
        if self._is_loaded:
            return

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
        logger.info("SigLIP model loaded successfully (dim=%d)", self.embedding_dim)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def encode_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[bool]]:
        """
        Encodes list of texts into L2-normalized float embeddings with max 64 tokens.
        Returns: (embeddings, truncated_flags)
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded.")

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

    def rank(self, image: Image.Image, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks candidates by cosine similarity against the image embedding.
        Uses LRU cache for candidate text embeddings.
        Returns candidates sorted by score descending: [{"id": id, "score": float, "rank": int}]
        """
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

