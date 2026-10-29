"""
Pytest configuration and fixtures for offline testing.
Provides mock models and test client without downloading weights.
"""

import hashlib
import io
import math
from typing import Any, Dict, List, Optional, Tuple
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.config import Settings
from app.main import create_app
from app.model import SigLIPService, TextEmbeddingLRUCache
from app.tei_client import TEIClient


class FakeSigLIPService:
    """Offline fake SigLIP service producing deterministic 768-dim L2-normalized embeddings."""

    def __init__(self, cache_size: int = 256) -> None:
        self.model_id = "fake-siglip2-base"
        self.embedding_dim = 768
        self._is_loaded = True
        self.lru_cache = TextEmbeddingLRUCache(maxsize=cache_size)
        self.encode_text_calls = 0
        self.encode_image_calls = 0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _generate_vector(self, seed_str: str) -> List[float]:
        # Generate deterministic pseudo-random float vector from hash
        h = hashlib.sha256(seed_str.encode("utf-8")).digest()
        # Create 768 floats
        vec = []
        for i in range(self.embedding_dim):
            byte_val = h[i % len(h)]
            vec.append(float((byte_val + i) % 100) / 100.0 - 0.5)

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def encode_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[bool]]:
        self.encode_text_calls += 1
        embeddings: List[List[float]] = []
        truncated: List[bool] = []
        for t in texts:
            # Simulate 64 token limit: words > 64 considered truncated
            is_trunc = len(t.split()) > 64
            truncated.append(is_trunc)
            embeddings.append(self._generate_vector(f"text:{t}"))
        return embeddings, truncated

    def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        self.encode_image_calls += 1
        embeddings: List[List[float]] = []
        for img in images:
            # Use image size and dominant color as seed
            seed = f"img:{img.size}:{img.mode}"
            embeddings.append(self._generate_vector(seed))
        return embeddings

    def rank(self, image: Image.Image, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Compute or retrieve candidate text embeddings
        missing = [c["text"] for c in candidates if self.lru_cache.get(c["text"]) is None]
        if missing:
            embs, _ = self.encode_texts(missing)
            for txt, emb in zip(missing, embs):
                self.lru_cache.put(txt, emb)

        img_emb = self.encode_images([image])[0]
        img_arr = np.array(img_emb, dtype=np.float32)

        results = []
        for c in candidates:
            t_emb = self.lru_cache.get(c["text"])
            t_arr = np.array(t_emb, dtype=np.float32)
            score = float(np.dot(img_arr, t_arr))
            results.append({"id": c["id"], "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(results, start=1):
            item["rank"] = idx
        return results


@pytest.fixture
def test_settings() -> Settings:
    settings = Settings()
    settings.SIGLIP_MODEL_ID = "fake-siglip2-base"
    settings.SIGLIP_API_KEY = "test-secret-key"
    settings.SIGLIP_MAX_CONCURRENCY = 2
    settings.SIGLIP_MAX_UPLOAD_BYTES = 50 * 1024  # 50 KB for test limits
    settings.SIGLIP_MAX_IMAGES_PER_REQUEST = 3
    settings.SIGLIP_MAX_CANDIDATES_PER_REQUEST = 10
    settings.SIGLIP_LRU_CACHE_SIZE = 16
    return settings


@pytest.fixture
def fake_siglip() -> FakeSigLIPService:
    return FakeSigLIPService(cache_size=16)


@pytest.fixture
def fake_tei() -> TEIClient:
    return TEIClient(
        embed_url="http://tei-embed-test:80",
        rerank_url="http://tei-rerank-test:80",
        api_key="tei-test-key",
    )


@pytest.fixture
def app(test_settings: Settings, fake_siglip: FakeSigLIPService, fake_tei: TEIClient):
    return create_app(
        settings=test_settings,
        siglip_service=fake_siglip,
        tei_client=fake_tei,
    )


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer test-secret-key"}


@pytest.fixture
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(buf, format="JPEG")
    return buf.getvalue()

