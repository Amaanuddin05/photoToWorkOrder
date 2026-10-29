"""
Offline pytest suite for SigLIP microservice endpoints, auth, and validation.
"""

import io
import json
import math
from typing import Dict
from PIL import Image
import pytest


def test_health_no_auth(client):
    """GET /health must be accessible without auth and return status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_id"] == "fake-siglip2-base"
    assert data["loaded"] is True
    assert data["embedding_dim"] == 768


def test_health_deps_no_auth(client):
    """GET /health/deps must be accessible without auth and return dependency status."""
    resp = client.get("/health/deps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "dependencies" in data
    assert "tei_embed" in data["dependencies"]
    assert "tei_rerank" in data["dependencies"]


def test_auth_enforcement(client, sample_image_bytes):
    """Protected endpoints must return 401 Unauthorized when auth is missing or invalid."""
    # 1. No auth
    r1 = client.post("/embed-text", json={"texts": ["hello"]})
    assert r1.status_code == 401

    r2 = client.post(
        "/embed-image",
        files=[("files", ("test.jpg", sample_image_bytes, "image/jpeg"))],
    )
    assert r2.status_code == 401

    r3 = client.post(
        "/rank",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"candidates": json.dumps([{"id": 1, "text": "candidate 1"}])},
    )
    assert r3.status_code == 401

    # 2. Invalid auth token
    bad_headers = {"Authorization": "Bearer wrong-token"}
    r_bad = client.post("/embed-text", json={"texts": ["hello"]}, headers=bad_headers)
    assert r_bad.status_code == 401

    # 3. Valid auth token
    good_headers = {"Authorization": "Bearer test-secret-key"}
    r_good = client.post("/embed-text", json={"texts": ["hello"]}, headers=good_headers)
    assert r_good.status_code == 200


def test_embed_text_vectors_and_normalization(client, auth_headers):
    """POST /embed-text must return L2-normalized vectors of expected dimension."""
    texts = ["Air conditioner repair", "Water leak in basement"]
    resp = client.post("/embed-text", json={"texts": texts}, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["dim"] == 768
    assert len(data["embeddings"]) == 2
    assert len(data["truncated"]) == 2

    # Check L2-normalization for each vector
    for emb in data["embeddings"]:
        assert len(emb) == 768
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-4


def test_embed_text_truncation_flag(client, auth_headers):
    """POST /embed-text reports which inputs were truncated."""
    short_text = "Clean the classroom floor"
    long_text = "word " * 80  # 80 words > 64 limit

    resp = client.post(
        "/embed-text",
        json={"texts": [short_text, long_text]},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] == [False, True]


def test_embed_text_max_candidates_limit(client, auth_headers):
    """POST /embed-text rejects requests exceeding candidate limit."""
    too_many = [f"Text {i}" for i in range(15)]  # Limit in test_settings is 10
    resp = client.post("/embed-text", json={"texts": too_many}, headers=auth_headers)
    assert resp.status_code == 400
    assert "exceeds max limit" in resp.json()["detail"]


def test_embed_image(client, auth_headers, sample_image_bytes):
    """POST /embed-image must return L2-normalized image embeddings."""
    files = [
        ("files", ("img1.jpg", sample_image_bytes, "image/jpeg")),
        ("files", ("img2.jpg", sample_image_bytes, "image/jpeg")),
    ]
    resp = client.post("/embed-image", files=files, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["dim"] == 768
    assert len(data["embeddings"]) == 2
    for emb in data["embeddings"]:
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-4


def test_embed_image_corrupt_and_empty(client, auth_headers):
    """POST /embed-image rejects corrupt or empty files with 400 Bad Request."""
    # 1. Empty file
    r1 = client.post(
        "/embed-image",
        files=[("files", ("empty.jpg", b"", "image/jpeg"))],
        headers=auth_headers,
    )
    assert r1.status_code == 400
    assert "empty" in r1.json()["detail"]

    # 2. Corrupt bytes
    r2 = client.post(
        "/embed-image",
        files=[("files", ("corrupt.jpg", b"not-a-valid-image", "image/jpeg"))],
        headers=auth_headers,
    )
    assert r2.status_code == 400
    assert "Failed to decode image" in r2.json()["detail"]


def test_upload_size_limit(client, auth_headers):
    """POST /embed-image rejects files exceeding SIGLIP_MAX_UPLOAD_BYTES."""
    oversized = b"0" * (60 * 1024)  # 60 KB > 50 KB test limit
    resp = client.post(
        "/embed-image",
        files=[("files", ("big.jpg", oversized, "image/jpeg"))],
        headers=auth_headers,
    )
    assert resp.status_code == 413
    assert "exceeds maximum allowed" in resp.json()["detail"]


def test_upload_image_count_limit(client, auth_headers, sample_image_bytes):
    """POST /embed-image rejects requests with more images than allowed."""
    files = [
        ("files", (f"img_{i}.jpg", sample_image_bytes, "image/jpeg"))
        for i in range(4)  # Test limit is 3
    ]
    resp = client.post("/embed-image", files=files, headers=auth_headers)
    assert resp.status_code == 400
    assert "exceeds max limit" in resp.json()["detail"]


def test_rank_endpoint(client, auth_headers, sample_image_bytes):
    """POST /rank sorts candidates by cosine similarity and returns {id, score, rank}."""
    candidates = [
        {"id": "service_1", "text": "air conditioner cooling unit repair"},
        {"id": "service_2", "text": "toilet flush valve leaking"},
        {"id": "service_3", "text": "broken door lock"},
    ]
    resp = client.post(
        "/rank",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
        data={"candidates": json.dumps(candidates)},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3

    # Check structure and monotonic rank ordering
    for idx, item in enumerate(data, start=1):
        assert "id" in item
        assert "score" in item
        assert item["rank"] == idx

    # Check scores sorted descending
    scores = [item["score"] for item in data]
    assert scores == sorted(scores, reverse=True)


def test_rank_invalid_candidates_json(client, auth_headers, sample_image_bytes):
    """POST /rank returns 400 on malformed candidates JSON or missing keys."""
    # 1. Non-JSON string
    r1 = client.post(
        "/rank",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
        data={"candidates": "{invalid-json"},
        headers=auth_headers,
    )
    assert r1.status_code == 400

    # 2. Missing 'text' key
    r2 = client.post(
        "/rank",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
        data={"candidates": json.dumps([{"id": 1}])},
        headers=auth_headers,
    )
    assert r2.status_code == 400
    assert "must be an object with 'id' and 'text'" in r2.json()["detail"]


def test_rank_lru_cache(client, auth_headers, sample_image_bytes, fake_siglip):
    """Candidate text embeddings are cached across repeated /rank requests."""
    candidates = [
        {"id": 101, "text": "reusable visual candidate description"},
    ]
    assert len(fake_siglip.lru_cache) == 0
    assert fake_siglip.encode_text_calls == 0

    # Request 1: cache miss -> encodes text
    resp1 = client.post(
        "/rank",
        files={"file": ("photo1.jpg", sample_image_bytes, "image/jpeg")},
        data={"candidates": json.dumps(candidates)},
        headers=auth_headers,
    )
    assert resp1.status_code == 200
    assert len(fake_siglip.lru_cache) == 1
    assert fake_siglip.encode_text_calls == 1

    # Request 2: identical candidate -> cache hit, encode_text_calls should NOT increase
    resp2 = client.post(
        "/rank",
        files={"file": ("photo2.jpg", sample_image_bytes, "image/jpeg")},
        data={"candidates": json.dumps(candidates)},
        headers=auth_headers,
    )
    assert resp2.status_code == 200
    assert len(fake_siglip.lru_cache) == 1
    assert fake_siglip.encode_text_calls == 1  # Hit cache!

