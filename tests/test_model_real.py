"""
Optional real-model test suite for SigLIP 2.
Executes only when SIGLIP_RUN_MODEL_TESTS=1.
"""

import os
from PIL import Image, ImageDraw
import pytest

from app.model import SigLIPService

RUN_MODEL_TESTS = os.getenv("SIGLIP_RUN_MODEL_TESTS") == "1"


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="Real model test requires SIGLIP_RUN_MODEL_TESTS=1")
def test_real_siglip_model_inference_and_ranking():
    """
    Loads real SigLIP 2 model once locally:
    - Embeds one image and 3 short texts (one matching, two not)
    - Confirms embedding dimension is 768
    - Confirms matching text ranks first (#1)
    """
    service = SigLIPService(
        model_id="google/siglip2-base-patch16-224",
        torch_threads=2,
        load_now=True,
    )
    assert service.is_loaded is True
    assert service.embedding_dim == 768

    # Create a clear visual image: a bright red emergency stop button / red circle
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 184, 184], fill="red", outline="darkred")

    candidates = [
        {"id": "matching", "text": "a bright red emergency circle button on a white background"},
        {"id": "non_matching_1", "text": "a dark blue deep ocean with swimming dolphins"},
        {"id": "non_matching_2", "text": "a bowl of green salad with cucumber and lettuce"},
    ]

    results = service.rank(img, candidates)
    assert len(results) == 3

    # Check top candidate
    top_candidate = results[0]
    assert top_candidate["id"] == "matching"
    assert top_candidate["rank"] == 1
    assert top_candidate["score"] > results[1]["score"]
    assert top_candidate["score"] > results[2]["score"]

    # Verify text embedding dimension
    embs, truncated = service.encode_texts(["short text"])
    assert len(embs[0]) == 768
    assert truncated[0] is False

