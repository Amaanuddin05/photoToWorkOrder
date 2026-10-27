"""
Zero-shot image ranking against cached ServiceImageProfile vectors.
Ported 1:1 from photoToWorkOrder/ptwo/image_rank.py (BaseTorchImageRanker).
Uses the in-process SigLIPService for image and text encoding.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("siglip.pipeline.image_rank")


def rank_image_against_profiles(
    image: Image.Image,
    service_profiles: dict[int, dict[str, Any]],
    siglip_service: Any,  # SigLIPService
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Synchronous zero-shot image ranking.
    Encodes image via SigLIP, cosine-dots against cached service text vectors.
    Middle-rank rule: services with no visualLine are never added to pool.

    Matches photoToWorkOrder's BaseTorchImageRanker.rank() exactly.
    """
    # Encode image → normalized 1D tensor
    img_embs = siglip_service.encode_images([image])
    img_vec = np.array(img_embs[0], dtype=np.float32)
    img_tensor = torch.from_numpy(img_vec)

    total_services = len(service_profiles)
    middle_rank = (total_services + 1) / 2.0  # 10.0 for 19 services

    visible_scores: list[tuple[int, str, str, float]] = []
    non_visual_services: list[tuple[int, str, str]] = []

    for svc_id, profile in service_profiles.items():
        vec = profile.get("vector")
        title = profile.get("title", "")
        desc = profile.get("description", "")
        if vec is not None and profile.get("visual_line"):
            sim = float(torch.dot(img_tensor, vec).item())
            visible_scores.append((svc_id, title, desc, sim))
        else:
            non_visual_services.append((svc_id, title, desc))

    visible_scores.sort(key=lambda x: x[3], reverse=True)

    cutoff_top = int(middle_rank) - 1
    ranked_all: list[dict[str, Any]] = []
    current_rank = 1

    for svc_id, title, desc, sim in visible_scores[:cutoff_top]:
        ranked_all.append({
            "service": title,
            "service_id": svc_id,
            "description": desc,
            "rank": float(current_rank),
            "score": sim,
            "has_visual_vote": True,
        })
        current_rank += 1

    for svc_id, title, desc in non_visual_services:
        ranked_all.append({
            "service": title,
            "service_id": svc_id,
            "description": desc,
            "rank": middle_rank,
            "score": 0.0,
            "has_visual_vote": False,
        })

    current_rank = int(middle_rank) + len(non_visual_services)
    for svc_id, title, desc, sim in visible_scores[cutoff_top:]:
        ranked_all.append({
            "service": title,
            "service_id": svc_id,
            "description": desc,
            "rank": float(current_rank),
            "score": sim,
            "has_visual_vote": True,
        })
        current_rank += 1

    ranked_all.sort(key=lambda x: (x["rank"], -x["score"]))
    valid = [item for item in ranked_all if item["has_visual_vote"]]
    return valid[:top_n]

