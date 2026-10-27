"""
Candidate pool merger — ported 1:1 from photoToWorkOrder/ptwo/pool.py.
Combines text top-N and image top-N into a bounded candidate pool.
"""

from __future__ import annotations

from typing import Any


def build_candidate_pool(
    text_top: list[dict[str, Any]] | None,
    image_top: list[dict[str, Any]] | None,
    pool_max: int = 10,
) -> list[dict[str, Any]]:
    """
    Union of text top-N and image top-N, sorted by min(text_rank, image_rank)
    with ties broken by text_rank then image_rank then service name.
    Capped at pool_max. Matches photoToWorkOrder's pool.py exactly.
    """
    text_list = text_top or []
    image_list = image_top or []

    candidates_by_name: dict[str, dict[str, Any]] = {}

    for item in text_list:
        svc = item["service"]
        candidates_by_name[svc] = {
            "service": svc,
            "service_id": item.get("service_id"),
            "description": item.get("description", ""),
            "text_rank": float(item["rank"]),
            "image_rank": float("inf"),
            "text_score": float(item.get("score", 0.0)),
            "image_score": 0.0,
        }

    for item in image_list:
        svc = item["service"]
        if svc in candidates_by_name:
            candidates_by_name[svc]["image_rank"] = float(item["rank"])
            candidates_by_name[svc]["image_score"] = float(item.get("score", 0.0))
            if not candidates_by_name[svc]["description"] and item.get("description"):
                candidates_by_name[svc]["description"] = item["description"]
            if candidates_by_name[svc]["service_id"] is None and item.get("service_id"):
                candidates_by_name[svc]["service_id"] = item["service_id"]
        else:
            candidates_by_name[svc] = {
                "service": svc,
                "service_id": item.get("service_id"),
                "description": item.get("description", ""),
                "text_rank": float("inf"),
                "image_rank": float(item["rank"]),
                "text_score": 0.0,
                "image_score": float(item.get("score", 0.0)),
            }

    pool: list[dict[str, Any]] = []
    for svc, meta in candidates_by_name.items():
        t_r = meta["text_rank"]
        i_r = meta["image_rank"]
        meta["min_rank"] = min(t_r, i_r)
        pool.append(meta)

    pool.sort(key=lambda x: (x["min_rank"], x["text_rank"], x["image_rank"], x["service"]))

    capped = pool[:pool_max]
    for idx, item in enumerate(capped, start=1):
        item["pool_rank"] = idx
        if item["text_rank"] == float("inf"):
            item["text_rank"] = None
        if item["image_rank"] == float("inf"):
            item["image_rank"] = None
        if item["min_rank"] == float("inf"):
            item["min_rank"] = None

    return capped

