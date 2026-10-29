"""
Smoke test script for multimodal microservices.
Executes and prints side-by-side results from:
1. TEI Text Embedder: POST {TEI_EMBED_URL}/embed
2. TEI Text Reranker: POST {TEI_RERANK_URL}/rerank
3. SigLIP Image Ranker: POST {SIGLIP_URL}/rank

Note: Does NOT implement fusion or decision logic (which belongs to photoToWorkOrder).
"""

import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

SIGLIP_DIR = Path(__file__).resolve().parent.parent
if str(SIGLIP_DIR) not in sys.path:
    sys.path.insert(0, str(SIGLIP_DIR))

import httpx
from PIL import Image

DEFAULT_CANDIDATES = [
    {"id": 1, "name": "AC Services", "text": "air conditioner unit on wall or ceiling, cooling vents, thermostat, or AC unit leaking water"},
    {"id": 2, "name": "Washroom Deep Cleaning", "text": "dirty toilet bowl, stained urinal, or washroom sink fixtures needing scrubbing"},
    {"id": 3, "name": "Waste Management Service", "text": "overflowing garbage dumpster, trash bins, or waste collection area"},
    {"id": 4, "name": "Handyman Service (Minor Repairs)", "text": "broken door hinge, damaged wall fixture, loose handle, or minor physical repair"},
    {"id": 5, "name": "Window Cleaning Service", "text": "dirty window glass panes with smudges, streaks, or water stains"},
]


async def run_smoke_test(
    image_path: Optional[str],
    description: str,
    candidates: List[Dict[str, Any]],
    siglip_url: str,
    tei_embed_url: str,
    tei_rerank_url: str,
    siglip_key: Optional[str] = None,
    tei_key: Optional[str] = None,
) -> None:
    print("=" * 80)
    print("MULTIMODAL MICROSERVICES SMOKE TEST")
    print("=" * 80)
    print(f"Description:    '{description}'")
    print(f"Image Path:     {image_path if image_path else '(synthetic sample image)'}")
    print(f"Candidates ({len(candidates)}):")
    for c in candidates:
        print(f"  [{c['id']}] {c['name']} -> {c['text'][:60]}...")
    print("-" * 80)

    # Prepare Image bytes
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            filename = Path(image_path).name
    else:
        # Create a small valid test JPEG
        buf = io.BytesIO()
        img = Image.new("RGB", (224, 224), color=(70, 130, 180))
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        filename = "synthetic_sample.jpg"

    candidate_texts = [c["text"] for c in candidates]
    candidate_map = {c["id"]: c for c in candidates}

    # 1. Call TEI Embed
    print(f"\n[1/3] Calling TEI Embedder ({tei_embed_url}/embed)...")
    tei_embed_result = None
    tei_headers = {"Content-Type": "application/json"}
    if tei_key:
        tei_headers["Authorization"] = f"Bearer {tei_key}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{tei_embed_url.rstrip('/')}/embed",
                json={"inputs": [description]},
                headers=tei_headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                vec = data[0] if isinstance(data, list) and len(data) > 0 else data
                tei_embed_result = vec
                print(f"      -> SUCCESS: Received embedding vector of dimension {len(vec)}.")
                print(f"      -> Sample values (first 5): {[round(x, 4) for x in vec[:5]]}")
            else:
                print(f"      -> FAILED: HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            print(f"      -> ERROR connecting to TEI Embedder: {exc}")

    # 2. Call TEI Rerank
    print(f"\n[2/3] Calling TEI Reranker ({tei_rerank_url}/rerank)...")
    tei_rerank_map: Dict[int, Dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{tei_rerank_url.rstrip('/')}/rerank",
                json={"query": description, "texts": candidate_texts},
                headers=tei_headers,
            )
            if resp.status_code == 200:
                rerank_data = resp.json()
                print(f"      -> SUCCESS: Received scores for {len(rerank_data)} candidates.")
                # Rerank items typically have {"index": int, "score": float}
                for r_idx, item in enumerate(rerank_data, start=1):
                    cand_idx = item.get("index")
                    if cand_idx is not None and cand_idx < len(candidates):
                        cand_id = candidates[cand_idx]["id"]
                        tei_rerank_map[cand_id] = {
                            "score": float(item.get("score", 0.0)),
                            "rank": r_idx,
                        }
            else:
                print(f"      -> FAILED: HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            print(f"      -> ERROR connecting to TEI Reranker: {exc}")

    # 3. Call SigLIP Rank
    print(f"\n[3/3] Calling SigLIP Image Ranker ({siglip_url}/rank)...")
    siglip_rank_map: Dict[Any, Dict[str, Any]] = {}
    siglip_headers = {}
    if siglip_key:
        siglip_headers["Authorization"] = f"Bearer {siglip_key}"

    siglip_candidates_payload = [
        {"id": c["id"], "text": c["text"]}
        for c in candidates
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            data = {"candidates": json.dumps(siglip_candidates_payload)}
            resp = await client.post(
                f"{siglip_url.rstrip('/')}/rank",
                files=files,
                data=data,
                headers=siglip_headers,
            )
            if resp.status_code == 200:
                rank_data = resp.json()
                print(f"      -> SUCCESS: Received SigLIP cosine rankings for {len(rank_data)} candidates.")
                for item in rank_data:
                    siglip_rank_map[item["id"]] = {
                        "score": float(item["score"]),
                        "rank": int(item["rank"]),
                    }
            else:
                print(f"      -> FAILED: HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            print(f"      -> ERROR connecting to SigLIP service: {exc}")

    # 4. Side-by-Side Comparison Display
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE CANDIDATE RESULTS")
    print("=" * 80)
    header = f"{'Candidate Name':<30} | {'TEI Rerank Score (Rank)':<24} | {'SigLIP Cosine Score (Rank)':<24}"
    print(header)
    print("-" * len(header))

    for c in candidates:
        cid = c["id"]
        cname = c["name"][:28]
        tei_info = tei_rerank_map.get(cid)
        if tei_info:
            tei_str = f"{tei_info['score']:8.4f} (#{tei_info['rank']})"
        else:
            tei_str = "N/A"

        sig_info = siglip_rank_map.get(cid)
        if sig_info:
            sig_str = f"{sig_info['score']:8.4f} (#{sig_info['rank']})"
        else:
            sig_str = "N/A"

        print(f"{cname:<30} | {tei_str:<24} | {sig_str:<24}")

    print("=" * 80)
    print("Note: Upstream pipeline fusion and decision logic reside inside 'photoToWorkOrder'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal services smoke test.")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (JPEG/PNG)")
    parser.add_argument("--description", type=str, default="Dirty toilet and stained urinal needing cleaning", help="Work order text description")
    parser.add_argument("--siglip-url", type=str, default=os.getenv("SIGLIP_URL", "http://localhost:8000"))
    parser.add_argument("--tei-embed-url", type=str, default=os.getenv("TEI_EMBED_URL", "http://localhost:8081"))
    parser.add_argument("--tei-rerank-url", type=str, default=os.getenv("TEI_RERANK_URL", "http://localhost:8082"))
    parser.add_argument("--siglip-key", type=str, default=os.getenv("SIGLIP_API_KEY", ""))
    parser.add_argument("--tei-key", type=str, default=os.getenv("TEI_API_KEY", ""))

    args = parser.parse_args()

    # If no image specified, look for an existing sample photo
    image_path = args.image
    if not image_path:
        candidate_paths = [
            "photos/wo_18.jpg",
            "../photos/wo_18.jpg",
            "photos/wo_13.jpg",
            "../photos/wo_13.jpg",
        ]
        for p in candidate_paths:
            if Path(p).exists():
                image_path = str(Path(p).resolve())
                break

    asyncio.run(
        run_smoke_test(
            image_path=image_path,
            description=args.description,
            candidates=DEFAULT_CANDIDATES,
            siglip_url=args.siglip_url,
            tei_embed_url=args.tei_embed_url,
            tei_rerank_url=args.tei_rerank_url,
            siglip_key=args.siglip_key or None,
            tei_key=args.tei_key or None,
        )
    )


if __name__ == "__main__":
    main()

