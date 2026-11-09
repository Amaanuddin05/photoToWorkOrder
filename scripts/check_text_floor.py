"""
Text-only floor and guard analysis script (scripts/check_text_floor.py).
Strictly read-only, no image needed.

Evaluates:
- Group A: 20 unrelated messages (greetings, lunch, meeting times, personal items, jokes, etc.)
- Group B: 10 completion updates ("cleaned the corridor, done", "AC repaired, working now", etc.)

For each message:
- Embeds description using TEI embedder (plain text, no prefix).
- Queries pgvector for active service profiles using description weight 0.7.
- Evaluates decision using the pipeline's make_decision engine with the absolute text-score guard.
- Reports top-1 service, raw text_score, action, and reason.
- Computes and prints min / median / max of raw text_score per group separately.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Use SelectorEventLoop on Windows for psycopg async compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure app imports resolve
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings
from app.db import repository as repo
from app.db.connection import create_pool
from app.pipeline.decision import make_decision, softmax_with_temperature
from app.pipeline.pool import build_candidate_pool
from app.tei_client import TEIClient

# ── Test Datasets ─────────────────────────────────────────────────────────────

UNRELATED_MESSAGES: list[str] = [
    "good morning everyone",
    "hello team, hope you are all having a great week",
    "who is coming to lunch today at the cafeteria?",
    "reminder that the all-hands meeting starts at 3pm",
    "did anyone find a blue water bottle in the hallway?",
    "happy birthday Sarah, cake in the breakroom!",
    "why did the chicken cross the road? to get to the other side",
    "where did I leave my car keys, has anyone seen them?",
    "thanks for the update, see you tomorrow morning",
    "what time does the soccer match start tonight?",
    "can someone send me the meeting minutes from yesterday?",
    "left my umbrella near the reception desk",
    "traffic on the highway is terrible this morning",
    "anyone want to order coffee from downstairs?",
    "just checking in to say hi to the crew",
    "great presentation in the auditorium earlier today",
    "does anyone have a spare phone charger I can borrow?",
    "remember to submit your timesheets before Friday",
    "leaving early today for a dental appointment",
    "let's schedule a quick sync after lunch to discuss the weekend plans",
]

COMPLETION_UPDATES: list[str] = [
    "cleaned the corridor, done",
    "AC repaired, working now",
    "task completed, room is ready",
    "finished replacing the broken light bulbs in the hallway",
    "spill has been mopped up and floor is dry",
    "lock fixed and door tested, all good",
    "leak stopped, pipe has been sealed successfully",
    "garbage bins emptied and taken out to the dumpster",
    "fire extinguisher inspection completed for block B",
    "painting touch-up on the lobby wall is finished",
]


async def evaluate_descriptions(
    descriptions: list[str],
    pool: Any,
    tei_client: TEIClient,
    settings: Any,
) -> list[dict[str, Any]]:
    """
    Embeds each description, queries pgvector active services,
    and runs make_decision with the text guard enabled.
    """
    vectors = await tei_client.embed([d.strip() for d in descriptions])

    results: list[dict[str, Any]] = []

    for desc, query_vec in zip(descriptions, vectors):
        text_top = await repo.text_search(
            pool,
            query_vec,
            settings.PIPELINE_TEXT_DESC_WEIGHT,
            settings.PIPELINE_TEXT_TOP_N,
        )

        if not text_top:
            results.append({
                "description": desc,
                "top1_service": "NONE",
                "top1_text_score": 0.0,
                "confidence": 0.0,
                "gap": 0.0,
                "action": "no_action",
                "reason": "No matching services found",
            })
            continue

        cand_pool = build_candidate_pool(
            text_top=text_top,
            image_top=None,
            pool_max=settings.PIPELINE_POOL_MAX,
        )

        # Bi-encoder softmax distribution over raw scores
        raw_adjusted = [float(it.get("text_score") or 0.0) for it in cand_pool]
        soft_scores = softmax_with_temperature(
            raw_adjusted,
            temperature=settings.PIPELINE_BIENCODER_TEMP,
        )

        reranked = []
        for item, soft_conf, raw_s in zip(cand_pool, soft_scores, raw_adjusted):
            updated = dict(item)
            updated["rerank_score"] = round(raw_s, 4)
            updated["confidence"] = round(soft_conf, 4)
            reranked.append(updated)

        reranked.sort(key=lambda x: (-x["confidence"], -x["rerank_score"]))
        for idx, item in enumerate(reranked, start=1):
            item["rerank_rank"] = idx

        conf = float(reranked[0]["confidence"]) if reranked else 0.0
        second_conf = float(reranked[1]["confidence"]) if len(reranked) > 1 else 0.0
        gap = max(0.0, round(conf - second_conf, 4))

        # Evaluate decision engine with text guard
        decision = make_decision(
            description=desc,
            has_image=False,
            text_top=text_top,
            image_top=None,
            pool=cand_pool,
            reranked=reranked,
            confidence=conf,
            gap=gap,
            agree=None,
            auto_create_enabled=settings.AUTO_CREATE_ENABLED,
            confidence_threshold=settings.DECISION_CONFIDENCE_THRESHOLD,
            gap_threshold=settings.DECISION_GAP_THRESHOLD,
            no_action_floor=settings.DECISION_NO_ACTION_FLOOR,
            require_agree=settings.DECISION_REQUIRE_AGREE,
            always_draft_services=settings.always_draft_services_list,
            min_description_words=settings.MIN_DESCRIPTION_WORDS,
            enable_reranker=settings.ENABLE_RERANKER,
            text_guard_enabled=settings.DECISION_TEXT_GUARD_ENABLED,
            text_no_action_floor=settings.DECISION_TEXT_NO_ACTION_FLOOR,
            text_auto_min=settings.DECISION_TEXT_AUTO_MIN,
        )

        top1_item = reranked[0]
        s_val = top1_item.get("text_score")
        if s_val is None:
            s_val = text_top[0].get("score", 0.0)

        results.append({
            "description": desc,
            "top1_service": top1_item.get("service", text_top[0].get("service")),
            "top1_text_score": float(s_val),
            "confidence": conf,
            "gap": gap,
            "action": decision.action,
            "reason": decision.reason,
            "text_guard": getattr(decision, "text_guard", None),
        })

    return results


def print_group_results(group_name: str, items: list[dict[str, Any]]) -> dict[str, float]:
    print("\n" + "=" * 105)
    print(f" {group_name.upper()} (n={len(items)})")
    print("=" * 105)
    header = f"{'#':<3} | {'Description':<42} | {'Top-1 Predicted Service':<30} | {'Score':>7} | {'Action':<10} | {'Reason'}"
    print(header)
    print("-" * len(header))

    scores = [it["top1_text_score"] for it in items]

    for idx, it in enumerate(items, start=1):
        d_short = it["description"][:41]
        svc_short = it["top1_service"][:29]
        reason_short = it["reason"].split("(")[0].strip()
        print(f"{idx:<3} | {d_short:<42} | {svc_short:<30} | {it['top1_text_score']:>7.4f} | {it['action']:<10} | {reason_short}")

    stats = {
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
    }

    print("-" * len(header))
    print(f"DISTRIBUTION (top-1 text_score): Min = {stats['min']:.4f} | Median = {stats['median']:.4f} | Max = {stats['max']:.4f}")
    
    # Action counts
    no_act_cnt = sum(1 for it in items if it["action"] == "no_action")
    draft_cnt = sum(1 for it in items if it["action"] == "draft")
    auto_cnt = sum(1 for it in items if it["action"] == "auto_create")
    print(f"ACTION BREAKDOWN: no_action = {no_act_cnt}/{len(items)} ({no_act_cnt/len(items)*100:.1f}%), draft = {draft_cnt}/{len(items)} ({draft_cnt/len(items)*100:.1f}%), auto_create = {auto_cnt}/{len(items)} ({auto_cnt/len(items)*100:.1f}%)")
    print("=" * 105)

    return stats


async def main_async() -> None:
    settings = get_settings()
    print("=" * 105)
    print("               TEXT-SCORE GUARD ANALYSIS: TEXT-ONLY JUNK & COMPLETION CHECK")
    print("=" * 105)
    print(f"Guard Settings: ENABLED={settings.DECISION_TEXT_GUARD_ENABLED} | NO_ACTION_FLOOR={settings.DECISION_TEXT_NO_ACTION_FLOOR:.2f} | AUTO_MIN={settings.DECISION_TEXT_AUTO_MIN:.2f}")
    print(f"Reranker Mode : {'ON (Cross-encoder)' if settings.ENABLE_RERANKER else 'OFF (Bi-encoder mode)'}")
    print(f"TEI Embed URL : {settings.TEI_EMBED_URL}")
    print(f"Database URL  : {settings.DATABASE_URL.split('@')[-1]}")

    pool = create_pool(settings)
    await pool.open()
    tei_client = TEIClient(
        settings.TEI_EMBED_URL,
        settings.TEI_RERANK_URL,
        api_key=settings.TEI_API_KEY,
        timeout=10,
    )

    try:
        # Group A: Unrelated Messages
        res_unrelated = await evaluate_descriptions(UNRELATED_MESSAGES, pool, tei_client, settings)
        stats_unrelated = print_group_results("Group A: 20 Unrelated Messages (Greetings, Lunch, Personal, Banter)", res_unrelated)

        # Group B: Completion Updates
        res_completion = await evaluate_descriptions(COMPLETION_UPDATES, pool, tei_client, settings)
        stats_completion = print_group_results("Group B: 10 Completion Updates (Work finished / done reports)", res_completion)

        print("\n" + "=" * 105)
        print(" SUMMARY COMPARISON: UNRELATED JUNK vs COMPLETION UPDATES")
        print("=" * 105)
        print(f"Group A (Unrelated, n=20) : Min = {stats_unrelated['min']:.4f} | Median = {stats_unrelated['median']:.4f} | Max = {stats_unrelated['max']:.4f}")
        print(f"Group B (Completion, n=10): Min = {stats_completion['min']:.4f} | Median = {stats_completion['median']:.4f} | Max = {stats_completion['max']:.4f}")
        print("=" * 105 + "\n")

    finally:
        await pool.close()
        await tei_client.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

