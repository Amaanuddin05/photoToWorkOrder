"""Message-intent gate: is a post an *issue report*, a *work-completed* note, or *other* chatter?

A multinomial logistic regression over the (L2-normalised) bge-m3 embedding of the description. The query is
already embedded for text search, so inference is a 3 x 1024 matrix product (~microseconds).  Weights come from
scripts/train_intent_classifier.py -> app/intent/intent_bge_m3.json.

Only `done` / `other` predictions with probability >= INTENT_MIN_PROB are acted on; anything uncertain falls
through to the normal pipeline, so a real issue is never dropped on a hunch.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("siglip.intent_gate")
_APP_DIR = Path(__file__).resolve().parent.parent


class IntentGate:
    def __init__(self, data: dict[str, Any]) -> None:
        self.labels: list[str] = data["labels"]
        self.tei_model_id: str = data["tei_model_id"]
        self.dim: int = int(data["dim"])
        self.coef = np.asarray(data["coef"], dtype=np.float64)
        self.intercept = np.asarray(data["intercept"], dtype=np.float64)

    def predict(self, vec: list[float]) -> tuple[str, float, dict[str, float]]:
        x = np.asarray(vec, dtype=np.float64)
        x = x / (np.linalg.norm(x) or 1.0)
        z = self.coef @ x + self.intercept
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        i = int(p.argmax())
        return self.labels[i], float(p[i]), {l: round(float(v), 4) for l, v in zip(self.labels, p)}


@lru_cache(maxsize=4)
def load_gate(path: str) -> IntentGate | None:
    p = Path(path)
    if not p.is_absolute():
        p = _APP_DIR / p
    try:
        return IntentGate(json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:  # missing / corrupt file must never break the pipeline
        logger.warning("Intent gate disabled: cannot load %s (%s)", p, exc)
        return None


def classify(settings: Any, text_model: str | None, query_vec: list[float] | None) -> dict[str, Any] | None:
    """Return {label, prob, probs, acted} or None when the gate is off / not applicable.

    Refuses to run when the live TEI model differs from the one the classifier was trained on.
    """
    if not getattr(settings, "INTENT_ENABLED", False) or not query_vec:
        return None
    gate = load_gate(settings.INTENT_MODEL_PATH)
    if gate is None:
        return None
    if text_model != gate.tei_model_id or len(query_vec) != gate.dim:
        logger.warning("Intent gate skipped: classifier trained on %s (dim %d), live model %s (dim %d)",
                       gate.tei_model_id, gate.dim, text_model, len(query_vec))
        return None
    label, prob, probs = gate.predict(query_vec)
    acted = label != "issue" and prob >= settings.INTENT_MIN_PROB
    return {"label": label, "prob": round(prob, 4), "probs": probs, "acted": acted}
