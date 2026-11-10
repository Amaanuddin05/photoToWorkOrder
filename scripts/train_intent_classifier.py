"""Train the message-intent classifier (issue / done / other) on bge-m3 embeddings from the live TEI.

  ..\\venv\\Scripts\\python.exe scripts\\train_intent_classifier.py [--c 10] [--save]

Fits on eval/intent_train.csv split=train, reports on split=dev (held-out TOPICS) and on the independent test set
(eval/intent_samples.csv + real groundTruth(_ar).csv descriptions as 'issue').  --save refits on train+dev and writes
app/intent/intent_bge_m3.json (weights + the TEI model id/dim, which the service verifies at load).
"""
import argparse, csv, json, sys
from pathlib import Path

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
LABELS = ["issue", "done", "other"]


def embed(texts, url, key):
    out = []
    for i in range(0, len(texts), 16):
        r = httpx.post(f"{url}/embed", json={"inputs": texts[i:i + 16]},
                       headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=120)
        r.raise_for_status()
        out += r.json()
    X = np.array(out, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_rows():
    rows = [(r["lang"], r["label"], r["text"]) for r in read(ROOT / "eval" / "intent_samples.csv")]
    # NB: intent_samples uses issue/done/other labels already
    for lang, fn in (("en", "groundTruth.csv"), ("ar", "groundTruth_ar.csv")):
        for r in read(ROOT / fn):
            if r["service"].strip().upper() != "NONE":
                rows.append((lang, "issue", r["description"]))
    return rows


def metrics(name, clf, X, y, lang, show=True):
    P = clf.predict_proba(X)
    p = P.argmax(1)
    res = {"n": len(y), "acc": round(100 * np.mean(p == y), 2)}
    for lg in ("en", "ar"):
        m = lang == lg
        if m.any():
            res[lg] = round(100 * np.mean(p[m] == y[m]), 2)
    for i, lab in enumerate(LABELS):
        m = y == i
        res[f"recall_{lab}"] = round(100 * np.mean(p[m] == i), 2)
    if show:
        print(f"[{name}]", res)
        print(confusion_matrix(y, p, labels=[0, 1, 2]))
    return res, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--url", default="http://localhost:8081")
    ap.add_argument("--key", default="test")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    info = httpx.get(f"{a.url}/info", headers={"Authorization": f"Bearer {a.key}"}, timeout=10).json()
    model_id = info["model_id"]
    print("TEI model:", model_id)
    tr = read(ROOT / "eval" / "intent_train.csv")
    Xall = embed([r["text"] for r in tr], a.url, a.key)
    yall = np.array([LABELS.index(r["label"]) for r in tr])
    split = np.array([r["split"] for r in tr])
    lang = np.array([r["lang"] for r in tr])
    trn, dev = split == "train", split == "dev"

    clf = LogisticRegression(C=a.c, max_iter=5000, class_weight="balanced").fit(Xall[trn], yall[trn])
    metrics("dev (held-out topics)", clf, Xall[dev], yall[dev], lang[dev])

    T = test_rows()
    Xt = embed([t[2] for t in T], a.url, a.key)
    yt = np.array([LABELS.index(t[1]) for t in T])
    lt = np.array([t[0] for t in T])
    res, P = metrics("independent test", clf, Xt, yt, lt)

    print("\nthreshold curve (test): act on done/other only when prob >= t")
    p = P.argmax(1); conf = P.max(1)
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        act = (p != 0) & (conf >= t)           # would be routed to no_action
        dropped_issue = int(np.sum(act & (yt == 0)))
        caught = int(np.sum(act & (yt != 0)))
        print(f"  t={t:.2f}: caught {caught}/{int(np.sum(yt != 0))} non-issues, real issues wrongly dropped {dropped_issue}/{int(np.sum(yt == 0))}")

    if a.save:
        full = LogisticRegression(C=a.c, max_iter=5000, class_weight="balanced").fit(Xall, yall)
        out = ROOT / "app" / "intent"
        out.mkdir(exist_ok=True)
        (out / "intent_bge_m3.json").write_text(json.dumps({
            "version": 1, "tei_model_id": model_id, "dim": int(Xall.shape[1]), "labels": LABELS,
            "normalize": True, "C": a.c, "n_train": int(len(yall)),
            "coef": full.coef_.tolist(), "intercept": full.intercept_.tolist(),
            "test_metrics": res,
        }), encoding="utf-8")
        print("saved", out / "intent_bge_m3.json")


if __name__ == "__main__":
    main()
