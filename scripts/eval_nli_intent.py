"""Benchmark zero-shot NLI models for the message-intent gate (issue / done / other).

Usage:  ..\\venv\\Scripts\\python.exe scripts\\eval_nli_intent.py [--models a,b] [--threads 2] [--quant]
Reads eval/intent_samples.csv plus the real descriptions in groundTruth*.csv (label=issue).
Writes app/eval/runs/model_compare/intent_<model>.json
"""
import argparse, csv, json, statistics, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "minilm": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
}
LABELS = ["issue", "done", "other"]
HYP = {
    "en": [
        "This message reports a problem that needs fixing.",
        "This message reports that work has been completed.",
        "This message is not about maintenance work.",
    ],
    "ar": [
        "تبلغ هذه الرسالة عن مشكلة تحتاج إلى إصلاح.",
        "تبلغ هذه الرسالة عن اكتمال العمل.",
        "هذه الرسالة لا تتعلق بأعمال الصيانة.",
    ],
}


def load_samples():
    rows = []
    with open(ROOT / "eval" / "intent_samples.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((r["lang"], r["label"], r["text"], "synthetic"))
    for lang, fn in (("en", "groundTruth.csv"), ("ar", "groundTruth_ar.csv")):
        with open(ROOT / fn, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["service"].strip().upper() != "NONE":
                    rows.append((lang, "issue", r["description"], "groundtruth"))
    return rows


class Nli:
    def __init__(self, model_id, quant):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        if quant:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent = l2i["entailment"]
        self.con = l2i.get("contradiction")

    @torch.inference_mode()
    def scores(self, text, hyps):
        enc = self.tok([text] * len(hyps), hyps, return_tensors="pt", padding=True, truncation=True, max_length=128)
        logits = self.model(**enc).logits
        return logits[:, self.ent].tolist()


def run(key, model_id, hyp_lang, samples, quant, threads):
    torch.set_num_threads(threads)
    nli = Nli(model_id, quant)
    nli.scores("warm up", HYP["en"])
    hyps = HYP[hyp_lang]
    preds, lat = [], []
    for lang, label, text, src in samples:
        t0 = time.perf_counter()
        s = nli.scores(text, hyps)
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(LABELS[max(range(3), key=lambda i: s[i])])
    return preds, lat


def report(name, samples, preds, lat):
    out = {"name": name, "n": len(samples), "ms_mean": round(statistics.mean(lat), 1),
           "ms_p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1)}
    ok = sum(p == s[1] for p, s in zip(preds, samples))
    out["acc"] = round(100 * ok / len(samples), 2)
    for lang in ("en", "ar"):
        idx = [i for i, s in enumerate(samples) if s[0] == lang]
        out[f"acc_{lang}"] = round(100 * sum(preds[i] == samples[i][1] for i in idx) / len(idx), 2)
        for lab in LABELS:
            li = [i for i in idx if samples[i][1] == lab]
            out[f"recall_{lang}_{lab}"] = round(100 * sum(preds[i] == lab for i in li) / len(li), 1)
    conf = {a: {b: 0 for b in LABELS} for a in LABELS}
    for p, s in zip(preds, samples):
        conf[s[1]][p] += 1
    out["confusion_true_to_pred"] = conf
    out["errors"] = [{"lang": s[0], "true": s[1], "pred": p, "text": s[2]}
                     for p, s in zip(preds, samples) if p != s[1]][:60]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mdeberta,minilm")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quant", action="store_true", help="torch dynamic int8")
    ap.add_argument("--hyp", default="en,ar", help="hypothesis languages to try")
    a = ap.parse_args()
    samples = load_samples()
    print(f"{len(samples)} samples", {l: sum(s[1] == l for s in samples) for l in LABELS})
    outdir = ROOT / "app" / "eval" / "runs" / "model_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for key in a.models.split(","):
        for hl in a.hyp.split(","):
            name = f"{key}{'-int8' if a.quant else ''}-hyp{hl}-t{a.threads}"
            preds, lat = run(key, MODELS[key], hl, samples, a.quant, a.threads)
            r = report(name, samples, preds, lat)
            results.append(r)
            print(f"{name}: acc={r['acc']} en={r['acc_en']} ar={r['acc_ar']} mean={r['ms_mean']}ms p95={r['ms_p95']}ms")
            print("   confusion", r["confusion_true_to_pred"])
    (outdir / f"intent_{'-'.join(a.models.split(','))}{'-int8' if a.quant else ''}_t{a.threads}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
