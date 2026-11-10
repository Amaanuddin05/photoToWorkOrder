"""Cache NLI logits for several hypothesis wordings, then evaluate raw vs calibrated decisions.
Usage: python scripts/nli_intent_features.py <minilm|mdeberta> [threads]"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import eval_nli_intent as m

HYPS = {  # name -> hypothesis
 "i1": "This message reports a problem that needs fixing.",
 "i2": "Something is broken, damaged or not working.",
 "i3": "A maintenance issue is being reported.",
 "d1": "This message reports that work has been completed.",
 "d2": "Something has been repaired, fixed or replaced.",
 "d3": "The task is done.",
 "o1": "This message is not about maintenance work.",
 "o2": "This is a casual or administrative message.",
 "o3": "This message is about a meeting, food, greetings or other everyday topics.",
}
key = sys.argv[1]; th = int(sys.argv[2]) if len(sys.argv) > 2 else 2
torch.set_num_threads(th)
S = m.load_samples()
n = m.Nli(m.MODELS[key], False)
names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(HYPS); hy = [HYPS[k] for k in names]
feats = []
t0 = time.time()
for lang, lab, text, src in S:
    enc = n.tok([text] * len(hy), hy, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.inference_mode():
        lg = n.model(**enc).logits
    feats.append(torch.log_softmax(lg, -1).numpy().tolist())   # [hyp][3]
print("feature time", round(time.time() - t0), "s", flush=True)
json.dump({"names": names, "id2label": n.model.config.id2label, "ent": n.ent,
           "samples": S, "feats": feats}, open(f"C:/Users/Win11/AppData/Local/Temp/claude/c0e9e142de3c626b/48b6b8e7-7955-43b5-bd8e-4bf6853adb6e/scratchpad/feats_{key}.json", "w", encoding="utf-8"), ensure_ascii=False)
