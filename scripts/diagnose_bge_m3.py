"""
Read-only text-embedding diagnostic for the current bge-m3 setup (scripts/diagnose_bge_m3.py).

Changes NOTHING: the catalog text is read from the DB with SELECTs only, everything
else is computed in memory from TEI /embed (plain text, no prefix). No config,
thresholds or defaults are changed.

Uses the 62 real (non-NONE) rows of ground_truth.csv inside eval_dataset.zip, text only.
Metrics are TEXT-ONLY top-1 / top-3 / mean rank of the correct service, where
text_score = w * cos(query, descEmb) + (1 - w) * cos(query, titleEmb) over the active catalog.

  A. weight sweep w = 0.0 .. 1.0
  B. hubness at the current w (PIPELINE_TEXT_DESC_WEIGHT)
  C. lowercase test (both / catalog-only / queries-only)
  D. mean-centering test (catalog mean of title+desc vectors)
  E. TEI vs sentence_transformers.SentenceTransformer("BAAI/bge-m3") on the 11 reference texts

Usage (from the siglip folder):
  ..\\venv\\Scripts\\python.exe scripts\\diagnose_bge_m3.py
Output: app/eval/runs/model_compare/diagnose_bge_m3_<ts>.txt (+ .json)
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path

import httpx
import numpy as np
import psycopg

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.config import get_settings  # noqa: E402

OUT_DIR = base_dir / "app" / "eval" / "runs" / "model_compare"
WEIGHTS = [round(i / 10, 1) for i in range(11)]
REF_QUERIES = ["AC in room 204 is blowing warm air", "المصعد معطل ولا يعمل", "good morning everyone"]


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        self.f.write(line + "\n")
        sys.stdout.write(line.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8", "replace") + "\n")


# ── data ──────────────────────────────────────────────────────────────────────

def load_gt_rows() -> list[dict]:
    with zipfile.ZipFile(base_dir / "eval_dataset.zip") as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("utf-8-sig")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        svc = (r.get("service") or "").strip()
        desc = (r.get("description") or "").strip()
        if svc.upper() == "NONE" or not desc:
            continue
        rows.append({"image": r["image"].strip(), "description": desc, "service": svc})
    return rows


def load_catalog(s, text_model: str) -> tuple[list[dict], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Active catalog text + the stored per-model vectors of `text_model` (ServiceTextEmbedding)."""
    with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
        conn.read_only = True
        cur = conn.execute(
            'SELECT s.id, s.title, s.description, e."titleEmb"::text, e."descEmb"::text, s.keywords FROM "WoService" s '
            'LEFT JOIN "ServiceTextEmbedding" e ON e."serviceId" = s.id AND e."textModel" = %s '
            'WHERE s."isActive" ORDER BY s.id;',
            (text_model,),
        )
        rows = cur.fetchall()
        conn.rollback()
    from app.text_embedding import catalog_texts
    # "description" = raw text (for reports); "embed_desc" = what is embedded: "{description}. {keywords}"
    cat = [{"id": r[0], "title": r[1], "description": r[2], "keywords": r[5],
            "embed_desc": catalog_texts(r[1], r[2], r[5])[1]} for r in rows]
    db_vecs = {
        r[0]: (np.array(json.loads(r[3]), dtype=np.float64), np.array(json.loads(r[4]), dtype=np.float64))
        for r in rows if r[3] and r[4]
    }
    return cat, db_vecs


class TEI:
    def __init__(self, s):
        h = {"Authorization": f"Bearer {s.TEI_API_KEY}"} if s.TEI_API_KEY else {}
        self.c = httpx.Client(base_url=s.TEI_EMBED_URL.rstrip("/"), headers=h, timeout=120.0)
        self.info = self.c.get("/info").raise_for_status().json()
        self.batch = int(self.info.get("max_client_batch_size") or 32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            out.extend(self.c.post("/embed", json={"inputs": texts[i:i + self.batch]}).raise_for_status().json())
        return np.array(out, dtype=np.float64)

    def token_counts(self, texts: list[str]) -> list[int]:
        out = []
        for i in range(0, len(texts), self.batch):
            toks = self.c.post("/tokenize", json={"inputs": texts[i:i + self.batch]}).raise_for_status().json()
            out.extend(sum(1 for t in seq if not t.get("special")) for seq in toks)
        return out


def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# ── ranking ───────────────────────────────────────────────────────────────────

def score_matrix(q: np.ndarray, t: np.ndarray, d: np.ndarray, w: float) -> np.ndarray:
    """(n_queries, n_services) text_score; inputs are L2-normalized so dot = cosine."""
    return w * (q @ d.T) + (1 - w) * (q @ t.T)


def evaluate(scores: np.ndarray, gold_idx: list[int]) -> dict:
    ranks = []
    top1 = []
    for i, g in enumerate(gold_idx):
        order = np.argsort(-scores[i], kind="stable")
        ranks.append(int(np.where(order == g)[0][0]) + 1)
        top1.append(int(order[0]))
    n = len(ranks)
    return {
        "top1": 100.0 * sum(r == 1 for r in ranks) / n,
        "top3": 100.0 * sum(r <= 3 for r in ranks) / n,
        "mean_rank": float(np.mean(ranks)),
        "ranks": ranks,
        "pred": top1,
    }


def sweep(q, t, d, gold) -> list[dict]:
    return [{"w": w, **{k: v for k, v in evaluate(score_matrix(q, t, d, w), gold).items() if k not in ("ranks", "pred")}}
            for w in WEIGHTS]


def print_sweep(out, title: str, rows: list[dict], cur_w: float) -> None:
    out(f"\n{title}")
    out(f"{'w (desc)':>9} | {'top-1':>7} | {'top-3':>7} | {'mean rank':>9}")
    out("-" * 42)
    for r in rows:
        mark = "  <- current w" if abs(r["w"] - cur_w) < 1e-9 else ""
        mark += "  (title-only)" if r["w"] == 0.0 else ("  (description-only)" if r["w"] == 1.0 else "")
        out(f"{r['w']:>9.1f} | {r['top1']:>6.2f}% | {r['top3']:>6.2f}% | {r['mean_rank']:>9.2f}{mark}")
    best = max(rows, key=lambda r: (r["top1"], r["top3"], -r["mean_rank"]))
    out(f"best: w={best['w']:.1f} top-1={best['top1']:.2f}% top-3={best['top3']:.2f}% mean rank={best['mean_rank']:.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    s = get_settings()
    cur_w = float(s.PIPELINE_TEXT_DESC_WEIGHT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Tee(OUT_DIR / f"diagnose_bge_m3_{ts}.txt")
    result: dict = {"created_at": ts, "current_w": cur_w}

    tei = TEI(s)
    out("=" * 100)
    out(" bge-m3 TEXT-ONLY DIAGNOSTIC (read-only; in memory; no prefix)")
    out("=" * 100)
    out(f"TEI model: {tei.info.get('model_id')}  pooling={tei.info.get('model_type')}  "
        f"max_input_length={tei.info.get('max_input_length')}  TEXT_EMBED_MODEL_ID={s.TEXT_EMBED_MODEL_ID}")
    out(f"Current PIPELINE_TEXT_DESC_WEIGHT w = {cur_w}")

    gt = load_gt_rows()
    cat, db_vecs = load_catalog(s, tei.info.get("model_id"))
    titles = [c["title"].strip() for c in cat]
    descs = [(c["description"] or "").strip() for c in cat]
    emb_descs = [c["embed_desc"] for c in cat]  # description + keywords, as embedded in the catalog
    lookup = {t.lower(): i for i, t in enumerate(titles)}
    gold = [lookup[r["service"].lower()] for r in gt]
    queries = [r["description"] for r in gt]
    out(f"Ground-truth real rows: {len(gt)}   Active catalog services: {len(cat)}")
    assert len(gt) == 62, f"expected 62 real rows, got {len(gt)}"

    # Embeddings as the pipeline uses them (plain text)
    T = tei.embed(titles)
    D = tei.embed(emb_descs)
    Q = tei.embed(queries)

    # Sanity: fresh TEI catalog vectors vs vectors stored in the DB
    diffs = [max(np.abs(T[i] - db_vecs[c["id"]][0]).max(), np.abs(D[i] - db_vecs[c["id"]][1]).max())
             for i, c in enumerate(cat) if c["id"] in db_vecs]
    out(f"Sanity: max |fresh TEI vector - stored ServiceTextEmbedding vector| = {max(diffs):.2e} ({len(diffs)} services)" if diffs else "Sanity: no stored vectors for this model yet")

    # ── A ──
    a_rows = sweep(Q, T, D, gold)
    print_sweep(out, "A. WEIGHT SWEEP (plain text, as served)", a_rows, cur_w)
    result["A"] = a_rows

    # ── B ──
    cur = evaluate(score_matrix(Q, T, D, cur_w), gold)
    S = score_matrix(Q, T, D, cur_w)
    tok_t = tei.token_counts(titles)
    tok_d = tei.token_counts(descs)
    stats = []
    for i, c in enumerate(cat):
        stats.append({
            "title": titles[i], "description": descs[i],
            "pred_top1": sum(1 for p in cur["pred"] if p == i),
            "correct_label": sum(1 for g in gold if g == i),
            "mean_score": float(S[:, i].mean()),
            "title_chars": len(titles[i]), "desc_chars": len(descs[i]),
            "title_tokens": tok_t[i], "desc_tokens": tok_d[i],
        })
    out(f"\nB. HUBNESS at current w={cur_w} (text-only top-1 here = {cur['top1']:.2f}%)")
    out(f"{'Service':<56} | {'pred top-1':>10} | {'correct':>7} | {'mean score':>10}")
    out("-" * 94)
    for st in sorted(stats, key=lambda x: (-x["pred_top1"], -x["mean_score"])):
        out(f"{st['title'][:56]:<56} | {st['pred_top1']:>10} | {st['correct_label']:>7} | {st['mean_score']:>10.4f}")
    hubs = sorted(stats, key=lambda x: (-(x["pred_top1"] - x["correct_label"]), -x["mean_score"]))[:5]
    out("\nTop 5 hubs (ranked by predicted-top-1 minus times-correct, then mean score):")
    for k, h in enumerate(hubs, 1):
        out(f"\n  {k}. {h['title']}")
        out(f"     predicted top-1: {h['pred_top1']} | correct answer for: {h['correct_label']} | mean text_score: {h['mean_score']:.4f}")
        out(f"     title: {h['title_chars']} chars, {h['title_tokens']} tokens")
        out(f"     description: {h['desc_chars']} chars, {h['desc_tokens']} tokens")
        out(f"     description text: {h['description']}")
    mean_by_svc = sorted(stats, key=lambda x: -x["mean_score"])
    out("\nMean text_score ranking across all 62 queries (higher = closer to everything):")
    for st in mean_by_svc[:5]:
        out(f"  {st['mean_score']:.4f}  {st['title']}")
    out("  ...")
    for st in mean_by_svc[-3:]:
        out(f"  {st['mean_score']:.4f}  {st['title']}")
    result["B"] = {"services": stats, "top5_hubs": [h["title"] for h in hubs]}

    # ── C ──
    Tl = tei.embed([t.lower() for t in titles])
    Dl = tei.embed([d.lower() for d in emb_descs])
    Ql = tei.embed([q.lower() for q in queries])
    c_rows = sweep(Ql, Tl, Dl, gold)
    print_sweep(out, "C. LOWERCASE TEST: catalog AND queries lowercased", c_rows, cur_w)
    cat_only = evaluate(score_matrix(Q, Tl, Dl, cur_w), gold)
    q_only = evaluate(score_matrix(Ql, T, D, cur_w), gold)
    both = evaluate(score_matrix(Ql, Tl, Dl, cur_w), gold)
    out(f"\nAt current w={cur_w}:")
    out(f"{'variant':<34} | {'top-1':>7} | {'top-3':>7} | {'mean rank':>9}")
    for name, r in (("as served (no lowercasing)", cur), ("catalog lowercased only", cat_only),
                    ("queries lowercased only", q_only), ("both lowercased", both)):
        out(f"{name:<34} | {r['top1']:>6.2f}% | {r['top3']:>6.2f}% | {r['mean_rank']:>9.2f}")
    result["C"] = {"sweep_both": c_rows,
                   "current_w": {k: {m: v[m] for m in ("top1", "top3", "mean_rank")} for k, v in
                                 (("as_served", cur), ("catalog_only", cat_only), ("queries_only", q_only), ("both", both))}}

    # ── D ──
    mu = np.vstack([T, D]).mean(axis=0)
    Tc, Dc, Qc = normalize(T - mu), normalize(D - mu), normalize(Q - mu)
    d_rows = sweep(Qc, Tc, Dc, gold)
    print_sweep(out, "D. MEAN-CENTERING (catalog title+desc mean subtracted from catalog and queries, re-normalized)", d_rows, cur_w)
    out(f"   |catalog mean vector| = {np.linalg.norm(mu):.4f} (0 = no shared direction; 1 = all identical)")
    Sc = score_matrix(Qc, Tc, Dc, cur_w)
    pc = evaluate(Sc, gold)["pred"]
    out("   top-1 counts after centering (current w), top 5: " + ", ".join(
        f"{titles[i]}={pc.count(i)}" for i in sorted(set(pc), key=lambda i: -pc.count(i))[:5]))
    result["D"] = {"sweep": d_rows, "catalog_mean_norm": float(np.linalg.norm(mu))}

    # Pairwise catalog similarity (context for hubness)
    out("\nCatalog geometry (plain text):")
    for name, M in (("title-title", T), ("desc-desc", D)):
        G = M @ M.T
        off = G[~np.eye(len(M), dtype=bool)]
        out(f"   {name} cosine off-diagonal: min={off.min():.4f} median={np.median(off):.4f} max={off.max():.4f}")
    out(f"   query-query cosine median: {np.median((Q @ Q.T)[~np.eye(len(Q), dtype=bool)]):.4f}")

    # ── E ──
    out("\nE. REFERENCE CHECK: TEI vs sentence_transformers.SentenceTransformer('BAAI/bge-m3')")
    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
    except Exception as exc:
        out(f"   sentence-transformers is not importable in this venv ({exc}); E skipped, nothing installed.")
        result["E"] = {"skipped": str(exc)}
    else:
        with psycopg.connect(s.DATABASE_URL, options=f"-c search_path={s.DB_SCHEMA}") as conn:
            conn.read_only = True
            ref_rows = conn.execute('SELECT title, description FROM "WoService" ORDER BY id LIMIT 4;').fetchall()
            conn.rollback()
        ref_texts = [x for r in ref_rows for x in r] + REF_QUERIES
        st_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        pool_mod = [m for m in st_model.modules() if type(m).__name__ == "Pooling"]
        out(f"   sentence-transformers {sentence_transformers.__version__}; modules: "
            + " -> ".join(type(m).__name__ for m in st_model._modules.values())
            + (f"; pooling config={pool_mod[0].get_config_dict()}" if pool_mod and hasattr(pool_mod[0], "get_config_dict") else ""))
        ref = st_model.encode(ref_texts, normalize_embeddings=False, convert_to_numpy=True).astype(np.float64)
        ref_norms = np.linalg.norm(ref, axis=1)
        tei_v = tei.embed(ref_texts)
        cos = (normalize(ref) * normalize(tei_v)).sum(axis=1)
        for t_, c_, n_ in zip(ref_texts, cos, ref_norms):
            out(f"   cos={c_:.6f}  |ST vec|={n_:.4f}  {t_[:70]}")
        out(f"   min cosine TEI vs sentence-transformers: {cos.min():.6f}")
        result["E"] = {"min_cosine": float(cos.min()), "cosines": [float(x) for x in cos], "texts": ref_texts}

    # ── catalog text audit ──
    out("\nCATALOG TEXT AUDIT")
    out(f"{'title':<56} | {'t ch/tok':>8} | {'d ch/tok':>8} | flags")
    flags_all = {}
    for st in stats:
        flags = []
        letters_t = [ch for ch in st["title"] if ch.isalpha()]
        letters_d = [ch for ch in st["description"] if ch.isalpha()]
        if letters_t and all(ch.isupper() for ch in letters_t):
            flags.append("ALL-CAPS title")
        if letters_d and sum(ch.isupper() for ch in letters_d) / len(letters_d) > 0.6:
            flags.append("mostly-caps description")
        if st["desc_chars"] < 60:
            flags.append("short description")
        if st["desc_chars"] > 300:
            flags.append("long description")
        if st["description"].count(",") >= 4 or st["description"].count(";") >= 2 or re.search(r"(?i)keywords?\s*:", st["description"]):
            flags.append(f"list-like description ({st['description'].count(',')} commas)")
        if st["title"].lower() in st["description"].lower():
            flags.append("title repeated in description")
        flags_all[st["title"]] = flags
        out(f"{st['title'][:56]:<56} | {st['title_chars']:>3}/{st['title_tokens']:<4} | {st['desc_chars']:>3}/{st['desc_tokens']:<4} | {', '.join(flags) or '-'}")
    out("\nFull catalog descriptions:")
    for st in stats:
        out(f"  [{st['title']}] {st['description']}")
    result["audit_flags"] = flags_all

    # ── summary ──
    cands = [("A plain", r) for r in a_rows] + [("C lowercase both", r) for r in c_rows] + [("D mean-centered", r) for r in d_rows]
    best_name, best = max(cands, key=lambda x: (x[1]["top1"], x[1]["top3"], -x[1]["mean_rank"]))
    out("\nSUMMARY")
    out(f"  Best text-only top-1 across A/C/D: {best['top1']:.2f}% ({best_name}, w={best['w']:.1f}; "
        f"top-3 {best['top3']:.2f}%, mean rank {best['mean_rank']:.2f})")
    out(f"  As served (A, w={cur_w}): {cur['top1']:.2f}%")
    out(f"  Top hubs: {', '.join(h['title'] + ' (' + str(h['pred_top1']) + ' top-1 / ' + str(h['correct_label']) + ' correct)' for h in hubs)}")
    result["summary"] = {"best": {"variant": best_name, **best}}

    (OUT_DIR / f"diagnose_bge_m3_{ts}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out(f"\nSaved: {OUT_DIR / f'diagnose_bge_m3_{ts}.txt'} (+ .json)")


if __name__ == "__main__":
    main()
