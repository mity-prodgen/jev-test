#!/usr/bin/env python3
"""Test 2: reranker bake-off on BEIR SciFact ("precision comes from Jev").

First stage: Qwen3-Embedding-0.6B takes the top 20 per query. Rerank with: (a) embeddings only,
(b) Jev Noul, (c) Jev Score 0-3, (d) bge-reranker-v2-m3 run locally.

    python rag/test2_rerank.py --local              # first stage + cross-encoder (local, free)
    python rag/test2_rerank.py                      # dry-run: count and price the Jev calls
    python rag/test2_rerank.py --live [--batched]   # call Jev (per-pair; --batched adds 20-docs-per-request)
    python rag/test2_rerank.py --analyze            # metrics, ties, overlap buckets, cost, charts, RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import time
from collections import defaultdict

import numpy as np

from common import (CACHE, DATA, EMBED_MODEL, JEV_MODEL, QUERY_INSTRUCTION, RERANK_MODEL, RESULTS, SEED, TOP_K,
                    batched_request, doc_text, pair_request, update_results_section)
from jev_harness.cache import DiskCache
from jev_harness.cost import JEV_TOKEN_FACTOR, JEV_USD_PER_MTOK_INPUT, jev_step_estimate, print_dry_run
from jev_harness.metrics import (bootstrap_ci, mrr_at_k, ndcg_at_k, paired_bootstrap_diff, percentile, recall_at_k)
from jev_harness.pairs import input_tokens_of, is_cached, noul_of, run_requests, score_of

STAGE1_FILE = RESULTS / "test2_stage1.json"
CE_FILE = RESULTS / "test2_crossenc.json"
PAIRS_FILE = RESULTS / "test2_jev_pairs.json"
BATCH_FILE = RESULTS / "test2_jev_batched.json"
STOP = set("a an the of in on at to for and or but is are was were be been being by with as from that this these those it its "
           "has have had not no do does did can may might will would should than then there their they we our which who whom "
           "into over under between among about after before during via per vs versus".split())


def load_scifact():
    base = DATA / "scifact" / "scifact"
    corpus = {}
    for line in open(base / "corpus.jsonl"):
        d = json.loads(line)
        corpus[d["_id"]] = doc_text(d.get("title", ""), d["text"])
    queries = {}
    for line in open(base / "queries.jsonl"):
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    qrels = defaultdict(set)
    for i, line in enumerate(open(base / "qrels" / "test.tsv")):
        if i == 0:
            continue
        qid, did, score = line.split()
        if int(score) > 0:
            qrels[qid].add(did)
    test_q = sorted(qrels, key=int)
    return corpus, queries, qrels, test_q


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device=device())
    m.max_seq_length = 1024
    return m


def fmt_query(q):
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{q}"


def stage1(corpus, queries, test_q):
    if STAGE1_FILE.exists():
        return json.load(open(STAGE1_FILE))
    ids = list(corpus)
    emb_file = DATA / "scifact" / "emb_qwen3_06b_docs.npy"
    q_file = DATA / "scifact" / "emb_qwen3_06b_queries.npy"
    model = embed_model()
    if emb_file.exists():
        D = np.load(emb_file)
    else:
        print(f"Embedding {len(ids)} documents on {device()}...")
        t = time.time()
        D = model.encode([corpus[i] for i in ids], batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.save(emb_file, D)
        print(f"  done in {time.time() - t:.0f}s")
    Q = model.encode([fmt_query(queries[q]) for q in test_q], batch_size=16, normalize_embeddings=True)
    np.save(q_file, Q)
    sims = Q @ D.T
    out = {}
    for i, qid in enumerate(test_q):
        top = np.argsort(-sims[i])[:40]
        out[qid] = [{"id": ids[j], "score": float(sims[i, j])} for j in top]
    json.dump(out, open(STAGE1_FILE, "w"))
    return out


def run_cross_encoder(corpus, queries, test_q, s1):
    if CE_FILE.exists():
        return json.load(open(CE_FILE))
    import torch
    from sentence_transformers import CrossEncoder
    dev = device()
    model = CrossEncoder(RERANK_MODEL, max_length=512, device=dev)
    batch = TOP_K
    for qid in test_q[:3]:  # warm-up, not timed
        model.predict([(queries[qid], corpus[d["id"]]) for d in s1[qid][:TOP_K]], batch_size=batch, show_progress_bar=False)
    scores, lats = {}, {}
    for n, qid in enumerate(test_q):
        pairs = [(queries[qid], corpus[d["id"]]) for d in s1[qid][:TOP_K]]
        if dev == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        s = model.predict(pairs, batch_size=batch, show_progress_bar=False)
        if dev == "mps":
            torch.mps.synchronize()
        lats[qid] = (time.perf_counter() - t) * 1000
        scores[qid] = {d["id"]: float(x) for d, x in zip(s1[qid][:TOP_K], s)}
        if (n + 1) % 50 == 0:
            print(f"  cross-encoder {n + 1}/{len(test_q)}", flush=True)
    hw = {"machine": platform.machine(), "processor": platform.processor(), "device": dev, "torch": torch.__version__,
          "batch_size": batch, "max_length": 512, "model": RERANK_MODEL, "pairs_per_query": TOP_K}
    try:
        import subprocess
        hw["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        hw["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30)
    except Exception:
        pass
    out = {"scores": scores, "latency_ms": lats, "hardware": hw}
    json.dump(out, open(CE_FILE, "w"))
    return out


def jev_requests(corpus, queries, test_q, s1):
    reqs = []
    for qid in test_q:
        for d in s1[qid][:TOP_K]:
            for kind in ("noul", "score"):
                reqs.append(pair_request(f"{qid}|{d['id']}|{kind}", queries[qid], corpus[d["id"]], kind))
    return reqs


def batched_requests(corpus, queries, test_q, s1):
    reqs = []
    for qid in test_q:
        docs = [corpus[d["id"]] for d in s1[qid][:TOP_K]]
        for kind in ("noul", "score"):
            reqs.append(batched_request(f"{qid}|batch|{kind}", queries[qid], docs, kind))
    return reqs


def req_text(r):
    return json.dumps(r.state) + json.dumps(r.questions)


# ----------------------------- analysis -----------------------------

def rank_by(scores, base):
    pos = {d: i for i, d in enumerate(base)}
    return sorted(base, key=lambda d: (-scores.get(d, -1e9), pos[d]))


def stem_set(text, stemmer):
    return {stemmer.stemWord(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def collect_methods(test_q, s1, ce, pair_rows, batch_rows):
    base = {q: [d["id"] for d in s1[q][:TOP_K]] for q in test_q}
    methods = {"embeddings": {q: base[q] for q in test_q}}
    raw = {}
    for name, rows, suffix in (("jev_noul", pair_rows, ""), ("jev_score", pair_rows, "")):
        if not rows:
            continue
        kind = name.split("_")[1]
        sc = defaultdict(dict)
        for r in rows:
            if r["kind"] == kind and r["value"] is not None:
                sc[r["qid"]][r["doc"]] = r["value"]
        if sc:
            raw[name] = sc
            methods[name] = {q: rank_by(sc[q], base[q]) for q in test_q}
    if batch_rows:
        for kind in ("noul", "score"):
            sc = defaultdict(dict)
            for r in batch_rows:
                if r["kind"] == kind and r["value"] is not None:
                    sc[r["qid"]][r["doc"]] = r["value"]
            if sc:
                raw[f"jev_{kind}_batched"] = sc
                methods[f"jev_{kind}_batched"] = {q: rank_by(sc[q], base[q]) for q in test_q}
    if ce:
        raw["cross_encoder"] = ce["scores"]
        methods["cross_encoder"] = {q: rank_by(ce["scores"][q], base[q]) for q in test_q}
    return methods, raw, base


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import snowballstemmer

    corpus, queries, qrels, test_q = load_scifact()
    s1 = json.load(open(STAGE1_FILE))
    ce = json.load(open(CE_FILE)) if CE_FILE.exists() else None
    pair_rows = json.load(open(PAIRS_FILE)) if PAIRS_FILE.exists() else []
    batch_rows = json.load(open(BATCH_FILE)) if BATCH_FILE.exists() else []
    methods, raw, base = collect_methods(test_q, s1, ce, pair_rows, batch_rows)
    names = list(methods)

    per_q = {m: {"ndcg": [], "mrr": [], "recall": []} for m in names}
    for m in names:
        for q in test_q:
            per_q[m]["ndcg"].append(ndcg_at_k(methods[m][q], qrels[q], 10))
            per_q[m]["mrr"].append(mrr_at_k(methods[m][q], qrels[q], 10))
            per_q[m]["recall"].append(recall_at_k(methods[m][q], qrels[q], 10))
    ceiling = statistics.mean(recall_at_k(base[q], qrels[q], TOP_K) for q in test_q)
    labels = {"embeddings": "Embeddings only", "jev_noul": "Jev Noul", "jev_score": "Jev Score", "cross_encoder": "Cross-encoder",
              "jev_noul_batched": "Jev Noul (batched)", "jev_score_batched": "Jev Score (batched)"}

    lines = [f"{len(test_q)} SciFact test queries; first stage {EMBED_MODEL}, top {TOP_K}; Recall@{TOP_K} of the first stage (ceiling for every method) = {ceiling:.3f}.", "",
             "| Method | nDCG@10 [95% CI] | MRR@10 | Recall@10 | nDCG@10 change vs embeddings [95% CI] |", "|---|---|---|---|---|"]
    summary = {}
    for m in names:
        n, lo, hi = bootstrap_ci(per_q[m]["ndcg"], seed=SEED)
        d = paired_bootstrap_diff(per_q[m]["ndcg"], per_q["embeddings"]["ndcg"], seed=SEED) if m != "embeddings" else None
        summary[m] = {"ndcg": n, "ndcg_lo": lo, "ndcg_hi": hi, "mrr": statistics.mean(per_q[m]["mrr"]),
                      "recall": statistics.mean(per_q[m]["recall"]), "diff": d}
        dtxt = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else "baseline"
        lines.append(f"| {labels[m]} | {n:.3f} [{lo:.3f}, {hi:.3f}] | {summary[m]['mrr']:.3f} | {summary[m]['recall']:.3f} | {dtxt} |")

    if "cross_encoder" in names:
        lines += ["", "**Jev vs the cross-encoder** (paired nDCG@10 difference per query, 95% bootstrap CI):", ""]
        for m in [x for x in names if x.startswith("jev_")]:
            d = paired_bootstrap_diff(per_q[m]["ndcg"], per_q["cross_encoder"]["ndcg"], seed=SEED)
            lines.append(f"- {labels[m]} minus cross-encoder: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    # ---- ties in the top 5 (adjacent scores within 0.01) ----
    lines += ["", "**Ties in the top 5** (adjacent reranked scores within 0.01):", "",
              "| Method | queries with at least one close pair in top 5 | mean close pairs per query (max 4) | queries where rank 1 and 2 are within 0.01 |", "|---|---|---|---|"]
    ties = {}
    for m in [x for x in names if x in ("jev_noul", "jev_score", "jev_noul_batched", "jev_score_batched")]:
        any_close, close_n, top12 = 0, 0, 0
        for q in test_q:
            s = [raw[m][q].get(d, -1e9) for d in methods[m][q][:5]]
            close = sum(1 for a, b in zip(s, s[1:]) if abs(a - b) <= 0.0101)
            close_n += close
            any_close += close > 0
            top12 += abs(s[0] - s[1]) <= 0.0101
        ties[m] = {"any": any_close / len(test_q), "mean_pairs": close_n / len(test_q), "top12": top12 / len(test_q)}
        lines.append(f"| {labels[m]} | {ties[m]['any']:.1%} | {ties[m]['mean_pairs']:.2f} | {ties[m]['top12']:.1%} |")

    # ---- overlap buckets ----
    stemmer = snowballstemmer.stemmer("english")
    pairs = []
    for q in test_q:
        qs = stem_set(queries[q], stemmer)
        for dgold in qrels[q]:
            shared = len(qs & stem_set(corpus[dgold], stemmer))
            pairs.append({"qid": q, "doc": dgold, "shared": shared, "overlap": shared / max(len(qs), 1)})
    rest = sorted(p["overlap"] for p in pairs if p["shared"] >= 2)
    c1, c2 = rest[len(rest) // 3], rest[2 * len(rest) // 3]
    for p in pairs:
        p["bucket"] = ("zero" if p["shared"] == 0 else "one word" if p["shared"] == 1
                       else "low" if p["overlap"] <= c1 else "medium" if p["overlap"] <= c2 else "high")
    order = ["zero", "one word", "low", "medium", "high"]
    bucket = {}
    lines += ["", f"**Rank quality by word overlap** (`zero` and `one word` = number of the query's stemmed content words found in the relevant document; for 2 or more shared words, overlap = share of the query's content words: low <= {c1:.2f} < medium <= {c2:.2f} < high; one row per relevant document, so small buckets are anecdotes):", "",
              "| Bucket | n | in first-stage top 20 (ceiling) | " + " | ".join(f"{labels[m]} MRR@10 / Recall@10" for m in names) + " |", "|---|---|---|" + "---|" * len(names)]
    for b in order:
        ps = [p for p in pairs if p["bucket"] == b]
        if not ps:
            continue
        cells = []
        for m in names:
            rr = [1 / (methods[m][p["qid"]].index(p["doc"]) + 1) if p["doc"] in methods[m][p["qid"]][:10] else 0.0 for p in ps]
            hit = [1.0 if p["doc"] in methods[m][p["qid"]][:10] else 0.0 for p in ps]
            bucket[(b, m)] = (statistics.mean(rr), statistics.mean(hit))
            cells.append(f"{statistics.mean(rr):.3f} / {statistics.mean(hit):.3f}")
        in20 = statistics.mean(1.0 if p["doc"] in base[p["qid"]] else 0.0 for p in ps)
        lines.append(f"| {b} | {len(ps)} | {in20:.0%} | " + " | ".join(cells) + " |")

    # ---- cost and latency ----
    lines += ["", "**Cost and latency**", "", "| Method | requests per query | Jev input tokens per query | cost per 1,000 queries | request latency median / p95 (ms) | per-query latency, 20 requests in parallel: median / p95 (ms) |", "|---|---|---|---|---|---|"]
    for kind, rows, tag in [("noul", pair_rows, "jev_noul"), ("score", pair_rows, "jev_score"), ("noul", batch_rows, "jev_noul_batched"), ("score", batch_rows, "jev_score_batched")]:
        rs = [r for r in rows if r["kind"] == kind]
        if tag.endswith("_batched"):
            rs = [r for r in rs if r["tokens"]]  # one row per batched request carries its tokens and latency
        if not rs:
            continue
        tok_q, lat_q, lat_all = defaultdict(int), defaultdict(float), []
        for r in rs:
            tok_q[r["qid"]] += r["tokens"] or 0
            lat_q[r["qid"]] = max(lat_q[r["qid"]], r["latency_ms"] or 0)
            lat_all.append(r["latency_ms"] or 0)
        mt = statistics.mean(tok_q.values())
        per_q_n = len(rs) / len(tok_q)
        lines.append(f"| {labels[tag]} | {per_q_n:.0f} | {mt:,.0f} | ${mt * 1000 * JEV_USD_PER_MTOK_INPUT / 1e6:.3f} | {statistics.median(lat_all):.0f} / {percentile(lat_all, 0.95):.0f} | {statistics.median(lat_q.values()):.0f} / {percentile(list(lat_q.values()), 0.95):.0f} |")
    if ce:
        v = list(ce["latency_ms"].values())
        hw = ce["hardware"]
        lines.append(f"| Cross-encoder ({RERANK_MODEL}, local) | {TOP_K} pairs, one batch | n/a | no dollar cost reported | n/a | {statistics.median(v):.0f} / {percentile(v, 0.95):.0f} |")
        lines += ["", f"Cross-encoder hardware: {hw.get('cpu', hw.get('processor'))}, {hw.get('ram_gb', '?')} GB RAM, torch {hw['torch']} on `{hw['device']}`, batch {hw['batch_size']}, max_length {hw['max_length']}."]
    models = sorted({r["model"] for r in pair_rows + batch_rows if r.get("model")})
    lines += ["", f"Jev model(s) returned: {', '.join(models) or 'none yet'}. Seed {SEED}."]
    body = "\n".join(lines)
    update_results_section("## Test 2: reranker bake-off", body)
    json.dump({"summary": summary, "ties": ties, "bucket_cuts": [c1, c2]}, open(RESULTS / "test2_metrics.json", "w"), indent=1)

    # ---- charts ----
    colors = {"embeddings": "#898781", "jev_noul": "#2a78d6", "jev_score": "#9ec5f4", "cross_encoder": "#e34948",
              "jev_noul_batched": "#1c5cab", "jev_score_batched": "#6da7ec"}
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [summary[m]["ndcg"] for m in names]
    err = [[summary[m]["ndcg"] - summary[m]["ndcg_lo"] for m in names], [summary[m]["ndcg_hi"] - summary[m]["ndcg"] for m in names]]
    bars = ax.bar([labels[m] for m in names], vals, yerr=err, capsize=4, color=[colors[m] for m in names])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3)
    ax.set_ylabel("nDCG@10 (95% bootstrap CI)")
    ax.set_title(f"Test 2: nDCG@10 on SciFact, top-{TOP_K} reranked")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(True, axis="y", color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(RESULTS / "test2_ndcg.png", dpi=150)
    fig, ax = plt.subplots(figsize=(9, 5))
    present = [b for b in order if any((b, m) in bucket for m in names)]
    w = 0.8 / len(names)
    for j, m in enumerate(names):
        ax.bar([i + j * w for i in range(len(present))], [bucket[(b, m)][0] for b in present], w, label=labels[m], color=colors[m])
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(present))])
    ax.set_xticklabels([f"{b}\n(n={sum(1 for p in pairs if p['bucket'] == b)})" for b in present])
    ax.set_ylabel("MRR@10 of the relevant document")
    ax.set_title("Test 2: rank quality by word overlap with the query")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(RESULTS / "test2_overlap.png", dpi=150)
    print(body)


def to_rows(results, kind_of, doc_of):
    rows = []
    for rid, res in results.items():
        qid, doc, kind = rid.split("|")
        rows.append({"qid": qid, "doc": doc, "kind": kind, "value": None, "tokens": input_tokens_of(res),
                     "latency_ms": res.get("latency_ms"), "model": res.get("model"), "error": res.get("error")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--batched", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    if args.analyze:
        analyze()
        return
    corpus, queries, qrels, test_q = load_scifact()
    s1 = stage1(corpus, queries, test_q)
    base = {q: [d["id"] for d in s1[q][:TOP_K]] for q in test_q}
    print(f"{len(test_q)} queries, {len(corpus)} docs; first-stage Recall@{TOP_K} = "
          f"{statistics.mean(recall_at_k(base[q], qrels[q], TOP_K) for q in test_q):.3f}, "
          f"nDCG@10 = {statistics.mean(ndcg_at_k(base[q], qrels[q], 10) for q in test_q):.3f}")
    if args.local:
        ce = run_cross_encoder(corpus, queries, test_q, s1)
        ranked = {q: sorted(base[q], key=lambda d: -ce['scores'][q][d]) for q in test_q}
        print(f"cross-encoder nDCG@10 = {statistics.mean(ndcg_at_k(ranked[q], qrels[q], 10) for q in test_q):.3f}; "
              f"latency median {statistics.median(ce['latency_ms'].values()):.0f} ms; hardware {ce['hardware']}")
        return

    cache = DiskCache(CACHE, "rag")
    sets = [("per-pair", jev_requests(corpus, queries, test_q, s1))]
    if args.batched:
        sets.append(("batched", batched_requests(corpus, queries, test_q, s1)))
    if not args.live:
        steps = []
        for name, reqs in sets:
            new = [r for r in reqs if not is_cached(cache, JEV_MODEL, r)]
            print(f"{name}: {len(reqs)} requests, {len(reqs) - len(new)} already cached")
            steps.append(jev_step_estimate(f"test2 {name}", [req_text(r) for r in new], token_factor=JEV_TOKEN_FACTOR))
        print_dry_run(steps)
        return

    for name, reqs in sets:
        print(f"Running {len(reqs)} {name} requests...")
        results = run_requests(reqs, JEV_MODEL, cache, max_workers=16)
        rows = to_rows(results, None, None)
        for r in rows:
            res = results[f"{r['qid']}|{r['doc']}|{r['kind']}"]
            r["value"] = noul_of(res, "relevance") if r["kind"] == "noul" else score_of(res, "relevance")
        if name == "batched":
            rows = []
            for rid, res in results.items():
                qid, _, kind = rid.split("|")
                for i, d in enumerate(base[qid]):
                    a = (res.get("raw_response") or {}).get("answers", {}).get(f"d{i}") if res.get("ok") else None
                    rows.append({"qid": qid, "doc": d, "kind": kind, "value": (a or {}).get(kind), "tokens": input_tokens_of(res) if i == 0 else 0,
                                 "latency_ms": res.get("latency_ms") if i == 0 else 0, "model": res.get("model"), "error": res.get("error")})
        json.dump(rows, open(BATCH_FILE if name == "batched" else PAIRS_FILE, "w"))
        print(f"  {name}: {len(rows)} rows, errors {sum(1 for r in rows if r['error'])}")


if __name__ == "__main__":
    main()
