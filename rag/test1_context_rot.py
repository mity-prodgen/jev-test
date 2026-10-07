#!/usr/bin/env python3
"""Test 1: context rot ("no chunking to tune").

50 SQuAD v1.1 dev questions. For each, documents of nominal 500 / 2K / 8K / 30K Jev tokens: the
answering paragraph plus padding paragraphs from other articles, with the answering paragraph at the
start, middle or end. Matching negatives: same padding, the answering paragraph swapped for another
unrelated paragraph. One Noul relevance question per (question, document).

    python rag/test1_context_rot.py                 # dry-run: build documents, print estimate
    python rag/test1_context_rot.py --live          # calibrate token factor (4 calls), then run all
    python rag/test1_context_rot.py --analyze       # AUC per length and position, chart, RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict

from common import (CACHE, DATA, JEV_MODEL, RELEVANCE_NOUL, RESULTS, SEED, pair_request,
                    update_results_section)
from jev_harness.cache import DiskCache
from jev_harness.cost import JEV_TOKEN_FACTOR, estimate_tokens, jev_step_estimate, print_dry_run
from jev_harness.metrics import auc_bootstrap, auc_diff_bootstrap
from jev_harness.pairs import input_tokens_of, noul_of, run_requests

LENGTHS = [500, 2000, 8000, 30000]  # nominal Jev tokens (document only)
POSITIONS = ["start", "middle", "end"]
N_QUESTIONS = 50
MAX_PER_ARTICLE = 2
JEV_CAP = 32000  # docs: "32k tokens for state plus the longest question"
PRONOUNS = {"he", "she", "they", "it", "his", "her", "their", "its", "this", "these", "those", "him", "them"}
CALIB_FILE = RESULTS / "test1_calibration.json"
ROWS_FILE = RESULTS / "test1_rows.json"


def load_paragraphs():
    data = json.load(open(DATA / "squad" / "dev-v1.1.json"))["data"]
    paras = []
    for a in data:
        for i, p in enumerate(a["paragraphs"]):
            paras.append({"title": a["title"], "pid": f"{a['title']}#{i}", "context": p["context"],
                          "tk": estimate_tokens(p["context"]), "qas": p["qas"]})
    return paras


def select_questions(paras):
    rng = random.Random(SEED)
    cands = []
    for p in paras:
        if not 80 <= p["tk"] <= 220:
            continue
        for qa in p["qas"]:
            q = qa["question"].strip()
            words = re.findall(r"[a-z0-9']+", q.lower())
            if len(words) < 6 or PRONOUNS & set(words):
                continue
            cands.append({"qid": qa["id"], "question": q, "answers": sorted({a["text"] for a in qa["answers"]}),
                          "title": p["title"], "pid": p["pid"], "context": p["context"], "tk": p["tk"]})
    rng.shuffle(cands)
    per_title, seen, picked = defaultdict(int), set(), []
    for c in cands:
        if per_title[c["title"]] >= MAX_PER_ARTICLE or c["pid"] in seen:
            continue
        per_title[c["title"]] += 1
        seen.add(c["pid"])
        picked.append(c)
        if len(picked) == N_QUESTIONS:
            break
    return picked


def trim_last(pads, pad_target):
    total = sum(estimate_tokens(p) for p in pads)
    while pads and total - pad_target > 20:
        sents = re.split(r"(?<=[.!?])\s+", pads[-1])
        if len(sents) <= 1:
            break
        pads[-1] = " ".join(sents[:-1])
        total = sum(estimate_tokens(p) for p in pads)
    return pads


def build_docs(questions, paras, factor):
    docs = []
    for q in questions:
        answers_l = [a.lower() for a in q["answers"]]
        pool = [p for p in paras if p["title"] != q["title"] and 60 <= p["tk"] <= 300
                and not any(a in p["context"].lower() for a in answers_l)]
        for L in LENGTHS:
            rng = random.Random(f"{SEED}-{q['qid']}-{L}")
            cand = pool[:]
            rng.shuffle(cand)
            pad_target = round(L / factor) - q["tk"]
            pads, used, total = [], set(), 0
            for p in cand:
                if total >= pad_target and len(pads) >= 2:
                    break
                pads.append(p["context"]); used.add(p["pid"]); total += p["tk"]
            pads = trim_last(pads, pad_target)
            repl = min((p for p in cand if p["pid"] not in used), key=lambda p: abs(p["tk"] - q["tk"]))["context"]
            tks = [estimate_tokens(p) for p in pads]
            bounds = [sum(tks[:i]) for i in range(len(pads) + 1)]
            mid = min(range(len(bounds)), key=lambda i: abs(bounds[i] - bounds[-1] / 2))
            arrangements = {"start": [q["context"]] + pads, "end": pads + [q["context"]],
                            "middle": pads[:mid] + [q["context"]] + pads[mid:], "neg": pads[:mid] + [repl] + pads[mid:]}
            for pos, paragraphs in arrangements.items():
                doc = "\n\n".join(paragraphs)
                docs.append({"id": f"{q['qid']}|L{L}|{pos}", "qid": q["qid"], "question": q["question"],
                             "length": L, "position": pos, "document": doc, "tk": estimate_tokens(doc)})
    return docs


def requests_for(docs):
    return [pair_request(d["id"], d["question"], d["document"], "noul") for d in docs]


def req_text(r):
    return json.dumps(r.state) + json.dumps(r.questions)


def load_factor():
    if CALIB_FILE.exists():
        return json.load(open(CALIB_FILE))["factor"]
    return JEV_TOKEN_FACTOR


def calibrate(cache, paras, questions):
    """4 live calls (one per tier): measure real Jev tokens vs tiktoken, check the cap."""
    docs = [d for d in build_docs(questions[:1], paras, JEV_TOKEN_FACTOR) if d["position"] == "start"]
    reqs = requests_for(docs)
    out = {"tiers": []}
    for d, r in zip(docs, reqs):
        res = run_requests([r], JEV_MODEL, cache, max_workers=1, progress_every=0)[r.id]
        tokens = input_tokens_of(res)
        tk_req = estimate_tokens(req_text(r))
        out["tiers"].append({"length": d["length"], "doc_tk": d["tk"], "request_tk": tk_req, "jev_input_tokens": tokens,
                             "ratio": (tokens / tk_req) if tokens else None, "ok": res["ok"], "error": res["error"]})
        print(f"  tier {d['length']}: tiktoken request {tk_req} -> Jev {tokens} tokens ok={res['ok']} {res['error'] or ''}")
        if not res["ok"]:
            break
    ratios = [t["ratio"] for t in out["tiers"] if t["ratio"]]
    out["factor"] = out["tiers"][-1]["ratio"] if ratios else JEV_TOKEN_FACTOR  # largest tier is document-dominated
    json.dump(out, open(CALIB_FILE, "w"), indent=2)
    return out["factor"]


def summarize_build(docs, factor):
    print(f"Documents built with token factor {factor:.3f} (Jev tokens per tiktoken token)")
    print(f"{'length':>7} {'n':>5} {'mean tk':>9} {'est Jev tk':>11} {'max est':>8}")
    worst = 0
    for L in LENGTHS:
        ds = [d for d in docs if d["length"] == L]
        mean_tk = statistics.mean(d["tk"] for d in ds)
        mx = max(d["tk"] for d in ds) * factor + 250
        worst = max(worst, mx)
        print(f"{L:>7} {len(ds):>5} {mean_tk:>9.0f} {mean_tk * factor:>11.0f} {mx:>8.0f}")
    print(f"Largest estimated request (document + question): {worst:.0f} tokens; cap {JEV_CAP}")


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(ROWS_FILE))
    by = defaultdict(dict)
    toks = defaultdict(list)
    for r in rows:
        if r["p"] is None:
            continue
        by[(r["length"], r["position"])][r["qid"]] = r["p"]
        toks[r["length"]].append(r["input_tokens"] or 0)
    qids = sorted({r["qid"] for r in rows if r["p"] is not None})

    def lists(L, positions):
        pos = [[by[(L, p)][q] for p in positions if q in by[(L, p)]] for q in qids]
        neg = [[by[(L, "neg")][q]] if q in by[(L, "neg")] else [] for q in qids]
        keep = [i for i in range(len(qids)) if pos[i] and neg[i]]
        return [pos[i] for i in keep], [neg[i] for i in keep]

    table, series = [], {p: [] for p in POSITIONS}
    lines = ["| Length (mean Jev tokens) | Position | mean p (positives) | mean p (negatives) | AUC [95% CI] |", "|---|---|---|---|---|"]
    for L in LENGTHS:
        for pos in POSITIONS:
            P, N = lists(L, [pos])
            a, lo, hi = auc_bootstrap(P, N, seed=SEED)
            mp = statistics.mean(x for ps in P for x in ps)
            mn = statistics.mean(x for ns in N for x in ns)
            series[pos].append((a, lo, hi))
            table.append({"length": L, "position": pos, "mean_p_pos": mp, "mean_p_neg": mn, "auc": a, "lo": lo, "hi": hi})
            lines.append(f"| {L} ({statistics.mean(toks[L]):.0f}) | {pos} | {mp:.3f} | {mn:.3f} | {a:.3f} [{lo:.3f}, {hi:.3f}] |")

    P0, N0 = lists(LENGTHS[0], POSITIONS)
    drop = None
    diff_lines = []
    for L in LENGTHS[1:]:
        P, N = lists(L, POSITIONS)
        d, lo, hi = auc_diff_bootstrap(P, N, P0, N0, seed=SEED)
        diff_lines.append(f"- {L}: AUC change vs {LENGTHS[0]} = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if drop is None and hi < 0:
            drop = L
    verdict = (f"Separation first drops (95% CI of the AUC change vs the shortest tier excludes 0) at the {drop}-token tier."
               if drop else "No tier showed an AUC drop whose 95% CI excludes 0 versus the shortest tier.")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(LENGTHS)))
    for pos, c in zip(POSITIONS, ["#2a78d6", "#e34948", "#898781"]):
        a = [s[0] for s in series[pos]]
        ax.plot(xs, a, marker="o", color=c, label=f"answer at {pos}")
        ax.fill_between(xs, [s[1] for s in series[pos]], [s[2] for s in series[pos]], color=c, alpha=0.12)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{L}\n({statistics.mean(toks[L]):.0f} Jev tok)" for L in LENGTHS])
    ax.set_xlabel("Document length (nominal; mean measured Jev input tokens)")
    ax.set_ylabel("AUC (p separates answering from non-answering documents)")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, color="#e1e0d9")
    ax.set_title("Test 1: AUC vs document length, one line per position (95% bootstrap bands)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "test1_auc_vs_length.png", dpi=150)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for pos, c in zip(POSITIONS, ["#2a78d6", "#e34948", "#898781"]):
        ax2.plot(xs, [t["mean_p_pos"] for t in table if t["position"] == pos], marker="o", color=c, label=f"answering document, answer at {pos}")
    ax2.plot(xs, [next(t["mean_p_neg"] for t in table if t["length"] == L) for L in LENGTHS], marker="s", color="black", linestyle="--", label="non-answering document")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{L}\n({statistics.mean(toks[L]):.0f} Jev tok)" for L in LENGTHS])
    ax2.set_xlabel("Document length (nominal; mean measured Jev input tokens)")
    ax2.set_ylabel("Mean p (probability the document answers the question)")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, color="#e1e0d9")
    ax2.set_title("Test 1: mean p by length. The margin narrows slightly; the ranking does not change")
    ax2.legend(loc="center right", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(RESULTS / "test1_mean_p_vs_length.png", dpi=150)
    json.dump(table, open(RESULTS / "test1_table.json", "w"), indent=2)
    body = "\n".join(lines) + "\n\n" + "\n".join(diff_lines) + "\n\n" + verdict + f"\n\n{len(qids)} questions; model {rows[0]['model']}; seed {SEED}."
    update_results_section("## Test 1: context rot", body)
    print(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
        return

    paras = load_paragraphs()
    questions = select_questions(paras)
    cache = DiskCache(CACHE, "rag")
    print(f"{len(questions)} questions from {len({q['title'] for q in questions})} articles")

    if args.live and not CALIB_FILE.exists():
        print("Calibrating token factor with 4 live calls (one per length tier)...")
        calibrate(cache, paras, questions)
    factor = load_factor()
    docs = build_docs(questions, paras, factor)
    reqs = requests_for(docs)
    summarize_build(docs, factor)

    if not args.live:
        print_dry_run([jev_step_estimate("test1 (jev noul)", [req_text(r) for r in reqs], token_factor=factor)])
        print(f"Calibration (4 calls, about $0.002) runs first on --live. Rate limit: 100K tokens/s, 80 req/s.")
        return

    results = run_requests(reqs, JEV_MODEL, cache, max_workers=8)
    meta = {d["id"]: d for d in docs}
    rows = []
    for rid, res in results.items():
        d = meta[rid]
        rows.append({"id": rid, "qid": d["qid"], "length": d["length"], "position": d["position"],
                     "label": d["position"] != "neg", "p": noul_of(res, "relevance"), "model": res.get("model"),
                     "input_tokens": input_tokens_of(res), "latency_ms": res.get("latency_ms"),
                     "cached": res.get("cached"), "error": res.get("error")})
    json.dump(rows, open(ROWS_FILE, "w"), indent=1)
    print(f"{len(rows)} rows -> {ROWS_FILE}; errors: {sum(1 for r in rows if r['error'])}")


if __name__ == "__main__":
    main()
