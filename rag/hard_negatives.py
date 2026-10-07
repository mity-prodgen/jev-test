#!/usr/bin/env python3
"""Hard negatives: on-topic documents that do NOT answer the query (SciFact claims).

Claude drafts about 100 pairs into a CSV and the script STOPS. You review the CSV by hand and set
`status` to `approved` (or `rejected`, or edit the text first). Jev is only called for approved rows.

    python rag/hard_negatives.py draft              # dry-run: show the drafting estimate
    python rag/hard_negatives.py draft --live       # Claude drafts -> rag/results/hard_negatives_review.csv, then stop
    python rag/hard_negatives.py run                # dry-run: count approved rows and Jev calls
    python rag/hard_negatives.py run --live         # Jev + embeddings on approved rows; report false-positive rates
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics

import numpy as np

from common import (CACHE, DATA, EMBED_MODEL, JEV_MODEL, RESULTS, SEED, doc_text, pair_request, update_results_section)
from jev_harness.cache import DiskCache
from jev_harness.cost import JEV_TOKEN_FACTOR, estimate_tokens, jev_step_estimate, print_dry_run
from jev_harness.llm_judge import PRICING
from jev_harness.metrics import auc, bootstrap_ci
from jev_harness.pairs import noul_of, run_requests, is_cached

N_PAIRS = 100
BATCH = 10
DRAFT_MODEL = "claude-sonnet-5-5"
REVIEW_CSV = RESULTS / "hard_negatives_review.csv"
TYPES = {
    "different_aspect": "same entities and topic as the claim, but the text reports a different question (for example prevalence, "
                        "mechanism, methods or epidemiology) and says nothing that supports or refutes the claim",
    "near_miss": "same topic, but about a different population, dose, species, time point or condition than the one in the claim, "
                 "so it does not address the claim as written",
    "unrelated_negation": "same entities, with at least one negated statement (for example 'did not differ', 'no significant "
                          "change') about a DIFFERENT outcome than the claim. It must not state the opposite of the claim",
    "background": "background or review-style text that mentions the claim's topic and cites no evidence for or against the claim",
}
SYSTEM = ("You write test documents for a retrieval evaluation. Each document must look like a short scientific abstract "
          "(a title line, then 90 to 150 words of plain scientific prose with specific details). Each document must be ON TOPIC for "
          "the claim it is paired with, and must NOT contain evidence that supports or refutes that claim. Do not state the claim "
          "or its opposite. Return only a JSON array.")


def pick_queries():
    from test2_rerank import load_scifact
    corpus, queries, qrels, test_q = load_scifact()
    rng = random.Random(SEED)
    chosen = rng.sample(test_q, N_PAIRS)
    types = list(TYPES)
    return corpus, queries, qrels, [(q, types[i % len(types)]) for i, q in enumerate(chosen)]


def draft_prompt(items, queries):
    lines = []
    for q, t in items:
        lines.append(f'- id "{q}", type "{t}" ({TYPES[t]}). Claim: {queries[q]}')
    return ("Write one hard-negative document for each item below.\n\n" + "\n".join(lines) +
            '\n\nReturn a JSON array. Each element: {"id": <id>, "type": <type>, "title": <title>, "text": <90-150 words>, '
            '"why_not_answer": <one sentence on why the document does not support or refute the claim>}.')


def draft(live):
    corpus, queries, qrels, items = pick_queries()
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    in_tok = sum(estimate_tokens(SYSTEM + draft_prompt(b, queries)) for b in batches)
    out_tok = N_PAIRS * 260
    pin, pout = PRICING[DRAFT_MODEL]
    est = (in_tok * pin + out_tok * pout) / 1e6
    print(f"{len(batches)} Claude calls ({DRAFT_MODEL}); about {in_tok:,} input and {out_tok:,} output tokens "
          f"(+ unknown thinking tokens at low effort) -> about ${est:.2f}; likely up to ${est * 1.8:.2f}")
    if not live:
        print("Dry-run only. Pass --live to draft.")
        return
    import anthropic
    client = anthropic.Anthropic()
    rows, cost = [], 0.0
    for n, b in enumerate(batches):
        msg = client.messages.create(model=DRAFT_MODEL, max_tokens=12000, system=SYSTEM,
                                     messages=[{"role": "user", "content": draft_prompt(b, queries)}],
                                     extra_body={"output_config": {"effort": "low"}})
        text = next(x.text for x in msg.content if x.type == "text")
        data = json.loads(text[text.index("["): text.rindex("]") + 1])
        cost += (msg.usage.input_tokens * pin + msg.usage.output_tokens * pout) / 1e6
        for it in data:
            qid = str(it["id"])
            rows.append({"id": f"hn{len(rows):03d}", "query_id": qid, "query": queries[qid], "type": it["type"],
                         "title": it["title"], "text": it["text"], "why_not_answer": it["why_not_answer"], "status": "pending"})
        print(f"  batch {n + 1}/{len(batches)} done (actual cost so far ${cost:.3f})", flush=True)
    with open(REVIEW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} drafts to {REVIEW_CSV}\nSTOP: review the file, set status to approved/rejected, then run `run`.")


def approved():
    if not REVIEW_CSV.exists():
        raise SystemExit(f"{REVIEW_CSV} not found: run `draft --live` first")
    rows = list(csv.DictReader(open(REVIEW_CSV)))
    ok = [r for r in rows if r["status"].strip().lower() == "approved"]
    print(f"{len(rows)} drafted; approved {len(ok)}, rejected {sum(r['status'].strip().lower() == 'rejected' for r in rows)}, "
          f"pending {sum(r['status'].strip().lower() == 'pending' for r in rows)}")
    if not ok:
        raise SystemExit("No approved rows. Nothing to run.")
    return ok


def run(live):
    from test2_rerank import load_scifact, stage1, embed_model, fmt_query
    ok = approved()
    corpus, queries, qrels, test_q = load_scifact()
    gold = {r["query_id"]: sorted(qrels[r["query_id"]])[0] for r in ok}
    reqs = []
    for r in ok:
        reqs.append(pair_request(f"neg|{r['id']}", r["query"], doc_text(r["title"], r["text"]), "noul"))
        reqs.append(pair_request(f"pos|{r['query_id']}", r["query"], corpus[gold[r["query_id"]]], "noul"))
    reqs = list({r.id: r for r in reqs}.values())
    cache = DiskCache(CACHE, "rag")
    new = [r for r in reqs if not is_cached(cache, JEV_MODEL, r)]
    print(f"{len(reqs)} Jev requests ({len(reqs) - len(new)} cached, {len(new)} new)")
    if not live:
        print_dry_run([jev_step_estimate("hard negatives", [json.dumps(r.state) + json.dumps(r.questions) for r in new], token_factor=JEV_TOKEN_FACTOR)])
        return
    res = run_requests(reqs, JEV_MODEL, cache, max_workers=16)
    # embeddings
    model = embed_model()
    ids = list(corpus)
    D = np.load(DATA / "scifact" / "emb_qwen3_06b_docs.npy")
    idx = {d: i for i, d in enumerate(ids)}
    qv = model.encode([fmt_query(r["query"]) for r in ok], normalize_embeddings=True)
    nv = model.encode([doc_text(r["title"], r["text"]) for r in ok], normalize_embeddings=True)
    pos_cos = [float(qv[i] @ D[idx[gold[r["query_id"]]]]) for i, r in enumerate(ok)]
    neg_cos = [float(qv[i] @ nv[i]) for i in range(len(ok))]
    pos_p = [noul_of(res[f"pos|{r['query_id']}"], "relevance") for r in ok]
    neg_p = [noul_of(res[f"neg|{r['id']}"], "relevance") for r in ok]
    keep = [i for i in range(len(ok)) if pos_p[i] is not None and neg_p[i] is not None]
    pos_p, neg_p = [pos_p[i] for i in keep], [neg_p[i] for i in keep]
    pos_cos, neg_cos = [pos_cos[i] for i in keep], [neg_cos[i] for i in keep]
    types = [ok[i]["type"] for i in keep]

    tpr = statistics.mean(p >= 0.5 for p in pos_p)
    fp_j = [1.0 if p >= 0.5 else 0.0 for p in neg_p]
    k = max(1, round(tpr * len(pos_cos)))
    thr = sorted(pos_cos, reverse=True)[k - 1]
    fp_e = [1.0 if c >= thr else 0.0 for c in neg_cos]
    fj, flo, fhi = bootstrap_ci(fp_j, seed=SEED)
    fe, elo, ehi = bootstrap_ci(fp_e, seed=SEED)
    lines = [f"{len(keep)} approved hard-negative pairs, each paired with the query's real relevant document. Embedding: {EMBED_MODEL}.", "",
             f"At p >= 0.5 Jev Noul accepts {tpr:.1%} of the real relevant documents. The embedding threshold (cosine >= {thr:.3f}) is set to accept the same share, so false-positive rates compare at equal recall.", "",
             "| Method | false-positive rate on hard negatives [95% CI] | AUC (relevant vs hard negative) |", "|---|---|---|",
             f"| Jev Noul (p >= 0.5) | {fj:.1%} [{flo:.1%}, {fhi:.1%}] | {auc(pos_p, neg_p):.3f} |",
             f"| Embedding cosine (equal recall) | {fe:.1%} [{elo:.1%}, {ehi:.1%}] | {auc(pos_cos, neg_cos):.3f} |", "",
             "| Hard-negative type | n | Jev false positives | Embedding false positives |", "|---|---|---|---|"]
    for t in TYPES:
        ii = [i for i, x in enumerate(types) if x == t]
        if ii:
            lines.append(f"| {t} | {len(ii)} | {sum(fp_j[i] for i in ii):.0f} ({statistics.mean(fp_j[i] for i in ii):.0%}) | {sum(fp_e[i] for i in ii):.0f} ({statistics.mean(fp_e[i] for i in ii):.0%}) |")
    body = "\n".join(lines)
    update_results_section("## Test 2b: hard negatives", body)
    json.dump({"pos_p": pos_p, "neg_p": neg_p, "pos_cos": pos_cos, "neg_cos": neg_cos, "types": types, "threshold": thr}, open(RESULTS / "hard_negatives_scores.json", "w"))
    print(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["draft", "run"])
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    draft(a.live) if a.command == "draft" else run(a.live)
