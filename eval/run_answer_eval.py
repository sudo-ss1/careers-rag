"""End-to-end answer evaluation: does the system answer when it should, abstain
when it should, and cite only things it actually retrieved?

Retrieval metrics measure the ceiling. This measures what a user experiences.
Two failure modes are tracked separately because they cost different things:

  * FALSE ABSTENTION -- refusing a question the corpus can answer. Looks broken,
    erodes trust, and is invisible unless you measure it.
  * HALLUCINATED CITATION -- naming a requisition that was never retrieved. The
    expensive one: a candidate applies to a role that does not exist.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careers_rag.answer import REQ_ID_RE, Answerer, diversify  # noqa: E402
from careers_rag.corpus import build_chunks, load_snapshot     # noqa: E402
from careers_rag.embed import EmbeddingStore, load_env         # noqa: E402
from careers_rag.retrieve import Retriever                     # noqa: E402


def main() -> None:
    load_env(ROOT)
    postings = load_snapshot(ROOT / "data/raw/snapshot-2026-09-05")
    chunks = build_chunks(postings)
    by_chunk = {c.chunk_id: c for c in chunks}
    req_of = {p.job_id: p.req_id for p in postings}

    store = EmbeddingStore(os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                           ROOT / "data/index")
    vectors = store.embed([c.text for c in chunks], label="chunks")
    r = Retriever([c.chunk_id for c in chunks], [c.text for c in chunks],
                  store=store, vectors=vectors)

    answerer = Answerer()
    golden = json.loads((ROOT / "eval/golden.json").read_text())

    rows = []
    for i, q in enumerate(golden, 1):
        hits = r.search(q["query"], mode="hybrid", k=10)
        top = [by_chunk[h.chunk_id] for h in hits]
        dense = r.search(q["query"], mode="dense", k=1)
        ans = answerer.answer(q["query"], top, dense[0].score if dense else 0.0)

        gold_reqs = {req_of.get(j, "") for j in q["relevant"]}
        retrieved_reqs = set(ans.retrieved)
        cited = set(ans.citations)

        rows.append({
            "query_id": q["query_id"], "family": q["family"], "subtype": q["subtype"],
            "query": q["query"], "answerable": bool(q["relevant"]),
            "abstained": ans.abstained, "reason": ans.reason,
            "citations": sorted(cited),
            "cited_not_retrieved": sorted(cited - retrieved_reqs),   # hallucinations
            "cited_in_gold": sorted(cited & gold_reqs),
            "text": ans.text[:400], "top_score": round(ans.top_score, 4),
        })
        print(f"  {i}/{len(golden)} {q['query_id'][:28]:30} "
              f"{'ABSTAIN' if ans.abstained else 'ANSWER '} ({ans.reason or 'ok'})",
              flush=True)
        time.sleep(0.1)

    answerable = [r_ for r_ in rows if r_["answerable"]]
    unanswerable = [r_ for r_ in rows if not r_["answerable"]]
    answered = [r_ for r_ in answerable if not r_["abstained"]]

    summary = {
        "n_queries": len(rows),
        "answerable": {
            "n": len(answerable),
            "answered": len(answered),
            "answer_rate": len(answered) / max(1, len(answerable)),
            "false_abstention_rate": 1 - len(answered) / max(1, len(answerable)),
            # Of answers given, how many cited at least one truly-relevant role
            "citation_hit_rate": sum(1 for r_ in answered if r_["cited_in_gold"])
                                 / max(1, len(answered)),
            # Precision of individual citations against the gold set
            "citation_precision": (
                sum(len(r_["cited_in_gold"]) for r_ in answered)
                / max(1, sum(len(r_["citations"]) for r_ in answered))
            ),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "abstained": sum(1 for r_ in unanswerable if r_["abstained"]),
            "correct_abstention_rate": sum(1 for r_ in unanswerable if r_["abstained"])
                                       / max(1, len(unanswerable)),
        },
        "hallucinated_citations": sum(len(r_["cited_not_retrieved"]) for r_ in rows),
        "abstention_reasons": dict(Counter(r_["reason"] for r_ in rows if r_["abstained"])),
    }

    out = ROOT / "reports/answer-eval.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    a, u = summary["answerable"], summary["unanswerable"]
    print(f"\nanswerable   n={a['n']:3}  answered={a['answer_rate']:.0%}  "
          f"false-abstention={a['false_abstention_rate']:.0%}  "
          f"citation-hit={a['citation_hit_rate']:.0%}  "
          f"citation-precision={a['citation_precision']:.0%}")
    print(f"unanswerable n={u['n']:3}  correctly abstained={u['correct_abstention_rate']:.0%}")
    print(f"hallucinated citations: {summary['hallucinated_citations']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
