"""Record real end-to-end runs so the report can show the full pipeline.

Hybrid ranking needs a query embedding and answering needs a chat call, so
neither can run in a static page. Rather than fake them, actual runs are
executed once and their outputs — answer text, citations, retrieval scores,
abstention reason — are frozen into JSON the page renders verbatim.

Everything here is genuine output. Nothing is written by hand.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# This file lives at src/careers_rag/, so the repo root is three levels up.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from careers_rag.answer import Answerer, diversify  # noqa: E402
from careers_rag.corpus import build_chunks, load_snapshot  # noqa: E402
from careers_rag.embed import EmbeddingStore, load_env  # noqa: E402
from careers_rag.retrieve import Retriever  # noqa: E402

QUERIES = [
    ("roles working on retrieval augmented generation and vector databases",
     "Semantic query — no exact tokens to latch onto."),
    ("What is requisition 1440127 about?",
     "Exact identifier — the query type dense retrieval scores 0.000 on."),
    ("engineering roles in Poland",
     "Metadata-flavoured query, answered from posting text rather than a filter."),
    ("do you have any openings for a professional chef",
     "Nothing in the corpus supports this. The correct output is a refusal."),
]


def main() -> None:
    load_env(ROOT)
    postings = load_snapshot(ROOT / "data/raw/snapshot-2026-09-05")
    chunks = build_chunks(postings)
    by = {c.chunk_id: c for c in chunks}

    store = EmbeddingStore(os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                           ROOT / "data/index")
    vectors = store.embed([c.text for c in chunks], label="chunks")
    r = Retriever([c.chunk_id for c in chunks], [c.text for c in chunks],
                  store=store, vectors=vectors)
    answerer = Answerer()

    out = []
    for q, why in QUERIES:
        hits = r.search(q, mode="hybrid", k=10)
        top = [by[h.chunk_id] for h in hits]
        dense = r.search(q, mode="dense", k=1)
        score = dense[0].score if dense else 0.0
        ans = answerer.answer(q, top, score)

        cited = {c.posting.req_id: c.posting for c in diversify(top)}
        out.append({
            "query": q, "why": why,
            "answer": ans.text, "abstained": ans.abstained, "reason": ans.reason,
            "top_score": round(ans.top_score, 3),
            "citations": [
                {"req": rid, "title": cited[rid].title,
                 "loc": cited[rid].location_str, "url": cited[rid].source_url}
                for rid in ans.citations if rid in cited
            ],
            "retrieved": [
                {"req": c.posting.req_id, "title": c.posting.title,
                 "section": c.section}
                for c in diversify(top)[:5]
            ],
        })
        print(f"  {'ABSTAIN' if ans.abstained else 'ANSWER '}  {q[:52]}")

    dest = ROOT / "docs/data/examples.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
