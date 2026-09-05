"""Run the golden set against every available retrieval mode and report.

Output is a single JSON file per run: config, per-mode metrics, per-family
breakdown, and every individual query result. Keeping the per-query rows is
what makes the run diagnosable later -- an aggregate that dropped two points
tells you nothing about which queries caused it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402

from careers_rag.corpus import build_chunks, load_snapshot  # noqa: E402
from careers_rag.embed import EmbeddingStore, load_env  # noqa: E402
from careers_rag.retrieve import Retriever  # noqa: E402
from metrics import dedupe_to_docs, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data/raw/snapshot-2026-09-05")
    ap.add_argument("--golden", default="eval/golden.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--modes", default="bm25,dense,hybrid")
    args = ap.parse_args()

    load_env(ROOT)

    postings = load_snapshot(ROOT / args.snapshot)
    chunks = build_chunks(postings)
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]
    parent = {c.chunk_id: c.job_id for c in chunks}
    print(f"corpus: {len(postings)} postings -> {len(chunks)} chunks")

    modes = args.modes.split(",")
    store = vectors = None
    if {"dense", "hybrid"} & set(modes):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("!! OPENAI_API_KEY missing -- running lexical only")
            modes = ["bm25"]
        else:
            model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            store = EmbeddingStore(model, ROOT / "data/index")
            print(f"embedding {len(chunk_texts)} chunks with {model} ...")
            vectors = store.embed(chunk_texts, label="chunks")

    retriever = Retriever(chunk_ids, chunk_texts, store=store, vectors=vectors)
    golden = json.loads((ROOT / args.golden).read_text())

    results: dict[str, dict] = {}
    for mode in modes:
        rows, t0 = [], time.perf_counter()
        for q in golden:
            hits = retriever.search(q["query"], mode=mode, k=200)
            ranked_docs = dedupe_to_docs([parent[h.chunk_id] for h in hits])
            rows.append({
                "query_id": q["query_id"], "family": q["family"],
                "subtype": q["subtype"], "query": q["query"],
                "relevant": set(q["relevant"]),
                "ranked": ranked_docs[: args.k],
                "top_score": hits[0].score if hits else 0.0,
            })
        elapsed = time.perf_counter() - t0

        by_family: dict[str, list] = defaultdict(list)
        for r in rows:
            by_family[r["family"]].append(r)
        by_subtype: dict[str, list] = defaultdict(list)
        for r in rows:
            by_subtype[r["subtype"]].append(r)

        results[mode] = {
            "overall": summarize(rows),
            "by_family": {f: summarize(rs) for f, rs in by_family.items()},
            "by_subtype": {s: summarize(rs) for s, rs in by_subtype.items()},
            "ms_per_query": 1000 * elapsed / max(1, len(golden)),
            "queries": [
                {**{k: v for k, v in r.items() if k != "relevant"},
                 "relevant": sorted(r["relevant"]),
                 "hit@10": int(bool(set(r["ranked"][:10]) & r["relevant"]))}
                for r in rows
            ],
        }
        o = results[mode]["overall"]
        print(f"\n{mode:7} recall@10={o.get('recall@10', 0):.3f} "
              f"nDCG@10={o.get('ndcg@10', 0):.3f} MRR={o.get('mrr', 0):.3f} "
              f"({results[mode]['ms_per_query']:.1f} ms/query)")

    out = Path(args.out or ROOT / f"reports/eval-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "corpus": {"postings": len(postings), "chunks": len(chunks),
                   "snapshot": args.snapshot},
        "golden": {"queries": len(golden), "path": args.golden},
        "modes": results,
    }, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
