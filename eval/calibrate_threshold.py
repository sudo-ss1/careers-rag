"""Choose the abstention threshold from data instead of guessing it.

Answerable queries should score high, out-of-corpus queries should score low.
The threshold is picked to maximise Youden's J (sensitivity + specificity - 1)
across the golden set, and the full sweep is reported so the tradeoff is
visible rather than hidden behind one number.

Re-run this whenever the corpus or the embedding model changes -- a threshold
calibrated against a different corpus is just a magic constant.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careers_rag.corpus import build_chunks, load_snapshot  # noqa: E402
from careers_rag.embed import EmbeddingStore, load_env  # noqa: E402
from careers_rag.retrieve import Retriever  # noqa: E402


def main() -> None:
    load_env(ROOT)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")

    postings = load_snapshot(ROOT / "data/raw/snapshot-2026-09-05")
    chunks = build_chunks(postings)
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]

    store = EmbeddingStore(os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                           ROOT / "data/index")
    vectors = store.embed(texts, label="chunks")
    r = Retriever(ids, texts, store=store, vectors=vectors)

    golden = json.loads((ROOT / "eval/golden.json").read_text())

    # Gate on raw DENSE cosine, not the fused RRF score: RRF scores are
    # rank-derived and bounded by 1/(k+1), so they carry no notion of "how
    # similar is this really" -- which is exactly what a threshold needs.
    scored = []
    for q in golden:
        hits = r.search(q["query"], mode="dense", k=5)
        scored.append({
            "query": q["query"],
            "answerable": bool(q["relevant"]),
            "top": hits[0].score if hits else 0.0,
        })

    pos = [s["top"] for s in scored if s["answerable"]]
    neg = [s["top"] for s in scored if not s["answerable"]]
    print(f"answerable  n={len(pos)}  min={min(pos):.3f} mean={sum(pos)/len(pos):.3f}")
    print(f"out-of-corpus n={len(neg)}  max={max(neg):.3f} mean={sum(neg)/len(neg):.3f}")

    best, sweep = None, []
    for i in range(20, 71):
        t = i / 100
        tp = sum(1 for v in pos if v >= t)
        fn = len(pos) - tp
        tn = sum(1 for v in neg if v < t)
        fp = len(neg) - tn
        sens = tp / max(1, len(pos))
        spec = tn / max(1, len(neg))
        j = sens + spec - 1
        sweep.append({"threshold": t, "sensitivity": sens, "specificity": spec,
                      "youden_j": j, "tp": tp, "fn": fn, "tn": tn, "fp": fp})
        if best is None or j > best["youden_j"]:
            best = sweep[-1]

    print(f"\nbest threshold = {best['threshold']:.2f}  "
          f"(answers {best['sensitivity']:.0%} of answerable, "
          f"abstains on {best['specificity']:.0%} of out-of-corpus)")

    out = ROOT / "reports/threshold-calibration.json"
    out.write_text(json.dumps({"scored": scored, "sweep": sweep, "best": best}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
