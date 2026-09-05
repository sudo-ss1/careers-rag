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

    # Gate on raw retriever scores, never on the fused RRF score: RRF is
    # rank-derived and bounded by 1/(k+1), so it carries no notion of "how
    # similar is this actually" -- which is the only thing a threshold needs.
    #
    # BOTH arms are calibrated, because a dense-only gate false-abstained on
    # every exact-token query (grpc, matlab, selenium, tensorflow) -- the exact
    # queries where dense is the weak retriever. The gate has to be hybrid for
    # the same reason retrieval does.
    scored = []
    for q in golden:
        dense = r.search(q["query"], mode="dense", k=5)
        lex = r.bm25.search(q["query"], k=5)
        scored.append({
            "query": q["query"],
            "answerable": bool(q["relevant"]),
            "top": dense[0].score if dense else 0.0,
            "bm25": lex[0][1] if lex else 0.0,
        })

    pos = [s["top"] for s in scored if s["answerable"]]
    neg = [s["top"] for s in scored if not s["answerable"]]
    print(f"answerable  n={len(pos)}  min={min(pos):.3f} mean={sum(pos)/len(pos):.3f}")
    print(f"out-of-corpus n={len(neg)}  max={max(neg):.3f} mean={sum(neg)/len(neg):.3f}")

    lex_pos = [s_["bm25"] for s_ in scored if s_["answerable"]]
    lex_neg = [s_["bm25"] for s_ in scored if not s_["answerable"]]
    print(f"bm25 answerable mean={sum(lex_pos)/len(lex_pos):.1f}  "
          f"out-of-corpus max={max(lex_neg):.1f}")

    # Joint sweep over the OR rule: answer if EITHER arm is confident.
    best, sweep = None, []
    for i in range(20, 71):
        td = i / 100
        for tb in [0, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
            def answers(s_):
                return s_["top"] >= td or s_["bm25"] >= tb
            tp = sum(1 for s_ in scored if s_["answerable"] and answers(s_))
            tn = sum(1 for s_ in scored if not s_["answerable"] and not answers(s_))
            sens = tp / max(1, len(pos))
            spec = tn / max(1, len(neg))
            j = sens + spec - 1
            row = {"dense_threshold": td, "bm25_threshold": tb,
                   "sensitivity": sens, "specificity": spec, "youden_j": j}
            sweep.append(row)
            # Tie-break toward higher sensitivity: a false abstention is the
            # failure users actually notice.
            if best is None or (j, sens) > (best["youden_j"], best["sensitivity"]):
                best = row

    print(f"\nbest gate: dense >= {best['dense_threshold']:.2f} OR bm25 >= {best['bm25_threshold']}  "
          f"(answers {best['sensitivity']:.0%} of answerable, "
          f"abstains on {best['specificity']:.0%} of out-of-corpus)")

    out = ROOT / "reports/threshold-calibration.json"
    out.write_text(json.dumps({"scored": scored, "sweep": sweep, "best": best}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
