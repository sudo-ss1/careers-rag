"""CLI: ask the corpus a question.

    ./.venv/bin/python ask.py "remote python roles in poland"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from careers_rag.answer import Answerer                    # noqa: E402
from careers_rag.corpus import build_chunks, load_snapshot  # noqa: E402
from careers_rag.embed import EmbeddingStore, load_env      # noqa: E402
from careers_rag.retrieve import Retriever                  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]) or "backend engineering roles working with LLMs"
    load_env(ROOT)
    import os

    postings = load_snapshot(ROOT / "data/raw/snapshot-2026-09-05")
    chunks = build_chunks(postings)
    by_id = {c.chunk_id: c for c in chunks}
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]

    store = EmbeddingStore(os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                           ROOT / "data/index")
    vectors = store.embed(texts, label="chunks")
    r = Retriever(ids, texts, store=store, vectors=vectors)

    hits = r.search(question, mode="hybrid", k=8)
    top_chunks = [by_id[h.chunk_id] for h in hits]
    dense_top = r.search(question, mode="dense", k=1)
    ans = Answerer().answer(question, top_chunks, dense_top[0].score if dense_top else 0.0)

    print(f"\nQ: {question}\n")
    print(ans.text)
    if ans.abstained:
        print(f"\n[abstained: {ans.reason}, top_score={ans.top_score:.3f}]")
    else:
        print(f"\n[cited: {', '.join(ans.citations)} | top_score={ans.top_score:.3f}]")
        for c in top_chunks[:4]:
            if c.posting.req_id in ans.citations:
                print(f"  {c.posting.req_id}  {c.posting.title}  ({c.posting.location_str})")
                print(f"     {c.posting.source_url}")


if __name__ == "__main__":
    main()
