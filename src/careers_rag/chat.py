"""Multi-turn chat with LLM query condensation.

The problem a single-shot RAG pipeline cannot handle: turn 2 is often
meaningless on its own.

    user: software engineer jobs
    user: in Bangalore          <- retrieving on this alone returns Bangalore
                                   sales roles, finance roles, everything

The fix is **condensation**: before retrieving, rewrite the latest turn into a
standalone query using the conversation so far. Retrieval then sees
"software engineer jobs in Bangalore" and behaves correctly.

Two details that matter and are easy to get wrong:

  * Condense against the *rewritten* history, not the raw turns. Otherwise
    ambiguity compounds across a long session.
  * Condensation is a rewrite, never an answer. A model that starts answering
    here will invent roles before retrieval has run, and the grounding check
    downstream cannot catch what was never retrieved.

The web version of this page approximates condensation with deterministic
slot-filling, because a static page cannot hold an API key. This module is the
real thing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from careers_rag.answer import Answerer, diversify  # noqa: E402
from careers_rag.corpus import Chunk, build_chunks, load_snapshot  # noqa: E402
from careers_rag.embed import EmbeddingStore, load_env  # noqa: E402
from careers_rag.retrieve import Retriever  # noqa: E402

CONDENSE_SYSTEM = """Rewrite the user's latest message into a single standalone job-search query.

Rules:
- Carry forward constraints from earlier turns (role, seniority, location, work type)
  unless the latest message replaces them.
- A message naming only a location or work type NARROWS the previous query — keep the role.
- A message naming a different role REPLACES the role but keeps location and work type.
- Output ONLY the rewritten query. No preamble, no explanation, no answer.
- If the latest message is already standalone, return it unchanged."""


@dataclass
class Turn:
    user: str
    standalone: str
    answer: str
    abstained: bool
    citations: list[str] = field(default_factory=list)


class ChatSession:
    def __init__(self, retriever: Retriever, chunks: dict[str, Chunk],
                 model: str | None = None, history_turns: int = 6):
        self._r = retriever
        self._by_id = chunks
        self._client = openai.OpenAI(timeout=45.0, max_retries=3)
        self._model = model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self._answerer = Answerer(model=self._model)
        self._history_turns = history_turns
        self.turns: list[Turn] = []

    def _condense(self, message: str) -> str:
        if not self.turns:
            return message                      # first turn is standalone by definition

        # Feed back the CONDENSED queries, not the raw user text, so ambiguity
        # does not compound across a long conversation.
        history = "\n".join(
            f"user: {t.standalone}" for t in self.turns[-self._history_turns:]
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": CONDENSE_SYSTEM},
                {"role": "user",
                 "content": f"Conversation so far:\n{history}\n\nLatest message: {message}"},
            ],
            temperature=0,
            max_tokens=80,
        )
        rewritten = (resp.choices[0].message.content or "").strip().strip('"')
        # A condenser that returns something implausible (empty, or a paragraph)
        # has misunderstood its job; fall back to the raw turn rather than
        # retrieving on garbage.
        if not rewritten or len(rewritten) > 220:
            return message
        return rewritten

    def ask(self, message: str) -> Turn:
        standalone = self._condense(message)

        hits = self._r.search(standalone, mode="hybrid", k=10)
        chunks = diversify([self._by_id[h.chunk_id] for h in hits])
        dense = self._r.search(standalone, mode="dense", k=1)
        ans = self._answerer.answer(standalone, chunks, dense[0].score if dense else 0.0)

        turn = Turn(user=message, standalone=standalone, answer=ans.text,
                    abstained=ans.abstained, citations=ans.citations)
        self.turns.append(turn)
        return turn


def build_session(snapshot: str = "data/raw/snapshot-2026-09-05") -> ChatSession:
    load_env(ROOT)
    postings = load_snapshot(ROOT / snapshot)
    chunks = build_chunks(postings)
    store = EmbeddingStore(os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                           ROOT / "data/index")
    vectors = store.embed([c.text for c in chunks], label="chunks")
    r = Retriever([c.chunk_id for c in chunks], [c.text for c in chunks],
                  store=store, vectors=vectors)
    return ChatSession(r, {c.chunk_id: c for c in chunks})


def main() -> None:
    session = build_session()
    print("\nMulti-turn job search. Ctrl-C to exit.\n"
          "Try:  software engineer jobs  ->  in Bangalore  ->  remote only\n")
    while True:
        try:
            msg = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        turn = session.ask(msg)
        if turn.standalone != msg:
            print(f"    [condensed: {turn.standalone}]")
        print(f"bot > {turn.answer}")
        if turn.citations:
            print(f"    [cited: {', '.join(turn.citations)}]")
        print()


if __name__ == "__main__":
    main()
