"""Grounded answering with a two-layer abstention gate.

The product claim this whole repo exists to support: **a careers assistant that
says "I don't know" is more useful than one that invents a job.** A hallucinated
requisition wastes a candidate's application and erodes trust in the whole
channel, and it is the failure mode most retrieval demos quietly ignore.

Abstention is enforced twice, because either layer alone leaks:

  1. RETRIEVAL GATE -- if the best chunk's similarity is below a calibrated
     threshold, no model call happens at all. Cheaper, deterministic, and
     immune to a persuasive-sounding generation.
  2. GROUNDING CHECK -- the model must cite requisition ids, and any id it
     cites that wasn't in the retrieved context is dropped. An answer left with
     no citations is converted into an abstention.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import openai

from careers_rag.corpus import Chunk

SYSTEM = """You answer questions about open job requisitions using ONLY the postings provided.

Rules:
- Use only the CONTEXT. Never use outside knowledge about the company.
- Cite every role you mention by its Req ID in square brackets, e.g. [1440127].
- If the context does not contain roles matching the question, reply exactly:
  NO_MATCH
- Do not invent titles, locations, requisition ids, salaries, or dates.
- Be concise: at most 120 words. Mention at most 4 roles.
"""

REQ_ID_RE = re.compile(r"\[(\d{6,8})\]")


@dataclass
class Answer:
    text: str
    abstained: bool
    reason: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    top_score: float = 0.0
    usage: dict = field(default_factory=dict)


def build_context(chunks: list[Chunk], limit: int = 6) -> str:
    """Highest-ranked chunk FIRST.

    Position matters: models attend most reliably to the beginning and end of a
    long context, so burying the best evidence in the middle measurably hurts
    ('lost in the middle'). Ranked order is not cosmetic.
    """
    parts = []
    for c in chunks[:limit]:
        p = c.posting
        parts.append(
            f"--- Req ID {p.req_id} ---\n"
            f"Title: {p.title}\nLocation: {p.location_str}\n"
            f"Category: {p.category} | {p.employment_type} | {p.remote_type}\n"
            f"Section ({c.section}): {c.body[:900]}"
        )
    return "\n\n".join(parts)


class Answerer:
    def __init__(self, model: str | None = None, min_score: float = 0.30):
        self._client = openai.OpenAI(timeout=45.0, max_retries=3)
        self._model = model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        # Calibrated on the golden set -- see eval/calibrate_threshold.py.
        # Not a guessed constant: it is chosen to separate answerable from
        # out-of-corpus queries, and it is re-derived whenever the corpus or
        # embedding model changes.
        self._min_score = min_score

    def answer(self, question: str, chunks: list[Chunk], top_score: float) -> Answer:
        retrieved_reqs = [c.posting.req_id for c in chunks[:6]]

        # --- layer 1: retrieval gate -----------------------------------------
        if not chunks or top_score < self._min_score:
            return Answer(
                text=("I don't have any open roles matching that in the current "
                      "posting snapshot."),
                abstained=True, reason="below_retrieval_threshold",
                retrieved=retrieved_reqs, top_score=top_score,
            )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"CONTEXT:\n{build_context(chunks)}\n\nQUESTION: {question}"},
            ],
            temperature=0,
            max_tokens=400,
        )
        raw = (resp.choices[0].message.content or "").strip()
        usage = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }

        # --- model's own abstention ------------------------------------------
        if "NO_MATCH" in raw:
            return Answer(
                text=("I don't have any open roles matching that in the current "
                      "posting snapshot."),
                abstained=True, reason="model_no_match",
                retrieved=retrieved_reqs, top_score=top_score, usage=usage,
            )

        # --- layer 2: grounding check ----------------------------------------
        cited = REQ_ID_RE.findall(raw)
        allowed = set(retrieved_reqs)
        hallucinated = [c for c in cited if c not in allowed]
        grounded = [c for c in cited if c in allowed]

        if hallucinated:
            # A cited id that was never retrieved is fabricated by definition.
            # Strip it rather than trusting prose that references it.
            for bad in hallucinated:
                raw = raw.replace(f"[{bad}]", "")

        if not grounded:
            return Answer(
                text=("I don't have any open roles matching that in the current "
                      "posting snapshot."),
                abstained=True, reason="no_grounded_citation",
                retrieved=retrieved_reqs, top_score=top_score, usage=usage,
            )

        return Answer(
            text=raw, abstained=False, citations=grounded,
            retrieved=retrieved_reqs, top_score=top_score, usage=usage,
        )
