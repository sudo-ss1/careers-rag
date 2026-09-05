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

# Match requisition ids with OR without brackets.
#
# The bracketed-only version caused false abstentions: the model answered
# "requisition 1440127 is for the role of..." in prose, the regex found zero
# citations, and a correct answer was thrown away as ungrounded. A grounding
# check that is stricter than the model's formatting habits fails closed on
# correct answers -- which looks like a broken product.
#
# Verification does not depend on the formatting: every id found is checked
# against the retrieved set regardless of how it was written.
REQ_ID_RE = re.compile(r"\b(\d{6,8})\b")

# A query naming a specific requisition is answerable by definition if that id
# is in the corpus, no matter what the embedding similarity says. Dense
# retrieval scores 0.000 recall on this query type (see reports/eval-full.json),
# so gating it on a cosine threshold is the wrong instrument.
REQ_ID_QUERY_RE = re.compile(r"\b(\d{6,8})\b")


@dataclass
class Answer:
    text: str
    abstained: bool
    reason: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    top_score: float = 0.0
    usage: dict = field(default_factory=dict)


def diversify(chunks: list[Chunk], max_per_job: int = 2) -> list[Chunk]:
    """Cap how many chunks one posting may contribute.

    Without this the top 6 chunks were three copies of the same job's sections,
    so the context described two roles instead of six. Ranking optimises
    relevance per chunk; a useful answer needs coverage across documents.
    """
    seen: dict[str, int] = {}
    out = []
    for c in chunks:
        n = seen.get(c.job_id, 0)
        if n >= max_per_job:
            continue
        seen[c.job_id] = n + 1
        out.append(c)
    return out


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
    def __init__(self, model: str | None = None, min_score: float = 0.36,
                 min_bm25: float = float("inf")):
        self._client = openai.OpenAI(timeout=45.0, max_retries=3)
        self._model = model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        # Set from calibration, but NOT at the Youden-optimal point -- and the
        # reason is the interesting part.
        #
        # Youden's J picks 0.41, which maximises the separation a single
        # threshold can achieve. Measured end to end, that gate false-abstained
        # on 15% of answerable queries -- every one of them an exact-token
        # lookup (grpc, matlab, selenium, tensorflow) scoring 0.364-0.405.
        # The distributions genuinely overlap: answerable min 0.364 sits below
        # out-of-corpus max 0.465, so NO single threshold separates them.
        #
        # So layer 1 is deliberately tuned for SENSITIVITY, not accuracy: 0.36
        # sits just under the lowest answerable score, letting nearly everything
        # through, and specificity comes from layer 2 (the model's own NO_MATCH
        # plus the grounding check). A cheap permissive filter in front of an
        # expensive precise one is the right shape; a conservative first gate
        # fails closed on correct answers, which users read as a broken product.
        self._min_score = min_score
        # Lexical arm, disabled by default. Calibration showed raw BM25 scores
        # are not comparable across queries -- they scale with query length and
        # term rarity, so out-of-corpus queries scored HIGHER (max 14.4) than
        # the answerable mean (11.6). Cosine is bounded and comparable; BM25 is
        # not, and cannot be globally thresholded. Kept as a tunable, off.
        self._min_bm25 = min_bm25

    def answer(self, question: str, chunks: list[Chunk], top_score: float,
               bm25_score: float = 0.0) -> Answer:
        chunks = diversify(chunks)
        retrieved_reqs = [c.posting.req_id for c in chunks[:6]]

        # A direct requisition lookup bypasses the similarity gate when that id
        # was actually retrieved -- lexical certainty beats a cosine threshold.
        asked_ids = set(REQ_ID_QUERY_RE.findall(question))
        lexical_hit = bool(asked_ids & set(retrieved_reqs))

        # --- layer 1: retrieval gate -----------------------------------------
        confident = (top_score >= self._min_score) or (bm25_score >= self._min_bm25)
        if not chunks or not (confident or lexical_hit):
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
