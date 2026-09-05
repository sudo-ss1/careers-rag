"""Ranking metrics. Deterministic, cheap, and therefore CI-gateable.

Everything here is computed at DOCUMENT level, never chunk level. Ground truth
is a set of job ids; a retriever returns ranked chunks; chunks are collapsed to
their parent job keeping best rank. That indirection is deliberate -- it means
changing the chunking strategy does not invalidate the labels, and re-chunking
is the experiment that gets run most often.
"""

from __future__ import annotations

import math


def dedupe_to_docs(ranked_chunk_doc_ids: list[str]) -> list[str]:
    """Collapse a ranked chunk list to a ranked doc list, first occurrence wins."""
    seen, out = set(), []
    for doc in ranked_chunk_doc_ids:
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def recall_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs that made the top k.

    The ceiling metric: a generator cannot use what was never retrieved, so
    this bounds end-to-end accuracy no matter how good the model is.
    """
    if not relevant:
        return float("nan")
    return len(set(ranked_docs[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked_docs[:k]) & relevant) / k


def mrr(ranked_docs: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit. Right metric when one answer is wanted."""
    for i, doc in enumerate(ranked_docs, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """Rank-aware: a hit at position 1 is worth more than a hit at position 9.

    Binary relevance here, so gain is 1 or 0 and the ideal DCG is simply the
    best achievable arrangement of min(|relevant|, k) hits at the top.
    """
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(ranked_docs[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def hit_rate_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """Did ANY relevant doc appear in the top k. Crude, but it's the number a
    product owner actually understands."""
    return 1.0 if set(ranked_docs[:k]) & relevant else 0.0


def summarize(rows: list[dict], ks=(1, 3, 5, 10, 20)) -> dict:
    """Mean of each metric across queries, ignoring queries with no gold set
    (those are abstention cases and are scored separately)."""
    scored = [r for r in rows if r["relevant"]]
    if not scored:
        return {}

    out: dict[str, float] = {"n_queries": len(scored)}
    for k in ks:
        out[f"recall@{k}"] = sum(
            recall_at_k(r["ranked"], r["relevant"], k) for r in scored
        ) / len(scored)
        out[f"hit@{k}"] = sum(
            hit_rate_at_k(r["ranked"], r["relevant"], k) for r in scored
        ) / len(scored)
    out["ndcg@10"] = sum(ndcg_at_k(r["ranked"], r["relevant"], 10) for r in scored) / len(scored)
    out["mrr"] = sum(mrr(r["ranked"], r["relevant"]) for r in scored) / len(scored)
    out["precision@5"] = sum(
        precision_at_k(r["ranked"], r["relevant"], 5) for r in scored
    ) / len(scored)
    return out
