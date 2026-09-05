"""Retrieval strategies: lexical, dense, and the hybrid fusion of both.

The reason hybrid exists, in one sentence: dense retrieval understands meaning
but blurs exact tokens, lexical retrieval nails exact tokens but understands
nothing -- and job search needs both, often in the same query ("remote Golang
roles" is half vocabulary, half concept).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from careers_rag.bm25 import BM25Index
from careers_rag.embed import DenseIndex, EmbeddingStore

RRF_K = 60  # standard damping constant; large enough that rank 1 vs 2 isn't decisive


@dataclass
class Hit:
    chunk_id: str
    score: float


def reciprocal_rank_fusion(
    runs: list[list[tuple[str, float]]], weights: list[float] | None = None, k: int = 50
) -> list[Hit]:
    """Fuse ranked lists by RANK, not by score.

    This is the whole trick. BM25 scores are unbounded and corpus-dependent;
    cosine similarities live in [-1, 1]. They cannot be added or averaged
    meaningfully -- any attempt needs a normalisation that shifts every time the
    corpus changes. RRF sidesteps it entirely by throwing away the scores and
    keeping only the ordering, which is comparable across any two retrievers.
    """
    weights = weights or [1.0] * len(runs)
    fused: dict[str, float] = {}
    for run, w in zip(runs, weights):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (RRF_K + rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [Hit(doc_id, score) for doc_id, score in ranked]


class Retriever:
    """One object, three modes, so the eval can compare them on identical inputs."""

    def __init__(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        store: EmbeddingStore | None = None,
        vectors: np.ndarray | None = None,
    ):
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Index(chunk_ids, chunk_texts)
        self.dense = DenseIndex(chunk_ids, vectors) if vectors is not None else None
        self.store = store
        self._qcache: dict[str, np.ndarray] = {}

    def _query_vec(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = self.store.embed([query], label="q")[0]
        return self._qcache[query]

    def search(self, query: str, mode: str = "hybrid", k: int = 50) -> list[Hit]:
        if mode == "bm25":
            return [Hit(i, s) for i, s in self.bm25.search(query, k)]

        if mode == "dense":
            if self.dense is None:
                raise RuntimeError("dense index not built -- embeddings missing")
            return [Hit(i, s) for i, s in self.dense.search(self._query_vec(query), k)]

        if mode == "hybrid":
            if self.dense is None:
                raise RuntimeError("dense index not built -- embeddings missing")
            # Over-fetch from each arm before fusing: a document ranked 40th by
            # one retriever and 3rd by the other should still surface, and it
            # cannot if each arm only contributes its top k.
            lex = self.bm25.search(query, k * 2)
            vec = self.dense.search(self._query_vec(query), k * 2)
            return reciprocal_rank_fusion([lex, vec], k=k)

        raise ValueError(f"unknown mode: {mode}")
