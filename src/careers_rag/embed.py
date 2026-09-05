"""Dense retrieval over OpenAI embeddings.

Everything here is built to be re-run cheaply: embeddings are the expensive
part of the pipeline, so they are cached to disk keyed by (model, text hash).
Re-running an eval after a retrieval-parameter change should cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import openai

BATCH = 128


def load_env(root: Path) -> None:
    """Read .env without requiring python-dotenv at import time."""
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _fingerprint(model: str, texts: list[str]) -> str:
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


class EmbeddingStore:
    def __init__(self, model: str, cache_dir: Path):
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = openai.OpenAI(timeout=60.0, max_retries=4)

    def embed(self, texts: list[str], label: str) -> np.ndarray:
        """Embed a list of texts, caching the whole batch under its fingerprint."""
        fp = _fingerprint(self.model, texts)
        cache = self.cache_dir / f"{label}-{fp}.npy"
        if cache.exists():
            return np.load(cache)

        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            for attempt in range(5):
                try:
                    resp = self._client.embeddings.create(model=self.model, input=batch)
                    vectors.extend(d.embedding for d in resp.data)
                    break
                except openai.RateLimitError:
                    # The SDK already retries; this is the outer guard for
                    # sustained quota pressure during a large backfill.
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"embedding failed at batch {i}")
            print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)

        arr = np.asarray(vectors, dtype=np.float32)
        # L2-normalise once at write time so search is a plain dot product
        # instead of recomputing norms on every query.
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        np.save(cache, arr)
        return arr


class DenseIndex:
    """Exact (brute-force) cosine search.

    Deliberately exact rather than ANN: at ~1k documents an exhaustive scan is
    sub-millisecond, and it means the numbers in the eval report measure
    *retrieval quality* rather than the recall loss of an approximate index.
    At production scale this is where HNSW goes -- and where you would compare
    ANN recall against these exact results to size ef_search.
    """

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = ids
        self.vectors = vectors

    def search(self, query_vec: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        sims = self.vectors @ query_vec
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]
