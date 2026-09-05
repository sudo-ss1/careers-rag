"""BM25 lexical retrieval, implemented directly rather than pulled from a library.

It's here because dense retrieval alone fails badly on the queries that matter
most in job search: requisition numbers, exact tool names ("Terraform",
"LangGraph"), product names, acronyms. Embeddings blur precisely the tokens a
user is being most specific about.

BM25 in one line: TF-IDF with two corrections -- term frequency saturates (the
tenth occurrence of "Python" adds far less than the second), and long documents
are penalised so they can't win on length alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

K1 = 1.5   # term-frequency saturation
B = 0.75   # length-normalisation strength

# Split on non-alphanumerics but keep intra-token . + # - so "node.js", "c++",
# "ci/cd" and requisition ids survive tokenisation intact.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+#/-]*")

STOP = {
    "the","a","an","and","or","of","to","in","for","on","with","at","by","is",
    "are","be","as","that","this","it","you","your","we","our","will","have",
    "has","from","their","they","not","but","can","all","any","more","who",
}


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str]):
        self.doc_ids = doc_ids
        self.tokens = [tokenize(t) for t in texts]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        self.tf: list[Counter] = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))

        n = len(doc_ids)
        # Standard BM25 idf with the +0.5 smoothing that keeps very common
        # terms from going negative.
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        # Inverted index: only score documents that contain a query term.
        # Scoring all 900+ postings per query would be pointless work.
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, counter in enumerate(self.tf):
            for term in counter:
                self.postings[term].append(i)

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scores: dict[int, float] = defaultdict(float)

        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                freq = self.tf[i][term]
                norm = 1 - B + B * (self.lengths[i] / self.avg_len or 1)
                scores[i] += idf * (freq * (K1 + 1)) / (freq + K1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.doc_ids[i], s) for i, s in ranked]
