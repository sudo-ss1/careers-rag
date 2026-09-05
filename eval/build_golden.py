"""Build the golden evaluation set.

Ground truth comes from structured metadata that the retriever never sees as a
filter: the ML-extracted `skills` list, `category`, and `country`. The retriever
only ever indexes title + description text, so it has to *earn* the match.

Three query families, because they stress different halves of the system:

  A. LEXICAL   -- requisition ids and exact tool names. Dense embeddings blur
                  precisely these tokens, so this is where BM25 earns its place.
  B. SEMANTIC  -- paraphrases that deliberately avoid the corpus's own
                  vocabulary ("container orchestration", not "Kubernetes").
                  Generated separately by build_golden_semantic.py.
  C. UNANSWERABLE -- roles that plainly do not exist in this corpus. Gold set is
                  empty; the correct behaviour is to abstain, and abstention is
                  measured like any other capability.

STATED BIAS: family A labels derive from skills that were themselves extracted
from the description text, so the labelled term usually appears verbatim in the
document. That flatters lexical retrieval. It is reported rather than hidden,
and family B exists to counterbalance it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from careers_rag.corpus import load_snapshot  # noqa: E402

# Tool/technology tokens that are unambiguous enough to define relevance.
TECH_TOKENS = [
    "kubernetes", "terraform", "golang", "pytorch", "kafka", "ansible",
    "postgresql", "docker", "jenkins", "spark", "tensorflow", "grpc",
    "react", "angular", "selenium", "verilog", "matlab", "sap",
]

# Gold-set size is capped at 10 on purpose. recall@10 cannot exceed
# 10/|gold|, so a query with 22 relevant docs has a hard ceiling of 0.45 and
# the metric ends up measuring label size rather than retrieval quality.
# Capping keeps recall@10 attainable and therefore meaningful.
MIN_GOLD, MAX_GOLD = 2, 10


def _skill_index(postings) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for p in postings:
        blob = " ".join(p.skills).lower()
        for tok in TECH_TOKENS:
            if tok in blob:
                idx.setdefault(tok, set()).add(p.job_id)
    return idx


def build(snapshot: Path) -> list[dict]:
    postings = load_snapshot(snapshot)
    by_id = {p.job_id: p for p in postings}
    queries: list[dict] = []

    # --- A1: requisition id lookup -- one exact answer, pure lexical ----------
    for p in postings[:: max(1, len(postings) // 12)][:12]:
        if not p.req_id:
            continue
        queries.append({
            "query_id": f"reqid-{p.req_id}",
            "family": "lexical",
            "subtype": "req_id",
            "query": f"What is requisition {p.req_id} about?",
            "relevant": [p.job_id],
        })

    # --- A2: technology tokens ------------------------------------------------
    skills = _skill_index(postings)
    for tok, ids in sorted(skills.items()):
        if MIN_GOLD <= len(ids) <= MAX_GOLD:
            queries.append({
                "query_id": f"tech-{tok}",
                "family": "lexical",
                "subtype": "tech_token",
                "query": f"Which open roles work with {tok}?",
                "relevant": sorted(ids),
            })

    # --- A3: category + country ----------------------------------------------
    combos = Counter((p.category, p.country) for p in postings)
    for (cat, country), n in combos.most_common(40):
        if not cat or not country or not (MIN_GOLD <= n <= MAX_GOLD):
            continue
        ids = [p.job_id for p in postings if p.category == cat and p.country == country]
        queries.append({
            "query_id": f"catloc-{cat[:12]}-{country[:12]}".replace(" ", ""),
            "family": "lexical",
            "subtype": "category_location",
            "query": f"Are there {cat} openings in {country}?",
            "relevant": sorted(ids),
        })

    # --- C: unanswerable -- correct behaviour is abstention -------------------
    for i, q in enumerate([
        "Do you have quantum cryptography research roles in Antarctica?",
        "I'm looking for a role as a commercial airline pilot.",
        "Are there any open positions for a professional chef?",
        "Do you hire marine biologists for coral reef research?",
        "Is there a remote role teaching high-school mathematics?",
        "What veterinary surgeon openings are available in Lisbon?",
    ]):
        queries.append({
            "query_id": f"unanswerable-{i}",
            "family": "unanswerable",
            "subtype": "out_of_corpus",
            "query": q,
            "relevant": [],
        })

    return queries


def main() -> None:
    snapshot = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eval/golden.json")
    qs = build(snapshot)
    out.write_text(json.dumps(qs, indent=1))

    fams = Counter(q["family"] for q in qs)
    subs = Counter(q["subtype"] for q in qs)
    scored = [q for q in qs if q["relevant"]]
    avg = sum(len(q["relevant"]) for q in scored) / max(1, len(scored))
    print(f"golden set: {len(qs)} queries -> {out}")
    print(f"  families: {dict(fams)}")
    print(f"  subtypes: {dict(subs)}")
    print(f"  avg gold docs per scored query: {avg:.1f}")


if __name__ == "__main__":
    main()
