"""Emit a compact corpus file the report page can search entirely in the browser.

Why client-side: an API key can never ship to a browser, and a hosted backend is
one more thing that can be down when someone clicks the link. Lexical retrieval
needs no key and no server, so the full snapshot is searchable offline in the
page — and the parts that genuinely need a key ship as recorded runs instead.

Payload discipline: descriptions are truncated to the first N characters. Job
postings front-load the useful signal (team, impact, qualifications), so the tail
costs bytes without buying much retrieval quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from careers_rag.corpus import load_snapshot  # noqa: E402

TEXT_CHARS = 1400
TEASER_CHARS = 240


def main() -> None:
    snapshot = Path(sys.argv[1])
    out = Path(sys.argv[2])
    postings = load_snapshot(snapshot)

    docs = []
    for p in postings:
        body = " ".join(p.description.split())
        docs.append({
            "r": p.req_id,
            "t": p.title,
            "c": p.category,
            "l": p.location_str,
            "n": p.country,
            "e": p.employment_type,
            "rt": p.remote_type,
            "d": p.posted_date,
            "u": p.source_url,
            # Indexed text. Two deliberate choices:
            #  * title and skills repeat, because BM25 has no field weighting
            #    here -- repetition IS the weighting.
            #  * the requisition id is included. In the Python pipeline it
            #    arrives via each chunk's contextual header; omitting it here
            #    made exact-id lookup -- the query type this whole project is
            #    built around -- return nothing.
            "x": f"req {p.req_id} {p.req_id} {p.title} {p.title} "
                 f"{' '.join(p.skills[:14])} {body[:TEXT_CHARS]}",
            "s": body[:TEASER_CHARS],
        })

    # --- facets ------------------------------------------------------------
    # A gazetteer built FROM the corpus, not from a hardcoded list. A follow-up
    # turn like "in bangalore" is only interpretable if the client can tell a
    # place name from a role term, and the only authority on which places exist
    # is the snapshot itself.
    from collections import Counter
    cities = Counter()
    countries = Counter()
    for p_ in postings:
        countries[p_.country] += 1
        for loc in p_.locations:
            city = loc.split(",")[0].strip()
            if len(city) > 2:
                cities[city] += 1

    # Blank values must never reach the gazetteer. Some postings carry an empty
    # country, and an empty place name compiles to the regex \b\b -- which
    # matches every query and silently overwrote whatever place had really been
    # found. Filtering at the source, and guarding again client-side.
    def clean(names):
        return sorted({n.strip() for n in names if n and len(n.strip()) > 1})

    facets = {
        # Cities with a single posting are dropped: they add gazetteer noise
        # and a one-hit filter is not a useful conversational constraint.
        "cities": clean(c for c, n in cities.items() if n >= 2),
        "countries": clean(countries),
        "categories": clean(p_.category for p_ in postings),
        "types": clean(p_.employment_type for p_ in postings),
        "remote": clean(p_.remote_type for p_ in postings),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"snapshot": snapshot.name, "facets": facets, "docs": docs},
                              separators=(",", ":")))
    kb = out.stat().st_size / 1024
    print(f"{len(docs)} postings -> {out}  ({kb:.0f} KB raw)")


if __name__ == "__main__":
    main()
