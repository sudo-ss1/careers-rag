"""Load a snapshot and turn postings into retrievable chunks.

The chunking decisions here are the ones that move retrieval quality most, so
each is deliberate rather than a default:

  * Split on the posting's own section headings, not on a character count.
    Job descriptions are already structured ("Meet the Team", "Minimum
    Qualifications"); cutting every N characters shreds that structure and
    produces chunks whose embedding is an average of two unrelated topics.

  * Prepend a contextual header to every chunk (title, org, location, req id)
    before embedding. A bare "Minimum Qualifications" list is nearly
    unretrievable on its own -- it doesn't say what job it belongs to. This is
    contextual retrieval, and it is the cheapest recall win available.

  * Keep short postings whole. Splitting a 400-character posting into three
    fragments makes every fragment worse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Headings real postings use. Ordered longest-first so "Minimum Qualifications"
# wins over a bare "Qualifications" match.
SECTION_PATTERNS = [
    r"Meet the Team", r"Your Impact", r"Minimum Qualifications",
    r"Preferred Qualifications", r"Who You'?ll Work With", r"Who You Are",
    r"What You'?ll Do", r"Why Cisco", r"Responsibilities", r"Qualifications",
    r"Required Skills", r"About Us", r"Message to Applicants",
]
SECTION_RE = re.compile(
    r"^\s*(?:#+\s*)?(" + "|".join(SECTION_PATTERNS) + r")\s*:?\s*$",
    re.I | re.M,
)

MAX_CHARS = 1400   # ~350 tokens; comfortably inside any embedding context
MIN_CHARS = 120    # below this, a chunk carries no usable signal


@dataclass
class Posting:
    job_id: str
    req_id: str
    title: str
    category: str
    locations: list[str]
    country: str
    city: str
    employment_type: str
    remote_type: str
    posted_date: str
    skills: list[str]
    description: str
    source_url: str

    @property
    def location_str(self) -> str:
        return "; ".join(self.locations[:3])


@dataclass
class Chunk:
    chunk_id: str
    job_id: str          # ground truth is tracked at DOCUMENT level, not chunk
    section: str
    text: str            # what gets embedded: contextual header + body
    body: str            # the raw section text, for display and citation
    posting: Posting = field(repr=False)


def load_snapshot(path: Path) -> list[Posting]:
    postings = []
    for f in sorted((path / "jobs").glob("*.json")):
        d = json.loads(f.read_text())
        desc = (d.get("description") or "").strip()
        if len(desc) < 200:          # placeholder or expired posting
            continue
        postings.append(Posting(
            job_id=d["job_seq_no"], req_id=str(d.get("req_id") or ""),
            title=d.get("title") or "", category=d.get("category") or "",
            locations=[l for l in (d.get("locations") or []) if l],
            country=d.get("country") or "", city=d.get("city") or "",
            employment_type=d.get("employment_type") or "",
            remote_type=d.get("remote_type") or "",
            posted_date=(d.get("posted_date") or "")[:10],
            skills=d.get("skills") or [], description=desc,
            source_url=d.get("source_url") or "",
        ))
    return postings


def _split_sections(text: str) -> list[tuple[str, str]]:
    marks = [(m.start(), m.end(), m.group(1)) for m in SECTION_RE.finditer(text)]
    if not marks:
        return [("Description", text)]

    out = []
    if marks[0][0] > MIN_CHARS:                    # preamble before first heading
        out.append(("Overview", text[: marks[0][0]]))
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[end:stop].strip()
        if body:
            out.append((name.title(), body))
    return out


def _wrap(section: str, body: str) -> list[str]:
    """Hard-split an over-long section on paragraph boundaries, never mid-sentence."""
    if len(body) <= MAX_CHARS:
        return [body]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", body):
        if len(cur) + len(para) + 2 > MAX_CHARS and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_posting(p: Posting) -> list[Chunk]:
    # The header every chunk carries. Without it, "3+ years of Python" is a
    # sentence with no job attached, and no query can retrieve it reliably.
    header = (
        f"Job: {p.title}\nReq ID: {p.req_id}\nCategory: {p.category}\n"
        f"Location: {p.location_str}\nType: {p.employment_type} ({p.remote_type})"
    )

    chunks: list[Chunk] = []
    for section, body in _split_sections(p.description):
        for i, piece in enumerate(_wrap(section, body)):
            if len(piece) < MIN_CHARS and chunks:
                # Fold a scrap into the previous chunk rather than indexing noise.
                chunks[-1].body += "\n" + piece
                chunks[-1].text += "\n" + piece
                continue
            cid = f"{p.job_id}::{len(chunks)}"
            chunks.append(Chunk(
                chunk_id=cid, job_id=p.job_id, section=section,
                text=f"{header}\nSection: {section}\n\n{piece}",
                body=piece, posting=p,
            ))

    if not chunks:                                   # very short posting
        chunks.append(Chunk(f"{p.job_id}::0", p.job_id, "Description",
                            f"{header}\n\n{p.description}", p.description, p))
    return chunks


def build_chunks(postings: list[Posting]) -> list[Chunk]:
    return [c for p in postings for c in chunk_posting(p)]
