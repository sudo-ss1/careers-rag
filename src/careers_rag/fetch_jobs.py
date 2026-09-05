"""Ingest public job postings into a dated local snapshot.

Two stages, because the source exposes them separately:

  1. LIST  -- a paginated search endpoint returning structured metadata per job
             (title, location, category, req id, skills, a short teaser).
  2. DETAIL -- each posting page embeds a schema.org JobPosting as JSON-LD,
             which is where the full description text lives.

Design notes worth reading:
  * Snapshots are dated and immutable. A retrieval eval is only reproducible if
    the corpus it ran against is pinned -- postings appear and expire daily.
  * Detail fetches are rate-limited and resumable. Re-running skips what is
    already on disk, so an interrupted crawl costs nothing.
  * Only public data is stored, and each record keeps its source URL so every
    answer the system produces can cite where the text came from.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

BASE = "https://careers.cisco.com"
WIDGETS = f"{BASE}/widgets"
UA = "careers-rag-prototype/0.1 (portfolio project; contact via repo)"
PAGE_SIZE = 100
DETAIL_WORKERS = 2
DETAIL_DELAY = 0.9  # seconds between requests per worker


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": UA,
            "Referer": f"{BASE}/global/en/search-results",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", errors="ignore")


def fetch_listing(out: Path) -> list[dict]:
    """Page through the whole global job list."""
    jobs: list[dict] = []
    start, total = 0, None
    while True:
        page = _post(WIDGETS, {
            "lang": "en_global", "deviceType": "desktop", "country": "global",
            "pageName": "search-results", "ddoKey": "refineSearch",
            "sortBy": "Most recent", "subsearch": "", "from": start,
            "jobs": True, "counts": True,
            "all_fields": ["country", "state", "city", "category", "type"],
            "size": PAGE_SIZE, "clearAll": False, "jdsource": "facets",
            "isSliderEnable": False, "pageId": "page11", "siteType": "external",
            "keywords": "", "global": True,
        })["refineSearch"]

        total = total or page["totalHits"]
        batch = page["data"]["jobs"]
        if not batch:
            break
        jobs.extend(batch)
        print(f"  listing {len(jobs)}/{total}", file=sys.stderr)
        start += PAGE_SIZE
        if start >= total:
            break
        time.sleep(0.3)

    out.write_text(json.dumps(jobs, indent=1))
    return jobs


JOB_URL = re.compile(r"^/global/en/job/")


def _detail_url(job: dict) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", job.get("title", "")).strip("-")
    return f"{BASE}/global/en/job/{job['jobSeqNo']}/{slug}"


LD = re.compile(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', re.S)
TAG = re.compile(r"<[^>]+>")


def _clean_html(raw: str) -> str:
    """Unescape entities BEFORE stripping tags.

    The source double-encodes: the JSON-LD description arrives with markup as
    &lt;br /&gt;. Strip tags first and you delete nothing, then unescape and you
    are left with literal "<br />" as visible text -- which silently destroys
    every downstream heading boundary. Order matters here.
    """
    text = html.unescape(raw)
    text = html.unescape(text)                      # source is double-encoded
    # Block-level tags become newlines so section headings land on their own line.
    text = re.sub(r"</?(br|p|li|div|h[1-6]|tr)[^>]*>", "\n", text, flags=re.I)
    text = TAG.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def fetch_detail(job: dict, dest: Path) -> bool:
    """Pull the full description from the posting's JSON-LD. Returns True on success."""
    target = dest / f"{job['jobSeqNo']}.json"
    if target.exists():
        return True
    url = _detail_url(job)
    try:
        html = _get(url)
    except Exception as exc:
        # 403 here is rate limiting, not a permanent refusal. Back off and retry;
        # the snapshot is resumable so a later pass picks up whatever is missing.
        if "403" in str(exc):
            time.sleep(6.0)
            try:
                html = _get(url)
            except Exception:
                return False
        else:
            print(f"  ! {job['jobSeqNo']}: {exc}", file=sys.stderr)
            return False

    for block in LD.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if data.get("@type") != "JobPosting":
            continue
        target.write_text(json.dumps({
            "job_seq_no": job["jobSeqNo"],
            "req_id": job.get("reqId"),
            "title": data.get("title") or job.get("title"),
            "category": job.get("category"),
            "categories": job.get("multi_category") or [],
            "locations": job.get("multi_location") or [job.get("location")],
            "country": job.get("country"),
            "city": job.get("city"),
            "employment_type": job.get("type"),
            "remote_type": job.get("RemoteType"),
            "posted_date": job.get("postedDate"),
            "skills": job.get("ml_skills") or [],
            "teaser": job.get("descriptionTeaser"),
            "description": _clean_html(data.get("description", "")),
            "source_url": url,
            "apply_url": job.get("applyUrl"),
        }, indent=1))
        return True
    return False


def main() -> None:
    snapshot = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/snapshot")
    snapshot.mkdir(parents=True, exist_ok=True)
    details = snapshot / "jobs"
    details.mkdir(exist_ok=True)

    listing_file = snapshot / "listing.json"
    if listing_file.exists():
        jobs = json.loads(listing_file.read_text())
        print(f"listing cached: {len(jobs)} jobs")
    else:
        print("fetching listing...")
        jobs = fetch_listing(listing_file)
        print(f"listing: {len(jobs)} jobs")

    todo = [j for j in jobs if not (details / f"{j['jobSeqNo']}.json").exists()]
    print(f"details to fetch: {len(todo)}")

    def work(j):
        ok = fetch_detail(j, details)
        time.sleep(DETAIL_DELAY)
        return ok

    done = 0
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        for ok in pool.map(work, todo):
            done += 1
            if done % 50 == 0:
                print(f"  details {done}/{len(todo)}", file=sys.stderr)

    have = len(list(details.glob("*.json")))
    print(f"snapshot complete: {have} postings with full text")


if __name__ == "__main__":
    main()
