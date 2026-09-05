"""Re-clean stored descriptions in place after a cleaner fix.

Kept as a script rather than a one-off shell command because re-deriving a
snapshot's text is exactly the kind of thing you want reproducible: the raw
fetch is expensive and rate-limited, so you repair rather than refetch.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from careers_rag.fetch_jobs import _clean_html  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1]) / "jobs"
    fixed = 0
    for f in root.glob("*.json"):
        d = json.loads(f.read_text())
        cleaned = _clean_html(d.get("description", ""))
        if cleaned != d.get("description"):
            d["description"] = cleaned
            f.write_text(json.dumps(d, indent=1))
            fixed += 1
    print(f"repaired {fixed} postings")


if __name__ == "__main__":
    main()
