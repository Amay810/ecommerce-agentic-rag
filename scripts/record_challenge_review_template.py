"""Record a pending review-template hash without freezing locked labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150 or len({row["challenge_id"] for row in rows}) != 150:
        raise ValueError("review template must contain 150 unique rows")
    if any(row["human_review_status"] == "user_confirmed" for row in rows):
        raise ValueError("pending-template recorder cannot freeze confirmed labels")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_role") != "locked" or manifest.get("labels_frozen"):
        raise ValueError("expected an unfrozen locked manifest")
    manifest["review_template_csv"] = {
        "name": args.csv.name,
        "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "status_counts": {status: sum(row["human_review_status"] == status for row in rows)
                          for status in sorted({row["human_review_status"] for row in rows})},
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
