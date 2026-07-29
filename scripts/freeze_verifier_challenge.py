"""Freeze review labels and code/config provenance in a challenge manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from ecommerce_rag.claim_verifier import verifier_config_hash


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCE = ROOT / "ecommerce_rag" / "claim_verifier.py"
EVALUATOR_SOURCE = ROOT / "scripts" / "evaluate_verifier_challenge.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-role", choices=("development", "locked"), required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150 or len({row["challenge_id"] for row in rows}) != 150:
        raise ValueError("challenge CSV must contain 150 unique rows")
    confirmed = sum(row["human_review_status"] == "user_confirmed" for row in rows)
    if args.dataset_role == "locked" and confirmed != 150:
        raise ValueError("locked labels cannot be frozen until all 150 rows are user_confirmed")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest.update({
        "dataset_role": args.dataset_role,
        "human_review_csv": {
            "name": args.csv.name, "sha256": sha(args.csv), "row_count": len(rows),
            "status_counts": {status: sum(row["human_review_status"] == status for row in rows)
                              for status in sorted({row["human_review_status"] for row in rows})},
        },
        "labels_frozen": confirmed == 150,
        "verifier_code_commit": source_commit(VERIFIER_SOURCE),
        "verifier_source_sha256": sha(VERIFIER_SOURCE),
        "verifier_config_hash": verifier_config_hash(),
        "evaluator_code_commit": source_commit(EVALUATOR_SOURCE),
        "evaluator_source_sha256": sha(EVALUATOR_SOURCE),
    })
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
