"""Write the permanent non-admission closeout for verifier challenge dev_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_verifier_challenge import atomic_write_new


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("dataset_role") != "development" or report.get("admission_applicable"):
        raise ValueError("dev_v1 closeout accepts development reports only")
    closeout = {
        "schema_version": 1, "challenge": "verifier_challenge_dev_v1",
        "dataset_role": "development", "status": "closed_as_development_regression",
        "admission_applicable": False, "nscc_smoke_eligible": False,
        "permanent_constraints": [
            "dev_v1 may be used to develop and regress verifier v2",
            "dev_v1 metrics are not independent generalization evidence",
            "dev_v1 can never unlock NSCC smoke regardless of its metrics",
        ],
        "report_sha256": sha256(args.report), "manifest_sha256": sha256(args.manifest),
        "reported_metrics": {
            key: report[key] for key in (
                "contradiction_precision", "contradiction_recall", "unsupported_precision",
                "unsupported_recall", "supported_hard_failure_false_positive_rate",
                "unknown_rate", "automatic_decision_coverage", "citation_incorrect_precision",
                "citation_incorrect_recall",
            )
        },
    }
    atomic_write_new(
        args.output,
        (json.dumps(closeout, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


if __name__ == "__main__":
    main()
