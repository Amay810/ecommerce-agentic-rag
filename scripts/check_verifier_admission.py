"""Fail-closed admission check shared by answer-postprocessing PBS jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ecommerce_rag.claim_verifier import verifier_config_hash


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCE = ROOT / "ecommerce_rag" / "claim_verifier.py"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(VERIFIER_SOURCE.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-report", type=Path, required=True)
    parser.add_argument("--postprocess-report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.locked_report.read_text(encoding="utf-8"))
    provenance = report.get("provenance") or {}
    expected_commit = source_commit()
    expected_config = verifier_config_hash()
    checks = {
        "dataset_role_locked": report.get("dataset_role") == "locked",
        "hard_gate_admitted": report.get("hard_gate_admitted") is True,
        "nscc_smoke_eligible": report.get("nscc_smoke_eligible") is True,
        "verifier_code_commit_matches": provenance.get("verifier_code_commit") == expected_commit,
        "verifier_config_hash_matches": provenance.get("verifier_config_hash") == expected_config,
    }
    if args.postprocess_report:
        sidecar = json.loads(args.postprocess_report.read_text(encoding="utf-8"))
        checks.update({
            "sidecar_verifier_code_commit_matches": sidecar.get("verifier_code_commit") == expected_commit,
            "sidecar_verifier_config_hash_matches": sidecar.get("verifier_config_hash") == expected_config,
        })
    failures = sorted(key for key, value in checks.items() if not value)
    if failures:
        raise SystemExit(f"verifier admission failed closed: {', '.join(failures)}")
    print(json.dumps({"passed": True, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
