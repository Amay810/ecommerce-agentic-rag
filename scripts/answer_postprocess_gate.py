"""Fail-closed structural gate for frozen-trajectory answer postprocessing."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from ecommerce_rag.answer_postprocess import stable_hash


def base_rows(path: Path) -> dict[str, dict]:
    conn = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        rows = conn.execute("SELECT task_id, trajectory_json FROM trajectories").fetchall()
        return {task_id: json.loads(payload) for task_id, payload in rows}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-store", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow", "terminal_grounded"), required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument("--reference-run-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = base_rows(args.base_store)
    if args.task_manifest:
        manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
        wanted = set(manifest["task_ids"])
        base = {task_id: row for task_id, row in base.items() if task_id in wanted}
        if set(base) != wanted:
            raise ValueError("task manifest does not exactly match the frozen base store")
    sidecar = [json.loads(line) for line in args.sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = []
    checks.append({"name": "exact_row_count", "passed": len(sidecar) == args.expected_count == len(base),
                   "detail": {"base": len(base), "sidecar": len(sidecar)}})
    checks.append({"name": "unique_task_ids", "passed": len({row["task_id"] for row in sidecar}) == len(sidecar)})
    checks.append({"name": "exact_task_ids", "passed": {row["task_id"] for row in sidecar} == set(base)})
    mismatches = []
    for row in sidecar:
        trajectory = base.get(row["task_id"])
        if not trajectory:
            continue
        expected = {
            "source_trajectory_id": trajectory["trajectory_id"],
            "action_sequence_sha256": stable_hash(trajectory.get("actions") or []),
            "tool_calls_sha256": stable_hash(trajectory.get("tool_calls") or []),
            "terminal_state_sha256": stable_hash(trajectory.get("final_state") or {}),
            "evidence_ledger_sha256": stable_hash(trajectory.get("evidence_ledger") or []),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                mismatches.append({"task_id": row["task_id"], "field": key})
        base_handoff = any(action.get("action_type") == "handoff" for action in trajectory.get("actions") or [])
        if base_handoff and (row.get("eligible") or row.get("final_answer") != trajectory.get("final_answer")):
            mismatches.append({"task_id": row["task_id"], "field": "base_handoff_pass_through"})
        if not row.get("eligible") and row.get("final_answer") != trajectory.get("final_answer"):
            mismatches.append({"task_id": row["task_id"], "field": "ineligible_answer_pass_through"})
        if row.get("eligible") and row.get("ineligible_reason") is not None:
            mismatches.append({"task_id": row["task_id"], "field": "eligible_with_ineligible_reason"})
        if args.mode == "shadow" and row.get("final_answer") != trajectory.get("final_answer"):
            mismatches.append({"task_id": row["task_id"], "field": "shadow_changed_answer"})
        if row.get("error") or row.get("truncated"):
            mismatches.append({"task_id": row["task_id"], "field": "generation_failure"})
        if args.mode == "terminal_grounded" and row.get("eligible") and not row.get("generation_config_hash"):
            mismatches.append({"task_id": row["task_id"], "field": "missing_generation_config_hash"})
    checks.append({"name": "action_and_terminal_immutability", "passed": not mismatches,
                   "detail": mismatches})
    if args.run_report:
        run_report = json.loads(args.run_report.read_text(encoding="utf-8"))
        report_failures = []
        if run_report.get("mode") != args.mode:
            report_failures.append("mode")
        if run_report.get("verifier_role") != "diagnostic_only":
            report_failures.append("verifier_role")
        if run_report.get("verifier_gate_applied") is not False:
            report_failures.append("verifier_gate_applied")
        expected_generation_hash = run_report.get("generation_config_hash")
        sidecar_hashes = {row.get("generation_config_hash") for row in sidecar if row.get("eligible")}
        if args.mode == "terminal_grounded" and sidecar_hashes != {expected_generation_hash}:
            report_failures.append("generation_config_hash")
        checks.append({"name": "diagnostic_only_run_contract", "passed": not report_failures,
                       "detail": report_failures})
        if args.reference_run_report:
            reference = json.loads(args.reference_run_report.read_text(encoding="utf-8"))
            stable_fields = ("generation_config_hash", "verifier_code_commit", "verifier_config_hash",
                             "verifier_role", "verifier_gate_applied")
            drift = [field for field in stable_fields if run_report.get(field) != reference.get(field)]
            checks.append({"name": "frozen_smoke_to_dev_configuration", "passed": not drift,
                           "detail": drift})
    passed = all(check["passed"] for check in checks)
    report = {"schema_version": 1, "mode": args.mode, "passed": passed, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
