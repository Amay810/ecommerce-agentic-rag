"""Fail-closed structural gate for the versioned Phase-A smoke run."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from ecommerce_rag.evidence import EVIDENCE_BEARING_TOOLS


VARIANTS = ("base", "evidence_verify", "evidence_verify_repair")
REQUIRED_GRADE_FIELDS = {
    "success", "operational_success", "policy_compliant", "terminal_state_match",
    "hard_verification_pass", "answer_fact_pass", "joint_success",
    "citation_diagnostics", "repair_attempted", "repair_succeeded",
}


def _raw_source_count(tool_name: str, result: dict) -> int:
    if tool_name == "search_catalog":
        return len(result.get("items") or [])
    if tool_name == "get_policy":
        return len(result.get("policies") or [])
    if tool_name == "compare_products":
        return len(result.get("products") or [])
    if tool_name in {"get_product", "get_order"}:
        return int(bool(result.get("product" if tool_name == "get_product" else "order")))
    return int(any(key != "ok" for key in result))


def _load_store(path: Path) -> list[tuple[str, dict, dict]]:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    try:
        return [(task_id, json.loads(trajectory), json.loads(grade)) for task_id, trajectory, grade in
                conn.execute("SELECT task_id,trajectory_json,grade_json FROM trajectories ORDER BY task_id")]
    finally:
        conn.close()


def _conversion_errors(trajectory: dict) -> list[str]:
    errors: list[str] = []
    calls = {str(call.get("call_id")): call for call in trajectory.get("tool_calls", [])
             if call.get("name") in EVIDENCE_BEARING_TOOLS}
    spans = trajectory.get("evidence_conversion_spans")
    if not isinstance(spans, list):
        return ["evidence_conversion_spans missing"]
    ids = [str(span.get("tool_call_id")) for span in spans]
    if Counter(ids) != Counter(calls.keys()):
        errors.append(f"call/span mismatch calls={sorted(calls)} spans={sorted(ids)}")
    ledger = trajectory.get("evidence_ledger")
    if not isinstance(ledger, list):
        return [*errors, "evidence_ledger missing"]
    evidence_ids = [str(row.get("evidence_id")) for row in ledger]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate evidence ids")
    ledger_by_call: dict[str, list[str]] = {}
    for row in ledger:
        ledger_by_call.setdefault(str(row.get("tool_call_id")), []).append(str(row.get("evidence_id")))
    for span in spans:
        call_id = str(span.get("tool_call_id"))
        call = calls.get(call_id)
        status = span.get("status")
        span_ids = [str(value) for value in span.get("evidence_ids", [])]
        actual_ids = ledger_by_call.get(call_id, [])
        if status not in {"converted", "valid_empty", "tool_failed", "converter_missing"}:
            errors.append(f"{call_id}: invalid status {status}")
        if status == "converter_missing":
            errors.append(f"{call_id}: converter_missing")
        if span_ids != actual_ids or int(span.get("evidence_item_count", -1)) != len(actual_ids):
            errors.append(f"{call_id}: ledger/span evidence mismatch")
        if call is None:
            continue
        result = call.get("result") or {}
        source_count = int(span.get("source_item_count", -1))
        actual_source_count = _raw_source_count(str(call.get("name")), result)
        if source_count != actual_source_count:
            errors.append(f"{call_id}: source count {source_count} != raw {actual_source_count}")
        if status == "converted" and (not result.get("ok") or source_count <= 0 or not actual_ids):
            errors.append(f"{call_id}: invalid converted span")
        if status == "valid_empty" and (not result.get("ok") or source_count != 0 or actual_ids):
            errors.append(f"{call_id}: invalid valid_empty span")
        if status == "tool_failed" and (result.get("ok") or actual_ids):
            errors.append(f"{call_id}: invalid tool_failed span")
        if result.get("ok") and source_count > 0 and not actual_ids:
            errors.append(f"{call_id}: successful nonempty result produced no evidence")
    unknown_owner = sorted(set(ledger_by_call) - set(calls))
    if unknown_owner:
        errors.append(f"ledger rows owned by unknown calls: {unknown_owner}")
    return errors


def assess(manifest_path: Path, store_dir: Path, run_name: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = list(manifest.get("task_ids", []))
    expected = set(expected_ids)
    checks: list[dict] = []
    diagnostics: dict[str, dict] = {}

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("manifest_has_unique_task_ids", len(expected_ids) == len(expected),
          f"{len(expected_ids)} rows/{len(expected)} unique")
    for variant in VARIANTS:
        path = store_dir / f"{run_name}_{variant}.sqlite"
        try:
            rows = _load_store(path)
        except Exception as exc:
            check(f"{variant}:store_readable", False, str(exc))
            continue
        ids = [row[0] for row in rows]
        check(f"{variant}:trajectory_count", len(rows) == len(expected_ids), f"{len(rows)}/{len(expected_ids)}")
        check(f"{variant}:task_ids", set(ids) == expected and len(ids) == len(set(ids)),
              "exact id set" if set(ids) == expected else f"missing={sorted(expected-set(ids))} extra={sorted(set(ids)-expected)}")
        generation_errors = fallback_trajectories = too_many_repairs = illegal_changes = 0
        trace_errors: list[str] = []
        conversion_errors: list[str] = []
        grade_errors: list[str] = []
        for task_id, trajectory, grade in rows:
            illegal_changes += int(bool(grade.get("illegal_state_change")))
            too_many_repairs += int(len(trajectory.get("repair_spans", [])) > 1)
            missing_grade = sorted(REQUIRED_GRADE_FIELDS - set(grade))
            if missing_grade:
                grade_errors.append(f"{task_id}: {missing_grade}")
            trajectory_fallback = False
            model_calls = trajectory.get("model_calls")
            if not isinstance(model_calls, list) or not model_calls:
                trace_errors.append(f"{task_id}: model_calls missing")
            for model_call in model_calls or []:
                trace = model_call.get("llm")
                if not isinstance(trace, dict) or not isinstance(trace.get("attempts"), list):
                    trace_errors.append(f"{task_id}: incomplete llm trace")
                    continue
                trace_items = [trace] + ([trace["repair_llm"]] if isinstance(trace.get("repair_llm"), dict) else [])
                for item in trace_items:
                    trajectory_fallback |= item.get("resolution") == "fallback_handoff"
                    for attempt in item.get("attempts", []):
                        generation_errors += int(attempt.get("parse_stage") == "generation_error")
            fallback_trajectories += int(trajectory_fallback)
            conversion_errors.extend(f"{task_id}: {error}" for error in _conversion_errors(trajectory))
        check(f"{variant}:generation_errors_zero", generation_errors == 0, str(generation_errors))
        check(f"{variant}:fallback_trajectories_zero", fallback_trajectories == 0, str(fallback_trajectories))
        check(f"{variant}:illegal_state_changes_zero", illegal_changes == 0, str(illegal_changes))
        check(f"{variant}:repair_at_most_once", too_many_repairs == 0, str(too_many_repairs))
        check(f"{variant}:trace_fields_complete", not trace_errors, "; ".join(trace_errors[:5]))
        check(f"{variant}:grader_fields_complete", not grade_errors, "; ".join(grade_errors[:5]))
        check(f"{variant}:evidence_conversion_complete", not conversion_errors,
              "; ".join(conversion_errors[:8]))
        diagnostics[variant] = {
            "operational_success": sum(bool(grade.get("operational_success")) for _, _, grade in rows),
            "joint_success": sum(bool(grade.get("joint_success")) for _, _, grade in rows),
            "repairs_adopted": sum(bool(grade.get("repair_succeeded")) for _, _, grade in rows),
            "citation_diagnostic_rows": sum(bool(grade.get("citation_diagnostics")) for _, _, grade in rows),
        }
    return {"schema_version": 2, "passed": all(row["passed"] for row in checks),
            "checks": checks, "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.manifest, args.store_dir, args.run_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
