"""Regrade immutable trajectories and write versioned sidecar evidence.

The source SQLite database is opened read-only.  Regraded decisions are written
to JSONL so historical grades remain available for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

from ecommerce_rag.domain import ToolCall, Trajectory
from ecommerce_rag.harness import grade, load_tasks


def _trajectory(payload: dict[str, Any]) -> Trajectory:
    allowed = {field.name for field in fields(Trajectory)}
    values = {key: value for key, value in payload.items() if key in allowed}
    values["tool_calls"] = [ToolCall(**call) for call in payload.get("tool_calls", [])]
    return Trajectory(**values)


def regrade_store(tasks_path: Path, store_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    uri = store_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT trajectory_id, task_id, seed, trajectory_json, grade_json FROM trajectories ORDER BY trajectory_id"
        ).fetchall()
    finally:
        conn.close()

    output: list[dict[str, Any]] = []
    legacy_success = 0
    for trajectory_id, task_id, seed, trajectory_json, grade_json in rows:
        if task_id not in tasks:
            raise ValueError(f"trajectory {trajectory_id} references unknown task {task_id}")
        payload = json.loads(trajectory_json)
        original_grade = json.loads(grade_json)
        result = grade(
            tasks[task_id],
            _trajectory(payload),
            leakage_checked=bool(original_grade.get("leakage_checked")),
        ).to_dict()
        legacy_success += bool(original_grade.get("success"))
        output.append({
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "seed": seed,
            "grade": result,
        })

    ids = [row["trajectory_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory_id in source store")
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({row["grade"]["split"] for row in output}):
        subset = [row["grade"] for row in output if row["grade"]["split"] == split]
        by_split[split] = _summary(subset)
    report = {
        "schema_version": 2,
        "automatic_grade_scope": "operational; natural-language answer quality requires human review",
        "source_store": str(store_path).replace("\\", "/"),
        "source_store_sha256": hashlib.sha256(store_path.read_bytes()).hexdigest(),
        "trajectory_count": len(output),
        "legacy_automatic_operational_success": legacy_success / len(output) if output else None,
        "regraded": _summary([row["grade"] for row in output]),
        "by_split": by_split,
    }
    return output, report


def _summary(grades: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(grades)
    failures = Counter(grade["failure_type"] for grade in grades if grade.get("failure_type"))
    return {
        "trajectories": count,
        "operational_success": sum(bool(grade["success"]) for grade in grades) / count if count else None,
        "policy_compliance": sum(bool(grade["policy_compliant"]) for grade in grades) / count if count else None,
        "terminal_state_accuracy": sum(bool(grade["terminal_state_match"]) for grade in grades) / count if count else None,
        "forbidden_tool_attempt_rate": sum(bool(grade.get("forbidden_tool_attempt")) for grade in grades) / count if count else None,
        "illegal_state_change_rate": sum(bool(grade.get("illegal_state_change")) for grade in grades) / count if count else None,
        "failure_taxonomy": dict(sorted(failures.items())),
    }


def write_outputs(rows: list[dict[str, Any]], report: dict[str, Any], grades_path: Path, report_path: Path) -> None:
    grades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    grades_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--output-grades", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    rows, report = regrade_store(args.tasks, args.store)
    write_outputs(rows, report, args.output_grades, args.output_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
