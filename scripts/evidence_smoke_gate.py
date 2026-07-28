"""Fail-closed structural gate for the 24-task Phase-A smoke run."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


VARIANTS = ("base", "evidence_verify", "evidence_verify_repair")


def _load_store(path: Path) -> list[tuple[str, dict, dict]]:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    try:
        return [(task_id, json.loads(trajectory), json.loads(grade)) for task_id, trajectory, grade in
                conn.execute("SELECT task_id,trajectory_json,grade_json FROM trajectories ORDER BY task_id")]
    finally:
        conn.close()


def assess(manifest_path: Path, store_dir: Path, run_name: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = set(manifest.get("task_ids", []))
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for variant in VARIANTS:
        path = store_dir / f"{run_name}_{variant}.sqlite"
        try:
            rows = _load_store(path)
        except Exception as exc:  # fail closed on any malformed/missing store
            check(f"{variant}:store_readable", False, str(exc))
            continue
        ids = [row[0] for row in rows]
        check(f"{variant}:trajectory_count", len(rows) == len(expected), f"{len(rows)}/{len(expected)}")
        check(f"{variant}:task_ids", set(ids) == expected and len(ids) == len(set(ids)))
        generation_errors = 0
        fallback = 0
        too_many_repairs = 0
        illegal_changes = 0
        missing_ledger = 0
        for _task_id, trajectory, grade in rows:
            illegal_changes += int(bool(grade.get("illegal_state_change")))
            too_many_repairs += int(len(trajectory.get("repair_spans", [])) > 1)
            for model_call in trajectory.get("model_calls", []):
                trace = model_call.get("llm") or {}
                fallback += int(trace.get("resolution") == "fallback_handoff")
                trace_items = [trace]
                if isinstance(trace.get("repair_llm"), dict):
                    trace_items.append(trace["repair_llm"])
                for trace_item in trace_items:
                    fallback += int(trace_item is not trace and trace_item.get("resolution") == "fallback_handoff")
                    for attempt in trace_item.get("attempts", []):
                        generation_errors += int(attempt.get("parse_stage") == "generation_error")
            if variant != "base" and trajectory.get("retrievals") and not trajectory.get("evidence_ledger"):
                missing_ledger += 1
        check(f"{variant}:generation_errors_zero", generation_errors == 0, str(generation_errors))
        check(f"{variant}:fallbacks_zero", fallback == 0, str(fallback))
        check(f"{variant}:illegal_state_changes_zero", illegal_changes == 0, str(illegal_changes))
        check(f"{variant}:repair_at_most_once", too_many_repairs == 0, str(too_many_repairs))
        if variant != "base":
            check(f"{variant}:retrievals_have_ledger", missing_ledger == 0, str(missing_ledger))
    return {"schema_version": 1, "passed": all(row["passed"] for row in checks), "checks": checks}


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
