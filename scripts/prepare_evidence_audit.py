"""Join the 32-row audit scaffold to one complete calibration trajectory store."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.template.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []
    conn = sqlite3.connect(args.store)
    try:
        stored = {task_id: (json.loads(trajectory), json.loads(grade)) for task_id, trajectory, grade in
                  conn.execute("SELECT task_id,trajectory_json,grade_json FROM trajectories")}
    finally:
        conn.close()
    expected = {row["task_id"] for row in rows}
    missing, extra = expected - set(stored), set(stored) - expected
    if missing or extra or len(rows) != len(expected):
        raise ValueError(f"audit/store mismatch: missing={sorted(missing)}, extra={sorted(extra)}, duplicate_rows={len(rows)-len(expected)}")
    for row in rows:
        trajectory, grade = stored[row["task_id"]]
        row["variant"] = args.variant
        row["final_answer"] = trajectory.get("final_answer", "")
        auto_fact = grade.get("answer_fact_pass")
        row["auto_answer_fact_pass"] = "unknown" if auto_fact is None else str(bool(auto_fact)).lower()
        row["auto_contradiction_detected"] = str(bool(grade.get("contradicted_claims"))).lower()
        row["auto_verifier_details"] = json.dumps({
            "unsupported": grade.get("unsupported_high_risk_claims", []),
            "contradicted": grade.get("contradicted_claims", []),
            "omitted": grade.get("omitted_required_facts", []),
        }, ensure_ascii=False, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "variant": args.variant, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
