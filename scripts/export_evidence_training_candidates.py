"""Export deduplicated calibration-only candidates; this is not training data approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def _key(kind: str, payload: dict) -> str:
    canonical = json.dumps({"kind": kind, **payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--stores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    tasks = {row["task_id"]: row for row in (
        json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip())}
    seen, candidates = set(), []

    def add(kind: str, task_id: str, payload: dict) -> None:
        digest = _key(kind, payload)
        if digest in seen:
            return
        seen.add(digest)
        candidates.append({"candidate_id": digest[:20], "kind": kind, "task_id": task_id,
                           "split": "calibration", **payload})

    for store in args.stores:
        conn = sqlite3.connect(store)
        try:
            rows = conn.execute("SELECT task_id,trajectory_json,grade_json FROM trajectories").fetchall()
        finally:
            conn.close()
        for task_id, trajectory_json, grade_json in rows:
            task = tasks.get(task_id)
            if not task:
                raise ValueError(f"unknown task id in {store}: {task_id}")
            if task.get("split") != "calibration":
                raise ValueError(f"refusing to export non-calibration task: {task_id}")
            trajectory, grade = json.loads(trajectory_json), json.loads(grade_json)
            observations, actions = trajectory.get("observations", []), trajectory.get("actions", [])
            for observation, action in zip(observations, actions):
                add("observation_to_action", task_id, {"observation": observation, "chosen": action})
            if grade.get("answer_fact_pass") and trajectory.get("evidence_ledger") and trajectory.get("final_answer"):
                add("tool_evidence_to_grounded_answer", task_id, {
                    "evidence_ledger": trajectory["evidence_ledger"], "chosen": trajectory["final_answer"]})
            for span in trajectory.get("repair_spans", []):
                if not span.get("passed"):
                    continue
                add("failed_verification_to_repair", task_id, {
                    "observation": span.get("requested_reason"),
                    "rejected": span.get("original_answer"),
                    "chosen": (span.get("action") or {}).get("content"),
                })

    counts = {}
    for row in candidates:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    manifest = {
        "schema_version": 1, "candidate_count": len(candidates), "counts": counts,
        "deduplicated": True, "source_splits": ["calibration"],
        "status": "candidates_only_not_approved_for_sft_or_dpo",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates), encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
