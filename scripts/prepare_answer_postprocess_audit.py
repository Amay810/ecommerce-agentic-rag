"""Prepare 32 unique answers plus 16 shadow records for frozen holdout review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    return {row["task_id"]: row for row in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--terminal-grounded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    task_ids = manifest["task_ids"]
    shadow, grounded = rows(args.shadow), rows(args.terminal_grounded)
    if set(shadow) != set(grounded) or not set(task_ids).issubset(shadow):
        raise ValueError("paired sidecars do not cover the frozen holdout")
    answers, diagnostics = [], []
    for task_id in task_ids:
        s, g = shadow[task_id], grounded[task_id]
        if s["draft_answer"] != g["draft_answer"]:
            raise ValueError(f"base draft mismatch for {task_id}")
        common = {"task_id": task_id, "eligible": g["eligible"],
                  "ineligible_reason": g["ineligible_reason"]}
        answers.extend([
            {**common, "answer_id": f"{task_id}:base", "variant": "base",
             "answer": s["draft_answer"], "human_fact_status": "", "human_answer_complete": "",
             "human_overall_pass": "", "review_notes": ""},
            {**common, "answer_id": f"{task_id}:terminal_grounded", "variant": "terminal_grounded",
             "answer": g["final_answer"], "human_fact_status": "", "human_answer_complete": "",
             "human_overall_pass": "", "review_notes": ""},
        ])
        diagnostics.append({**common, "shadow_verification": s["verification"]})
    if len(answers) != 32 or len(diagnostics) != 16:
        raise ValueError("holdout audit must contain 32 unique answers and 16 shadow records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = args.output_dir / "answer_postprocess_holdout_v1_answers.jsonl"
    shadow_path = args.output_dir / "answer_postprocess_holdout_v1_shadow_diagnostics.jsonl"
    answer_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in answers),
                           encoding="utf-8")
    shadow_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in diagnostics),
                           encoding="utf-8")
    audit_manifest = {
        "schema_version": 1, "task_groups": 16, "independent_answers": 32,
        "shadow_diagnostic_records": 16, "holdout_manifest_sha256": sha(args.holdout_manifest),
        "shadow_sidecar_sha256": sha(args.shadow), "terminal_grounded_sidecar_sha256": sha(args.terminal_grounded),
        "answers_jsonl_sha256": sha(answer_path), "shadow_diagnostics_jsonl_sha256": sha(shadow_path),
        "shadow_is_not_a_third_answer": True,
    }
    (args.output_dir / "answer_postprocess_holdout_v1_audit_manifest.json").write_text(
        json.dumps(audit_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
