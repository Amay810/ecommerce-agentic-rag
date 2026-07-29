"""Freeze the preregistered 40-task terminal-grounding human-audit sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260818
TARGETS = {
    "compare": 6,
    "order_query": 6,
    "policy": 7,
    "product_qa": 7,
    "recommend": 6,
    "return": 7,
    "recovery_no_answer": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_rank(task_id: str) -> str:
    return hashlib.sha256(f"answer_postprocess_blind_audit_v1:{SEED}:{task_id}".encode()).hexdigest()


def template_family(task: dict[str, Any]) -> str:
    return str(task.get("metadata", {}).get("phase_a", {}).get("template_family", ""))


def select_tasks(tasks: dict[str, dict[str, Any]], grades: dict[str, dict[str, Any]],
                 smoke_ids: set[str], holdout_ids: list[str]) -> list[str]:
    fixed = [task_id for task_id in holdout_ids if grades[task_id].get("answer_fact_applicable")]
    if len(fixed) != 12:
        raise ValueError(f"expected 12 fact-applicable frozen holdout tasks, got {len(fixed)}")
    selected: list[str] = []
    for category, target in TARGETS.items():
        chosen = [task_id for task_id in fixed if tasks[task_id]["category"] == category]
        candidates = [
            task_id for task_id, task in tasks.items()
            if task.get("split") == "dev"
            and task.get("category") == category
            and grades.get(task_id, {}).get("answer_fact_applicable") is True
            and task_id not in smoke_ids
            and task_id not in chosen
        ]
        candidates.sort(key=stable_rank)
        seen = {template_family(tasks[task_id]) for task_id in chosen}
        diverse: list[str] = []
        remainder: list[str] = []
        for task_id in candidates:
            family = template_family(tasks[task_id])
            if family not in seen:
                diverse.append(task_id)
                seen.add(family)
            else:
                remainder.append(task_id)
        chosen.extend((diverse + remainder)[:target - len(chosen)])
        if len(chosen) != target:
            raise ValueError(f"{category}: expected {target} selected tasks, got {len(chosen)}")
        selected.extend(chosen)
    if len(selected) != 40 or len(set(selected)) != 40:
        raise ValueError("audit selection must contain exactly 40 unique tasks")
    if set(selected) & smoke_ids:
        raise ValueError("audit selection overlaps the smoke manifest")
    if not set(fixed).issubset(selected):
        raise ValueError("audit selection dropped a fact-applicable frozen holdout task")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen audit manifest: {args.output}")

    task_rows = read_jsonl(args.tasks)
    tasks = {row["task_id"]: row for row in task_rows}
    report = json.loads(args.base_report.read_text(encoding="utf-8"))
    grades = {row["task_id"]: row for row in report["details"]}
    smoke = json.loads(args.smoke_manifest.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    smoke_ids = set(smoke["task_ids"])
    selected = select_tasks(tasks, grades, smoke_ids, holdout["task_ids"])
    fixed = [task_id for task_id in holdout["task_ids"]
             if grades[task_id].get("answer_fact_applicable")]
    source = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "name": "answer_postprocess_blind_audit_v1",
        "status": "selection_frozen_before_grounded_outputs",
        "selection_seed": SEED,
        "sample_size_task_groups": 40,
        "expected_blinded_answers": 80,
        "category_targets": TARGETS,
        "selection_algorithm": (
            "include the 12 fact-applicable frozen holdout tasks; exclude all smoke tasks; "
            "within each category prefer unseen template families, then fill by seeded task-id hash"
        ),
        "fact_applicable_holdout_task_ids": fixed,
        "excluded_smoke_task_ids": sorted(smoke_ids),
        "selected_tasks": [
            {
                "task_id": task_id,
                "category": tasks[task_id]["category"],
                "template_family": template_family(tasks[task_id]),
                "scenario_family": tasks[task_id].get("metadata", {}).get("phase_a", {}).get("scenario_family"),
                "semantic_spec_hash": tasks[task_id].get("metadata", {}).get("phase_a", {}).get("semantic_spec_hash"),
                "from_frozen_holdout_v1": task_id in fixed,
                "answer_fact_applicable": True,
            }
            for task_id in selected
        ],
        "task_contract_sha256": sha256(args.tasks),
        "base_report_sha256": sha256(args.base_report),
        "smoke_manifest_sha256": sha256(args.smoke_manifest),
        "holdout_manifest_sha256": sha256(args.holdout_manifest),
        "selection_source_sha256": sha256(source),
        "repository_commit_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "task_groups": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
