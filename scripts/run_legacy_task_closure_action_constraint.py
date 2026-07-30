"""Paired formal run: protocol-fixed progress vs dynamic action constraint.

Protocol: legacy_task_closure_action_constraint_dev_v1
Does not overwrite protocol_fix or ActionEvaluator archives.
ActionEvaluator and SemanticFactExtractor stay disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure import LegacyTaskProgressReducer
from ecommerce_rag.legacy_closure_benchmark import (
    FROZEN_TASK_SHA256,
    PROTOCOL_FIX_OBSERVE_POLICY_TASK_IDS,
    PROTOCOL_FIX_TARGET_TASK_IDS,
    TypedScenarioUser,
    action_constraint_gate,
    aggregate,
    build_m1_tasks,
    clone_database,
    grade_record,
    prepare_database,
    to_task_spec,
    trajectory_record,
)
from ecommerce_rag.llm_policy import LLMPolicy, SYSTEM_PROMPT
from ecommerce_rag.tool_schema import TOOL_SCHEMAS

PROTOCOL = "legacy_task_closure_action_constraint_dev_v1"
CONFIGS = ("legacy_progress_protocol_fixed", "legacy_progress_constrained")


def _hash(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
               if not isinstance(value, str) else value)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def _tokens(record: dict[str, Any]) -> tuple[int, int]:
    prompt = completion = 0
    for call in record.get("model_calls", []):
        for attempt in (call.get("llm") or {}).get("attempts", []):
            prompt += int(attempt.get("prompt_tokens") or 0)
            completion += int(attempt.get("completion_tokens") or 0)
    return prompt, completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument(
        "--protocol-fix-commit",
        default="5fb8448c59f0f93f3477c944a2fed16cf02f0e39",
        help="Commit that produced accept_protocol_fix 38/40; lineage only.",
    )
    args = parser.parse_args()

    commit = _git("rev-parse", "HEAD")
    if commit != args.expected_code_commit:
        raise ValueError(f"code commit drift: {commit} != {args.expected_code_commit}")
    if _git("status", "--porcelain"):
        raise ValueError("formal action_constraint requires a clean worktree")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tasks = [task for task in build_m1_tasks() if task.split == "dev"]
    if len(tasks) != 40 or any(task.split != "dev" for task in tasks):
        raise ValueError("action_constraint requires exactly 40 dev tasks")
    pristine = prepare_database(tasks, args.output_dir / "pristine_dev.sqlite")

    template = LLMPolicy.from_env()
    meta = dict(template.generator_meta)
    if not str(meta.get("model", "")).endswith("Qwen3-4B-Instruct-2507"):
        raise ValueError("formal action_constraint requires Qwen3-4B-Instruct-2507")
    if meta.get("max_new_tokens") != 1024:
        raise ValueError("formal action_constraint requires max_new_tokens=1024")
    policies = {name: LLMPolicy(template.generate, generator_meta=meta) for name in CONFIGS}

    records: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    for task in tasks:
        for config in CONFIGS:
            database = clone_database(
                pristine, args.output_dir / "databases" / config / f"{task.task_id}.sqlite")
            trajectory, harness_grade = HarnessRunner(
                database,
                policy=policies[config],
                max_steps=8,
                progress_reducer=LegacyTaskProgressReducer(),
                expose_task_progress=True,
                enforce_action_constraint=(config == "legacy_progress_constrained"),
                user_simulator_factory=lambda _spec, current=task: TypedScenarioUser(
                    current.user_responses),
            ).run(to_task_spec(task))
            record = trajectory_record(task, config, trajectory, harness_grade)
            grade = grade_record(task, record)
            records.append(record)
            grades.append(grade)

    with (args.output_dir / "records.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "grades.jsonl").open("x", encoding="utf-8") as handle:
        for grade in grades:
            handle.write(json.dumps(grade, ensure_ascii=False) + "\n")

    summaries = {
        config: aggregate(grade for grade in grades if grade["config"] == config)
        for config in CONFIGS
    }
    for config in CONFIGS:
        counts = [_tokens(record) for record in records if record["config"] == config]
        summaries[config].update(
            prompt_tokens=sum(row[0] for row in counts),
            completion_tokens=sum(row[1] for row in counts),
        )
    fixed_grades = [row for row in grades if row["config"] == CONFIGS[0]]
    constrained_grades = [row for row in grades if row["config"] == CONFIGS[1]]
    constrained_records = [row for row in records if row["config"] == CONFIGS[1]]
    gate = action_constraint_gate(
        summaries[CONFIGS[0]], summaries[CONFIGS[1]],
        fixed_grades, constrained_grades, constrained_records,
        locked_executed=False,
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "split": "dev",
        "locked_executed": False,
        "code_commit": commit,
        "protocol_fix_commit": args.protocol_fix_commit,
        "task_manifest_sha256": FROZEN_TASK_SHA256,
        "task_count": len(tasks),
        "config_order": list(CONFIGS),
        "preregistered_protocol_targets": sorted(PROTOCOL_FIX_TARGET_TASK_IDS),
        "preregistered_observe_policy": sorted(PROTOCOL_FIX_OBSERVE_POLICY_TASK_IDS),
        "disabled_components": [
            "ActionEvaluator",
            "SemanticFactExtractor",
            "CompletionEvaluator",
            "locked_split",
        ],
        "model": meta.get("model"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "generation_config_hash": _hash(meta),
        "system_prompt_sha256": _hash(SYSTEM_PROMPT),
        "tool_schema_sha256": _hash(TOOL_SCHEMAS),
        "constraint_semantics": (
            "TaskProgress allowlist; illegal actions remapped once to preferred "
            "or fail-closed; no second LLM correction call"
        ),
        "summaries": summaries,
        "gate": gate,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
