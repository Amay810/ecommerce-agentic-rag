"""Run the dev-only legacy baseline/TaskProgress responsibility gate."""

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
    TypedScenarioUser,
    aggregate,
    build_m1_tasks,
    clone_database,
    grade_record,
    prepare_database,
    progress_gate,
    to_task_spec,
    trajectory_record,
)
from ecommerce_rag.llm_policy import LLMPolicy, SYSTEM_PROMPT
from ecommerce_rag.tool_schema import TOOL_SCHEMAS


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) if not isinstance(value, str) else value
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
    args = parser.parse_args()

    commit = _git("rev-parse", "HEAD")
    if commit != args.expected_code_commit:
        raise ValueError(f"code commit drift: {commit} != {args.expected_code_commit}")
    if _git("status", "--porcelain"):
        raise ValueError("formal dev requires a clean worktree")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    all_tasks = build_m1_tasks()
    tasks = [task for task in all_tasks if task.split == "dev"]
    if len(tasks) != 40:
        raise ValueError("progress dev requires exactly 40 dev tasks")
    pristine = prepare_database(tasks, args.output_dir / "pristine_dev.sqlite")

    template = LLMPolicy.from_env()
    meta = dict(template.generator_meta)
    if not str(meta.get("model", "")).endswith("Qwen3-4B-Instruct-2507"):
        raise ValueError("formal progress dev requires Qwen3-4B-Instruct-2507")
    if meta.get("max_new_tokens") != 1024:
        raise ValueError("formal progress dev requires max_new_tokens=1024")
    policies = {
        name: LLMPolicy(template.generate, generator_meta=meta)
        for name in ("legacy_baseline", "legacy_progress")
    }
    records: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    for task in tasks:
        for config in ("legacy_baseline", "legacy_progress"):
            database = clone_database(pristine, args.output_dir / "databases" / config / f"{task.task_id}.sqlite")
            reducer = LegacyTaskProgressReducer() if config == "legacy_progress" else None
            trajectory, harness_grade = HarnessRunner(
                database, policy=policies[config], max_steps=8,
                progress_reducer=reducer,
                expose_task_progress=config == "legacy_progress",
                user_simulator_factory=lambda _spec, current=task: TypedScenarioUser(current.user_responses),
            ).run(to_task_spec(task))
            record = trajectory_record(task, config, trajectory, harness_grade)
            grade = grade_record(task, record)
            records.append(record)
            grades.append(grade)

    record_path = args.output_dir / "records.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    grade_path = args.output_dir / "grades.jsonl"
    with grade_path.open("x", encoding="utf-8") as handle:
        for grade in grades:
            handle.write(json.dumps(grade, ensure_ascii=False) + "\n")

    summaries = {
        config: aggregate(grade for grade in grades if grade["config"] == config)
        for config in ("legacy_baseline", "legacy_progress")
    }
    token_counts = {}
    for config in summaries:
        selected = [record for record in records if record["config"] == config]
        counts = [_tokens(record) for record in selected]
        token_counts[config] = {"prompt_tokens": sum(item[0] for item in counts),
                                "completion_tokens": sum(item[1] for item in counts)}
        summaries[config].update(token_counts[config])
    progress_records = [record for record in records if record["config"] == "legacy_progress"]
    gate = progress_gate(summaries["legacy_baseline"], summaries["legacy_progress"], progress_records)
    report = {
        "schema_version": 1,
        "protocol": "legacy_task_closure_progress_dev_v1",
        "split": "dev",
        "locked_executed": False,
        "code_commit": commit,
        "task_manifest_sha256": FROZEN_TASK_SHA256,
        "task_count": len(tasks),
        "config_order": ["legacy_baseline", "legacy_progress"],
        "model": meta.get("model"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "generation_config_hash": _hash(meta),
        "system_prompt_sha256": _hash(SYSTEM_PROMPT),
        "tool_schema_sha256": _hash(TOOL_SCHEMAS),
        "intentional_context_difference": "legacy_progress exposes deterministic task_progress in SESSION",
        "summaries": summaries,
        "gate": gate,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
