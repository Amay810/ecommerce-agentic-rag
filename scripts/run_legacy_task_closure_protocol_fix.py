"""Formal Qwen protocol-fix run: legacy_task_closure_protocol_fix_dev_v1.

Does not overwrite archived legacy_progress_fixed artifacts.  SemanticFactExtractor
and ActionEvaluator stay disabled.  Locked is never selected.
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
    aggregate,
    build_m1_tasks,
    clone_database,
    grade_record,
    prepare_database,
    protocol_fix_gate,
    to_task_spec,
    trajectory_record,
)
from ecommerce_rag.llm_policy import LLMPolicy, SYSTEM_PROMPT
from ecommerce_rag.tool_schema import TOOL_SCHEMAS

PROTOCOL = "legacy_task_closure_protocol_fix_dev_v1"
CONFIG = "legacy_task_closure_protocol_fix"


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
        "--formal-baseline-commit",
        default="ff6af987ff034ec3140679070038ae928ec65ca0",
        help="Archived 34/40 evaluated commit; recorded for lineage, not executed here.",
    )
    args = parser.parse_args()

    commit = _git("rev-parse", "HEAD")
    if commit != args.expected_code_commit:
        raise ValueError(f"code commit drift: {commit} != {args.expected_code_commit}")
    if _git("status", "--porcelain"):
        raise ValueError("formal protocol_fix requires a clean worktree")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tasks = [task for task in build_m1_tasks() if task.split == "dev"]
    if len(tasks) != 40 or any(task.split != "dev" for task in tasks):
        raise ValueError("protocol_fix requires exactly 40 dev tasks")
    pristine = prepare_database(tasks, args.output_dir / "pristine_dev.sqlite")

    template = LLMPolicy.from_env()
    meta = dict(template.generator_meta)
    if not str(meta.get("model", "")).endswith("Qwen3-4B-Instruct-2507"):
        raise ValueError("formal protocol_fix requires Qwen3-4B-Instruct-2507")
    if meta.get("max_new_tokens") != 1024:
        raise ValueError("formal protocol_fix requires max_new_tokens=1024")
    policy = LLMPolicy(template.generate, generator_meta=meta)

    records: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    for task in tasks:
        database = clone_database(
            pristine, args.output_dir / "databases" / CONFIG / f"{task.task_id}.sqlite")
        trajectory, harness_grade = HarnessRunner(
            database,
            policy=policy,
            max_steps=8,
            progress_reducer=LegacyTaskProgressReducer(),
            expose_task_progress=True,
            # ActionEvaluator and SemanticFactExtractor remain off.
            user_simulator_factory=lambda _spec, current=task: TypedScenarioUser(
                current.user_responses),
        ).run(to_task_spec(task))
        record = trajectory_record(task, CONFIG, trajectory, harness_grade)
        grade = grade_record(task, record)
        records.append(record)
        grades.append(grade)

    with (args.output_dir / "records.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "grades.jsonl").open("x", encoding="utf-8") as handle:
        for grade in grades:
            handle.write(json.dumps(grade, ensure_ascii=False) + "\n")

    summary = aggregate(grades)
    prompt_tokens = completion_tokens = 0
    for record in records:
        prompt, completion = _tokens(record)
        prompt_tokens += prompt
        completion_tokens += completion
    summary.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    gate = protocol_fix_gate(summary, grades, locked_executed=False)

    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "config": CONFIG,
        "split": "dev",
        "locked_executed": False,
        "code_commit": commit,
        "formal_baseline_commit": args.formal_baseline_commit,
        "baseline_result_identity": {
            "config": "legacy_progress_fixed",
            "success": "34/40",
            "illegal_state_change": 0,
            "note": "Archived result; not overwritten by this run.",
        },
        "task_manifest_sha256": FROZEN_TASK_SHA256,
        "task_count": len(tasks),
        "preregistered_targets": sorted(PROTOCOL_FIX_TARGET_TASK_IDS),
        "preregistered_observe_only": sorted(PROTOCOL_FIX_OBSERVE_POLICY_TASK_IDS),
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
        "endpoint_drift_note": (
            "If ARAG_AGENT_BACKEND is not a frozen local weight path, also run the "
            "archived formal_baseline_commit in the same environment before claiming "
            "a code-only gain."
        ),
        "summary": summary,
        "gate": gate,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
