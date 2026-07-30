"""Run the dev-only fixed-progress versus ActionEvaluator responsibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ecommerce_rag.domain import AgentAction
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure import LegacyActionEvaluator, LegacyTaskProgressReducer
from ecommerce_rag.legacy_closure_benchmark import (
    FROZEN_TASK_SHA256,
    FROZEN_ACTION_CORRECTION_CHALLENGE_SHA256,
    TypedScenarioUser,
    action_evaluator_gate,
    aggregate,
    build_action_correction_challenges,
    build_m1_tasks,
    clone_database,
    grade_record,
    prepare_database,
    to_task_spec,
    trajectory_record,
)
from ecommerce_rag.llm_policy import LLMPolicy, SYSTEM_PROMPT
from ecommerce_rag.tool_schema import TOOL_SCHEMAS


CONFIGS = ("legacy_progress_fixed", "legacy_progress_action_eval")


class InjectedFirstActionPolicy:
    """Inject one known-bad action, then delegate correction and recovery to Qwen."""

    privileged = False

    def __init__(self, delegate: LLMPolicy):
        self.delegate = delegate
        self.injected = False
        self.last_trace: dict[str, Any] = {}

    @property
    def retry_count(self) -> int:
        return self.delegate.retry_count

    @property
    def max_parse_retries(self) -> int:
        return self.delegate.max_parse_retries

    @max_parse_retries.setter
    def max_parse_retries(self, value: int) -> None:
        self.delegate.max_parse_retries = value

    def act(self, observation):
        if not self.injected:
            self.injected = True
            action = AgentAction.handoff("injected_recoverable_handoff")
            self.last_trace = {
                "resolution": "preregistered_challenge_injection",
                "attempts": [],
                "action": action.action_type,
            }
            return action
        action = self.delegate.act(observation)
        self.last_trace = self.delegate.last_trace
        return action


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
    args = parser.parse_args()

    commit = _git("rev-parse", "HEAD")
    if commit != args.expected_code_commit:
        raise ValueError(f"code commit drift: {commit} != {args.expected_code_commit}")
    if _git("status", "--porcelain"):
        raise ValueError("formal dev requires a clean worktree")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tasks = [task for task in build_m1_tasks() if task.split == "dev"]
    if len(tasks) != 40 or any(task.split != "dev" for task in tasks):
        raise ValueError("action evaluator dev requires exactly 40 dev tasks")
    pristine = prepare_database(tasks, args.output_dir / "pristine_dev.sqlite")

    template = LLMPolicy.from_env()
    meta = dict(template.generator_meta)
    if not str(meta.get("model", "")).endswith("Qwen3-4B-Instruct-2507"):
        raise ValueError("formal action evaluator dev requires Qwen3-4B-Instruct-2507")
    if meta.get("max_new_tokens") != 1024:
        raise ValueError("formal action evaluator dev requires max_new_tokens=1024")
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
                action_evaluator=(LegacyActionEvaluator()
                                  if config == "legacy_progress_action_eval" else None),
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
    fixed_grades = [row for row in grades if row["config"] == "legacy_progress_fixed"]
    evaluated_grades = [row for row in grades
                        if row["config"] == "legacy_progress_action_eval"]
    fixed_records = [row for row in records if row["config"] == "legacy_progress_fixed"]

    challenge_tasks = build_action_correction_challenges()
    challenge_root = args.output_dir / "correction_challenge"
    challenge_pristine = prepare_database(
        challenge_tasks, challenge_root / "pristine.sqlite")
    challenge_records: list[dict[str, Any]] = []
    challenge_grades: list[dict[str, Any]] = []
    for task in challenge_tasks:
        database = clone_database(
            challenge_pristine, challenge_root / "databases" / f"{task.task_id}.sqlite")
        policy = InjectedFirstActionPolicy(LLMPolicy(template.generate, generator_meta=meta))
        trajectory, harness_grade = HarnessRunner(
            database,
            policy=policy,
            max_steps=8,
            progress_reducer=LegacyTaskProgressReducer(),
            expose_task_progress=True,
            action_evaluator=LegacyActionEvaluator(),
            user_simulator_factory=lambda _spec, current=task: TypedScenarioUser(
                current.user_responses),
        ).run(to_task_spec(task))
        record = trajectory_record(task, "correction_challenge", trajectory, harness_grade)
        grade = grade_record(task, record)
        challenge_records.append(record)
        challenge_grades.append(grade)
    with (challenge_root / "records.jsonl").open("x", encoding="utf-8") as handle:
        for record in challenge_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (challenge_root / "grades.jsonl").open("x", encoding="utf-8") as handle:
        for grade in challenge_grades:
            handle.write(json.dumps(grade, ensure_ascii=False) + "\n")
    challenge_summary = aggregate(challenge_grades)
    challenge_tokens = [_tokens(record) for record in challenge_records]
    challenge_summary.update(
        prompt_tokens=sum(row[0] for row in challenge_tokens),
        completion_tokens=sum(row[1] for row in challenge_tokens),
    )
    gate = action_evaluator_gate(
        summaries["legacy_progress_fixed"],
        summaries["legacy_progress_action_eval"],
        fixed_grades,
        evaluated_grades,
        fixed_records,
        challenge_summary,
    )
    report = {
        "schema_version": 1,
        "protocol": "legacy_task_closure_action_eval_dev_v1",
        "split": "dev",
        "locked_executed": False,
        "code_commit": commit,
        "task_manifest_sha256": FROZEN_TASK_SHA256,
        "correction_challenge_manifest_sha256": FROZEN_ACTION_CORRECTION_CHALLENGE_SHA256,
        "task_count": len(tasks),
        "correction_challenge_task_count": len(challenge_tasks),
        "config_order": list(CONFIGS),
        "model": meta.get("model"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "generation_config_hash": _hash(meta),
        "system_prompt_sha256": _hash(SYSTEM_PROMPT),
        "tool_schema_sha256": _hash(TOOL_SCHEMAS),
        "fixed_reducer_change": "return_reason_refused is blocked and never stored as a reason",
        "intentional_context_difference": (
            "action-evaluated configuration receives deterministic feedback only after a rejected action"),
        "action_evaluator_scope": [
            "allowed progress actions", "return reason before write",
            "explicit confirmation before write", "handoff appropriateness",
        ],
        "summaries": summaries,
        "correction_challenge": {
            "excluded_from_main_success_rate": True,
            "injection": "one preregistered inappropriate handoff before Qwen correction",
            "summary": challenge_summary,
        },
        "gate": gate,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
