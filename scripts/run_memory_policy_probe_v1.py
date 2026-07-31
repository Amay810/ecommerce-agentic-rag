"""Run frozen memory_policy_probe_v1 off/on paired causal experiment.

Only Memory injection differs. Action Constraint stays on for both arms.
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
from ecommerce_rag.legacy_closure_benchmark import TypedScenarioUser, clone_database
from ecommerce_rag.llm_policy import LLMPolicy, SYSTEM_PROMPT
from ecommerce_rag.memory_policy_probe import (
    PROTOCOL,
    FROZEN_PROBE_TASK_SHA256,
    FROZEN_TRAIN_CASE_SHA256,
    ScriptedThenLLMPolicy,
    aggregate_probe,
    assert_frozen_manifests,
    build_probe_tasks,
    conclude,
    offline_preferred,
    prepare_probe_database,
    resolve_probe_step,
    score_pair,
    scripted_prefix_for,
    seed_train_cases,
    to_probe_task_spec,
)
from ecommerce_rag.tool_schema import TOOL_SCHEMAS


def _hash(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if not isinstance(value, str) else value
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def _resolve_commit(value: str) -> str:
    """Accept short or full SHAs via git rev-parse."""
    try:
        return _git("rev-parse", "--verify", value)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot resolve expected commit: {value}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--case-db", type=Path, required=True)
    parser.add_argument("--seed-train", action="store_true")
    parser.add_argument("--approved-by", default="probe_operator")
    parser.add_argument(
        "--approval-reason",
        default=(
            "memory_policy_probe_v1 curated_contract_seed "
            "(experience_case=false; not trajectory paired replay)"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    commit = _git("rev-parse", "HEAD")
    expected = _resolve_commit(args.expected_code_commit)
    if commit != expected:
        raise ValueError(f"code commit drift: {commit} != {expected}")
    if _git("status", "--porcelain"):
        raise ValueError("memory_policy_probe_v1 requires a clean worktree")
    assert_frozen_manifests()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    if args.seed_train or not args.case_db.exists():
        args.case_db.parent.mkdir(parents=True, exist_ok=True)
        seed_rows = seed_train_cases(
            args.case_db,
            approved_by=args.approved_by,
            approval_reason=args.approval_reason,
            approve=True,
        )
    else:
        seed_rows = []

    tasks = build_probe_tasks()
    pristine = prepare_probe_database(tasks, args.output_dir / "pristine_probe.sqlite")

    template = LLMPolicy.from_env()
    meta = dict(template.generator_meta)
    if not str(meta.get("model", "")).endswith("Qwen3-4B-Instruct-2507"):
        raise ValueError("memory_policy_probe_v1 requires Qwen3-4B-Instruct-2507")
    if meta.get("max_new_tokens") != 1024:
        raise ValueError("memory_policy_probe_v1 requires max_new_tokens=1024")

    pairs: list[dict[str, Any]] = []
    for task in tasks:
        spec = to_probe_task_spec(task)
        prefix = scripted_prefix_for(task)
        arm_rows: dict[str, Any] = {}
        for arm, memory_on in (("memory_off", False), ("memory_on", True)):
            database = clone_database(
                pristine,
                args.output_dir / "databases" / arm / f"{task.task_id}.sqlite",
            )
            base = LLMPolicy(template.generate, generator_meta=meta)
            policy = ScriptedThenLLMPolicy(prefix, base) if prefix else base
            trajectory, grade = HarnessRunner(
                database,
                policy=policy,
                max_steps=args.max_steps,
                progress_reducer=LegacyTaskProgressReducer(),
                expose_task_progress=True,
                enforce_action_constraint=True,
                enable_case_memory=memory_on,
                case_memory_db=args.case_db,
                enable_case_writeback=False,
                user_simulator_factory=lambda _t=task: TypedScenarioUser(_t.user_responses),
            ).run(spec)
            probe_step = resolve_probe_step(task, policy)
            arm_rows[arm] = {
                "trajectory": trajectory,
                "grade": grade,
                "probe_step": probe_step,
            }
            (args.output_dir / "trajectories" / arm).mkdir(parents=True, exist_ok=True)
            (args.output_dir / "trajectories" / arm / f"{task.task_id}.json").write_text(
                json.dumps(trajectory.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        off = arm_rows["memory_off"]
        on = arm_rows["memory_on"]
        off_decision = next(
            (s for s in off["trajectory"].decision_spans
             if s.get("step") == off["probe_step"]),
            {},
        )
        preferred_payload = offline_preferred(
            off_decision.get("progress") or {
                "workflow": "return_resolution",
                "allowed_next_actions": [task.preferred_action],
                "pending": [],
                "guard_state": "",
                "blocked_by": "user_input",
            },
            db_path=args.case_db,
            fallback_expected_action=task.preferred_action,
        )
        pair = score_pair(
            task=task,
            off_trajectory=off["trajectory"],
            on_trajectory=on["trajectory"],
            off_grade=off["grade"],
            on_grade=on["grade"],
            probe_step_off=off["probe_step"],
            probe_step_on=on["probe_step"],
            offline_preferred_payload=preferred_payload,
        )
        pairs.append(pair)

    summary = aggregate_probe(pairs)
    verdict = conclude(summary)
    report = {
        "protocol": PROTOCOL,
        "code_commit": commit,
        "probe_task_sha256": FROZEN_PROBE_TASK_SHA256,
        "train_case_sha256": FROZEN_TRAIN_CASE_SHA256,
        "system_prompt_sha256": _hash(SYSTEM_PROMPT),
        "tool_schema_sha256": _hash(TOOL_SCHEMAS),
        "generator_meta": meta,
        "case_db": str(args.case_db),
        "seed_train_rows": seed_rows,
        "n_tasks": len(tasks),
        "summary": summary,
        "verdict": verdict,
        "pairs": pairs,
        "claim_boundary": {
            "engineering_loop": "complete_v1.1",
            "policy_gain": verdict["policy_memory_gain"],
            "formal_dev_40_40": "untouched",
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.output_dir / "pairs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs),
        encoding="utf-8",
    )
    print(json.dumps({"protocol": PROTOCOL, "verdict": verdict, "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
