"""Frozen helpers for the tau3 Retail post-training v1 experiment."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
EXPECTED_SPLITS = {"train": 74, "test": 40, "base": 114}


def git_head(repository: Path) -> str:
    """Return the repository HEAD without changing the worktree."""
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.resolve().as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def validate_tau2_checkout(tau_root: Path) -> dict[str, int]:
    """Fail closed when the benchmark commit or Retail splits drift."""
    actual_commit = git_head(tau_root)
    if actual_commit != TAU2_COMMIT:
        raise ValueError(f"tau2 commit drift: {actual_commit} != {TAU2_COMMIT}")

    split_path = tau_root / "data" / "tau2" / "domains" / "retail" / "split_tasks.json"
    task_path = tau_root / "data" / "tau2" / "domains" / "retail" / "tasks.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    actual = {name: len(ids) for name, ids in splits.items()}
    if actual != EXPECTED_SPLITS:
        raise ValueError(f"retail split drift: {actual} != {EXPECTED_SPLITS}")
    if len(tasks) != EXPECTED_SPLITS["base"]:
        raise ValueError(
            f"retail task count drift: {len(tasks)} != {EXPECTED_SPLITS['base']}"
        )
    return actual


def build_tau2_command(
    *,
    tau_python: Path,
    launcher_script: Path,
    phase: str,
    agent_model: str,
    user_model: str,
    pass_k: int,
    save_to: str,
    max_steps: int = 200,
    agent_name: str = "llm_agent",
    task_ids: Iterable[str] | None = None,
) -> list[str]:
    """Build the only command shapes allowed by the frozen v1 protocol."""
    if phase not in {"smoke", "teacher", "base", "sft"}:
        raise ValueError(f"unsupported phase: {phase}")
    if pass_k < 1:
        raise ValueError("pass_k must be positive")
    if max_steps < 20:
        raise ValueError("max_steps below 20 can truncate Retail conversations")
    if phase == "smoke" and pass_k != 1:
        raise ValueError("smoke is cost calibration only and must use pass_k=1")
    if agent_name not in {"llm_agent", "ecommerce_native"}:
        raise ValueError(f"unsupported agent: {agent_name}")

    command = [
        str(tau_python),
        str(launcher_script),
        "run",
        "--domain",
        "retail",
        "--task-set-name",
        "retail",
        "--task-split-name",
        "train" if phase in {"smoke", "teacher"} else "test",
        "--num-trials",
        str(pass_k),
        "--agent",
        agent_name,
        "--agent-llm",
        agent_model,
        "--user",
        "user_simulator",
        "--user-llm",
        user_model,
        "--max-steps",
        str(max_steps),
        "--save-to",
        save_to,
        "--auto-resume",
        "--verbose-logs",
        "--max-concurrency",
        "1" if phase == "smoke" else "3",
    ]
    selected_task_ids = [str(value) for value in (task_ids or [])]
    if selected_task_ids:
        command.extend(["--task-ids", *selected_task_ids])
    elif phase == "smoke":
        command.extend(["--num-tasks", "5"])
    return command


def _message_usage(simulations: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    for simulation in simulations:
        for message in simulation.get("messages") or []:
            usage = message.get("usage") or {}
            for name in totals:
                totals[name] += int(usage.get(name) or 0)
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def annotate_results(
    result_path: Path,
    *,
    phase: str,
    agent_model: str,
    user_model: str,
    nl_assertions_model: str,
    pass_k: int,
    wall_clock_seconds: float,
    expected_task_count: int | None = None,
) -> dict[str, Any]:
    """Embed required provenance and smoke cost statistics in a tau2 result."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    simulations = payload.get("simulations") or []
    expected_tasks = expected_task_count or (
        5
        if phase == "smoke"
        else EXPECTED_SPLITS["train"]
        if phase == "teacher"
        else EXPECTED_SPLITS["test"]
    )
    expected_count = expected_tasks * pass_k
    if len(simulations) != expected_count:
        raise ValueError(
            f"incomplete {phase} result: {len(simulations)} != {expected_count}"
        )
    info = payload.get("info") or {}
    recorded_commit = info.get("git_commit")
    if recorded_commit and recorded_commit != TAU2_COMMIT:
        raise ValueError(f"result commit drift: {recorded_commit} != {TAU2_COMMIT}")
    recorded_agent = (info.get("agent_info") or {}).get("llm")
    recorded_user = (info.get("user_info") or {}).get("llm")
    if recorded_agent and recorded_agent != agent_model:
        raise ValueError(f"result agent drift: {recorded_agent} != {agent_model}")
    if recorded_user and recorded_user != user_model:
        raise ValueError(f"result user drift: {recorded_user} != {user_model}")
    rewards = [
        (simulation.get("reward_info") or {}).get("reward")
        for simulation in simulations
    ]
    numeric_rewards = [float(value) for value in rewards if value is not None]
    infrastructure_errors = sum(
        simulation.get("termination_reason") == "infrastructure_error"
        for simulation in simulations
    )
    action_checks = [
        check
        for simulation in simulations
        for check in (simulation.get("reward_info") or {}).get("action_checks") or []
    ]
    write_checks = [
        check for check in action_checks if check.get("tool_type") == "write"
    ]
    failed_write_checks = sum(
        not check.get("action_match", False) for check in write_checks
    )
    db_mismatches = sum(
        not db_check.get("db_match", False)
        for simulation in simulations
        if (db_check := (simulation.get("reward_info") or {}).get("db_check"))
    )
    usage = _message_usage(simulations)
    count = len(simulations)
    summary = {
        "num_simulations": count,
        "passed": sum(value == 1.0 for value in numeric_rewards),
        "infrastructure_errors": infrastructure_errors,
        "valid": infrastructure_errors == 0,
        "mean_reward": sum(numeric_rewards) / count if count else None,
        "write_action_checks": len(write_checks),
        "failed_write_action_checks": failed_write_checks,
        "db_mismatches": db_mismatches,
        "wall_clock_seconds": wall_clock_seconds,
        "mean_wall_clock_seconds": wall_clock_seconds / count if count else None,
        **usage,
        "mean_tokens": usage["total_tokens"] / count if count else None,
    }
    payload.update(
        {
            "tau2_commit": TAU2_COMMIT,
            "user_simulator_model": user_model,
            "nl_assertions_model": nl_assertions_model,
            "agent_model": agent_model,
            "pass_k": pass_k,
            "experiment_phase": phase,
            "action_constraint": False,
            "protocol": "tau3_retail_posttraining_v1",
            "annotated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_summary": summary,
        }
    )
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(result_path)
    return summary
