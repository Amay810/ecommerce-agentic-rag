"""Run the minimal Base-versus-teacher pilot on compiled structures and hard tau3 tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _zero_success_task_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rewards: dict[str, list[float]] = defaultdict(list)
    for simulation in payload.get("simulations") or []:
        task_id = str(simulation.get("task_id") or "")
        reward = float((simulation.get("reward_info") or {}).get("reward") or 0.0)
        if task_id:
            rewards[task_id].append(reward)
    return sorted(task_id for task_id, values in rewards.items() if values and max(values) == 0.0)


def _run(command: list[str], environment: dict[str, str], check_only: bool) -> None:
    if check_only:
        command.append("--check-only")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-base-results", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, default=Path(r"E:\cv_codex\external\tau2-bench"))
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--user-model", default="deepseek/deepseek-chat")
    parser.add_argument("--nl-assertions-model", default="deepseek/deepseek-chat")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    hard_task_ids = _zero_success_task_ids(args.historical_base_results)
    if not hard_task_ids:
        raise SystemExit("historical baseline contains no zero-success tasks")

    for arm, model, base_url in (
        ("base", args.base_model, args.base_url),
        ("teacher", args.teacher_model, args.teacher_url),
    ):
        environment = os.environ.copy()
        environment["TAU3_AGENT_BASE_URL"] = base_url
        environment["TAU3_AGENT_API_KEY"] = environment.get("TAU3_AGENT_API_KEY") or "local-vllm"
        tau_command = [
            sys.executable,
            "-m",
            "scripts.run_tau3_retail_v1",
            "--tau-root",
            str(args.tau_root),
            "--phase",
            "teacher",
            "--agent-model",
            model,
            "--user-model",
            args.user_model,
            "--nl-assertions-model",
            args.nl_assertions_model,
            "--pass-k",
            "1",
            "--agent-name",
            "ecommerce_native",
            "--compaction",
            "off",
            "--save-to",
            f"m1_pilot_tau3_hard_{arm}",
            "--task-ids",
            *hard_task_ids,
        ]
        compiled_command = [
            sys.executable,
            "-m",
            "scripts.run_compiled_retail_teacher",
            "--tau-root",
            str(args.tau_root),
            "--agent-model",
            model,
            "--user-model",
            args.user_model,
            "--nl-assertions-model",
            args.nl_assertions_model,
            "--pass-k",
            "1",
            "--task-split-name",
            "pilot",
            "--save-to",
            f"m1_pilot_compiled_{arm}",
        ]
        _run(tau_command, environment, args.check_only)
        _run(compiled_command, environment, args.check_only)

    print(json.dumps({"compiled_structures": 47, "tau3_zero_success_tasks": len(hard_task_ids)}, indent=2))


if __name__ == "__main__":
    main()
