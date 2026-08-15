"""Run teacher rollouts on compiled_retail tasks using the official retail env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ecommerce_rag.tau3_retail_v1 import TAU2_COMMIT, git_head
from scripts.run_tau3_retail_v1 import _configure_provider_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-root", type=Path, default=Path(r"E:\cv_codex\external\tau2-bench"))
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/compiled_retail_m1/tasks.json"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/compiled_retail_m1/split_tasks.json"),
    )
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--nl-assertions-model", required=True)
    parser.add_argument("--pass-k", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--task-split-name", default="train")
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="tau2 worker count. Keep 1 unless short tasks and 4B DP replicas are both available.",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    actual = git_head(args.tau_root)
    if actual != TAU2_COMMIT:
        raise SystemExit(f"tau2 commit drift: {actual} != {TAU2_COMMIT}")

    tau_python = next(
        (
            path
            for path in (
                args.tau_root / ".venv" / "Scripts" / "python.exe",
                args.tau_root / ".venv" / "bin" / "python",
            )
            if path.exists()
        ),
        None,
    )
    if tau_python is None:
        raise SystemExit("tau2 python not found")

    launcher = Path(__file__).with_name("_tau3_cli_with_frozen_judge.py")
    command = [
        str(tau_python),
        str(launcher),
        "run",
        "--domain",
        "retail",
        "--task-set-name",
        "compiled_retail",
        "--task-split-name",
        args.task_split_name,
        "--num-trials",
        str(args.pass_k),
        "--agent",
        "ecommerce_native",
        "--agent-llm",
        args.agent_model,
        "--user",
        "user_simulator",
        "--user-llm",
        args.user_model,
        "--max-steps",
        str(args.max_steps),
        "--save-to",
        args.save_to,
        "--auto-resume",
        "--verbose-logs",
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    if args.num_tasks is not None:
        command.extend(["--num-tasks", str(args.num_tasks)])

    public = {
        "protocol": "compiled_retail_teacher_v1",
        "source": "compiled_retail",
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "nl_assertions_model": args.nl_assertions_model,
        "pass_k": args.pass_k,
        "task_split_name": args.task_split_name,
        "save_to": args.save_to,
        "command": command,
    }
    if args.check_only:
        print(json.dumps(public, ensure_ascii=False, indent=2))
        return

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["TAU3_NL_ASSERTIONS_MODEL"] = args.nl_assertions_model
    environment["ERAG_CONTEXT_COMPACTION"] = "0"
    project_root = str(Path(__file__).resolve().parents[1])
    environment["ERAG_PROJECT_ROOT"] = project_root
    python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        project_root if not python_path else project_root + os.pathsep + python_path
    )
    _configure_provider_environment(
        environment,
        args.agent_model,
        args.user_model,
        args.nl_assertions_model,
    )

    # Ensure registration import runs inside the launcher process too.
    environment["ERAG_COMPILED_RETAIL_TASKS"] = str(args.tasks.resolve())
    environment["ERAG_COMPILED_RETAIL_SPLITS"] = str(args.splits.resolve())

    started = time.time()
    completed = subprocess.run(command, cwd=project_root, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    result_path = args.tau_root / "data" / "simulations" / args.save_to / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["compiled_retail_meta"] = {
        **public,
        "wall_clock_seconds": time.time() - started,
        "source_label": "compiled_retail",
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "result": str(result_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
