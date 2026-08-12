"""Windows orchestration for formal M1 teacher calibration and rollouts.

Requires:
1. NSCC teacher vLLM via tunnel (default http://127.0.0.1:8124/v1)
2. Frozen DeepSeek user/judge credentials in the local process environment
3. Local pinned tau2-bench at --tau-root
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env or os.environ.copy())
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "calibrate", "tau3-train", "compiled", "all"), required=True)
    parser.add_argument("--tau-root", type=Path, default=Path(r"E:\cv_codex\external\tau2-bench"))
    parser.add_argument("--teacher-model-id", default="Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--agent-base-url", default=os.environ.get("TAU3_AGENT_BASE_URL", "http://127.0.0.1:8124/v1"))
    parser.add_argument("--user-model", default=os.environ.get("TAU3_USER_MODEL", "deepseek/deepseek-chat"))
    parser.add_argument(
        "--nl-assertions-model",
        default=os.environ.get("TAU3_NL_ASSERTIONS_MODEL", "deepseek/deepseek-chat"),
    )
    parser.add_argument("--calibration-tasks", type=int, default=12)
    parser.add_argument("--tau3-pass-k", type=int, default=6)
    parser.add_argument("--compiled-pass-k", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["TAU3_AGENT_BASE_URL"] = args.agent_base_url
    env["TAU3_AGENT_API_KEY"] = env.get("TAU3_AGENT_API_KEY") or "local-vllm"
    env["ERAG_CONTEXT_COMPACTION"] = "0"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["NO_PROXY"] = ",".join(
        dict.fromkeys(
            [
                *(env.get("NO_PROXY") or env.get("no_proxy") or "").split(","),
                "127.0.0.1",
                "localhost",
            ]
        )
    ).strip(",")
    env["no_proxy"] = env["NO_PROXY"]

    agent_model = f"hosted_vllm/{args.teacher_model_id}"
    common = [
        sys.executable,
        "-m",
        "scripts.run_tau3_retail_v1",
        "--tau-root",
        str(args.tau_root),
        "--agent-name",
        "ecommerce_native",
        "--compaction",
        "off",
        "--agent-model",
        agent_model,
        "--user-model",
        args.user_model,
        "--nl-assertions-model",
        args.nl_assertions_model,
        "--max-steps",
        "200",
    ]

    if args.phase in {"check", "all"} or args.check_only:
        _run(
            [
                *common,
                "--phase",
                "teacher",
                "--pass-k",
                "1",
                "--save-to",
                "tau3_teacher_m1_check",
                "--check-only",
            ],
            env=env,
        )
        if args.phase == "check" or args.check_only:
            return

    if args.phase in {"calibrate", "all"}:
        # Smoke-sized train-only calibration against the frozen protocol.
        _run(
            [
                *common,
                "--phase",
                "smoke",
                "--pass-k",
                "1",
                "--save-to",
                f"tau3_teacher_m1_calibrate_{args.teacher_model_id}",
            ],
            env=env,
        )

    if args.phase in {"tau3-train", "all"}:
        _run(
            [
                *common,
                "--phase",
                "teacher",
                "--pass-k",
                str(args.tau3_pass_k),
                "--save-to",
                f"tau3_retail_teacher_m1_k{args.tau3_pass_k}",
            ],
            env=env,
        )

    if args.phase in {"compiled", "all"}:
        _run(
            [
                sys.executable,
                "-m",
                "scripts.run_compiled_retail_teacher",
                "--tau-root",
                str(args.tau_root),
                "--agent-model",
                agent_model,
                "--user-model",
                args.user_model,
                "--nl-assertions-model",
                args.nl_assertions_model,
                "--pass-k",
                str(args.compiled_pass_k),
                "--task-split-name",
                "base",
                "--save-to",
                f"compiled_retail_teacher_m1_k{args.compiled_pass_k}",
            ],
            env=env,
        )

    print(json.dumps({"ok": True, "phase": args.phase, "finished_at": time.time()}, indent=2))


if __name__ == "__main__":
    main()
