"""Run the frozen tau3 Retail v1 smoke, Base, or SFT arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ecommerce_rag.tau3_retail_v1 import (
    annotate_results,
    build_tau2_command,
    validate_tau2_checkout,
)


def _first_env(*names: str) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def _configure_provider_environment(
    environment: dict[str, str],
    agent_model: str,
    user_model: str,
    nl_assertions_model: str,
) -> None:
    """Map project endpoint variables to provider variables without persisting keys."""
    shared_key = _first_env("ERAG_LLM_API_KEY", "ARAG_LLM_API_KEY")
    shared_base = _first_env("ERAG_LLM_BASE_URL", "ARAG_LLM_BASE_URL")
    agent_key = _first_env("TAU3_AGENT_API_KEY") or shared_key
    agent_base = _first_env("TAU3_AGENT_BASE_URL") or shared_base
    user_key = _first_env("TAU3_USER_API_KEY") or shared_key
    judge_key = _first_env("TAU3_JUDGE_API_KEY")

    if agent_model.startswith("openai/"):
        if not agent_key or not agent_base:
            raise ValueError(
                "openai/ agent requires TAU3_AGENT_API_KEY and TAU3_AGENT_BASE_URL "
                "(or ERAG_/ARAG_ fallbacks)"
            )
        environment["OPENAI_API_KEY"] = agent_key
        environment["OPENAI_API_BASE"] = agent_base
    elif agent_model.startswith("hosted_vllm/"):
        if not agent_base:
            raise ValueError("hosted_vllm/ agent requires TAU3_AGENT_BASE_URL")
        environment["HOSTED_VLLM_API_BASE"] = agent_base
        environment["HOSTED_VLLM_API_KEY"] = agent_key or "local-vllm"
        if agent_base.startswith(("http://127.0.0.1:", "http://localhost:")):
            no_proxy = environment.get("NO_PROXY") or environment.get("no_proxy", "")
            entries = [entry.strip() for entry in no_proxy.split(",") if entry.strip()]
            for host in ("127.0.0.1", "localhost"):
                if host not in entries:
                    entries.append(host)
            environment["NO_PROXY"] = ",".join(entries)
            environment["no_proxy"] = environment["NO_PROXY"]
    if user_model.startswith("deepseek/"):
        if not user_key:
            raise ValueError(
                "deepseek/ user requires TAU3_USER_API_KEY "
                "(or ERAG_/ARAG_ API key fallback)"
            )
        environment["DEEPSEEK_API_KEY"] = user_key
    elif user_model.startswith("openai/") and not environment.get("OPENAI_API_KEY"):
        if not user_key:
            raise ValueError("openai/ user requires TAU3_USER_API_KEY")
        environment["OPENAI_API_KEY"] = user_key
    if nl_assertions_model.startswith("deepseek/"):
        if not user_key:
            raise ValueError("deepseek/ judge requires TAU3_USER_API_KEY")
        environment["DEEPSEEK_API_KEY"] = user_key
    elif nl_assertions_model.startswith("openai/"):
        if not judge_key:
            raise ValueError("openai/ judge requires TAU3_JUDGE_API_KEY")
        environment["OPENAI_API_KEY"] = judge_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau-root",
        type=Path,
        default=Path(r"E:\cv_codex\external\tau2-bench"),
    )
    parser.add_argument(
        "--phase", choices=("smoke", "teacher", "base", "sft"), required=True
    )
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--nl-assertions-model", required=True)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--agent-name",
        choices=("llm_agent", "ecommerce_native"),
        default="ecommerce_native",
    )
    parser.add_argument("--compaction", choices=("off", "on"), default="off")
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Override tau2 worker count. Use 1 when remaining long-horizon tasks stall under 3-way contention.",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    splits = validate_tau2_checkout(args.tau_root)
    tau_python_candidates = (
        args.tau_root / ".venv" / "Scripts" / "python.exe",
        args.tau_root / ".venv" / "bin" / "python",
    )
    tau_python = next((path for path in tau_python_candidates if path.exists()), None)
    if tau_python is None:
        raise FileNotFoundError(
            "tau2 python not found in .venv/Scripts/python.exe or .venv/bin/python"
        )
    launcher_script = Path(__file__).with_name("_tau3_cli_with_frozen_judge.py")
    command = build_tau2_command(
        tau_python=tau_python,
        launcher_script=launcher_script,
        phase=args.phase,
        agent_model=args.agent_model,
        user_model=args.user_model,
        pass_k=args.pass_k,
        save_to=args.save_to,
        max_steps=args.max_steps,
        agent_name=args.agent_name,
        task_ids=args.task_ids,
        max_concurrency=args.max_concurrency,
    )
    public_config = {
        "protocol": "tau3_retail_posttraining_v1",
        "phase": args.phase,
        "agent_model": args.agent_model,
        "user_simulator_model": args.user_model,
        "nl_assertions_model": args.nl_assertions_model,
        "pass_k": args.pass_k,
        "max_steps": args.max_steps,
        "agent_name": args.agent_name,
        "compaction": args.compaction,
        "save_to": args.save_to,
        "splits": splits,
        "command": command,
        "task_ids": args.task_ids,
    }
    if args.check_only:
        print(json.dumps(public_config, ensure_ascii=False, indent=2))
        return

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["TAU3_NL_ASSERTIONS_MODEL"] = args.nl_assertions_model
    environment["ERAG_CONTEXT_COMPACTION"] = "1" if args.compaction == "on" else "0"
    project_root = str(Path(__file__).resolve().parents[1])
    environment["ERAG_PROJECT_ROOT"] = project_root
    python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (project_root, python_path) if entry
    )
    _configure_provider_environment(
        environment,
        args.agent_model,
        args.user_model,
        args.nl_assertions_model,
    )
    started = time.perf_counter()
    subprocess.run(command, cwd=args.tau_root, env=environment, check=True)
    elapsed = time.perf_counter() - started
    result_path = args.tau_root / "data" / "simulations" / args.save_to / "results.json"
    summary = annotate_results(
        result_path,
        phase=args.phase,
        agent_model=args.agent_model,
        user_model=args.user_model,
        nl_assertions_model=args.nl_assertions_model,
        pass_k=args.pass_k,
        wall_clock_seconds=elapsed,
        expected_task_count=len(args.task_ids) if args.task_ids else None,
    )
    if not summary["valid"]:
        raise RuntimeError(
            f"invalid run: {summary['infrastructure_errors']} infrastructure errors"
        )
    print(
        json.dumps({**public_config, "summary": summary}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
