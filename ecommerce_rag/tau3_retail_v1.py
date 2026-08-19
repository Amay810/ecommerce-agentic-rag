"""Frozen helpers for the tau3 Retail post-training v1 experiment.

Experiment-control discipline
-----------------------------
Command-line arguments are a *request*. They are not evidence that tau2 ran with
that configuration. This module therefore keeps four things apart and never lets
one stand in for another:

``requested``
    What the operator asked for. Every experiment-control variable is explicit;
    this module defines no defaults for them.
``command``
    The argv actually handed to tau2, derived from ``requested`` and verified
    against it by :func:`verify_command_matches_requested`.
``native_observed``
    What tau2 itself wrote into ``results.json`` under ``info``. Never
    overwritten, never synthesised.
``derived``
    Statistics computed from the simulations after the strict comparison passed.

:func:`annotate_results` compares ``requested`` against ``native_observed`` and
fails closed -- including when a native field is *absent* -- before it writes any
annotation. Absence used to pass silently; that made the annotation self-proving.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
EXPECTED_SPLITS = {"train": 74, "test": 40, "base": 114}
PROTOCOL = "tau3_retail_posttraining_v1"
SUPPORTED_PHASES = ("smoke", "teacher", "base", "sft")
SUPPORTED_AGENTS = ("llm_agent", "ecommerce_native")
SUPPORTED_COMPACTION = ("off", "on")
MAX_TEMPERATURE = 2.0


def resolve_tau2_data_dir(tau_root: Path, env: Mapping[str, str]) -> Path:
    """Return tau2's DATA_DIR for a run.

    tau2 reads simulations from ``TAU2_DATA_DIR`` when set; otherwise from the
    installed package source tree, which may differ from ``--tau-root``. Callers
    must set ``TAU2_DATA_DIR`` in the subprocess environment to this value.
    """
    override = env.get("TAU2_DATA_DIR")
    if override:
        return Path(override)
    return tau_root / "data"


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


def verify_tau2_source(tau_root: Path) -> dict[str, str]:
    """Establish tau2 identity before the run and fail closed on drift.

    A vendored snapshot deliberately excludes ``.git``. Running ``git rev-parse``
    inside one does not fail -- it walks *up* and reports the enclosing
    repository's HEAD, which has nothing to do with tau2. So:

    * ``SNAPSHOT.json`` present -> its ``upstream_commit`` is the authority and
      git is never consulted.
    * no ``SNAPSHOT.json`` -> git is consulted only when ``tau_root`` is itself a
      checkout, i.e. it directly contains ``.git`` (a directory, or a gitfile for
      worktrees and submodules).
    * neither -> fail closed rather than inherit a parent repository's HEAD.
    """
    snapshot_path = tau_root / "SNAPSHOT.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        commit = snapshot.get("upstream_commit")
        if not commit:
            raise ValueError(f"SNAPSHOT.json is missing upstream_commit: {snapshot_path}")
        if commit != TAU2_COMMIT:
            raise ValueError(f"tau2 snapshot commit drift: {commit} != {TAU2_COMMIT}")
        return {
            "commit": commit,
            "verified_by": "snapshot",
            "source": str(snapshot_path),
        }
    if not (tau_root / ".git").exists():
        raise ValueError(
            f"cannot establish tau2 identity for {tau_root}: no SNAPSHOT.json and "
            "tau_root is not itself a git checkout. Refusing to run git rev-parse, "
            "which would report an enclosing repository's HEAD instead of tau2's."
        )
    commit = git_head(tau_root)
    if commit != TAU2_COMMIT:
        raise ValueError(f"tau2 commit drift: {commit} != {TAU2_COMMIT}")
    return {"commit": commit, "verified_by": "git", "source": str(tau_root)}


def validate_tau2_checkout(tau_root: Path) -> dict[str, int]:
    """Fail closed when the benchmark commit or Retail splits drift."""
    verify_tau2_source(tau_root)

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


def _validate_temperature(name: str, value: float) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= MAX_TEMPERATURE:
        raise ValueError(f"{name} out of range: {numeric}")
    return numeric


def requested_config(
    *,
    phase: str,
    agent_name: str,
    agent_model: str,
    user_model: str,
    nl_assertions_model: str,
    agent_temperature: float,
    user_temperature: float,
    seed: int,
    pass_k: int,
    compaction: str,
    save_to: str,
    max_steps: int = 200,
    task_ids: Iterable[str] | None = None,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    """Return the single source of truth for one run's requested configuration.

    ``compaction`` is a project-side runtime variable rather than a tau2 CLI
    flag, so it cannot appear in the command or in tau2's native info. It is
    recorded here so the manifest still carries its provenance.
    """
    if phase not in SUPPORTED_PHASES:
        raise ValueError(f"unsupported phase: {phase}")
    if agent_name not in SUPPORTED_AGENTS:
        raise ValueError(f"unsupported agent: {agent_name}")
    if compaction not in SUPPORTED_COMPACTION:
        raise ValueError(f"unsupported compaction: {compaction}")
    if pass_k < 1:
        raise ValueError("pass_k must be positive")
    if max_steps < 20:
        raise ValueError("max_steps below 20 can truncate Retail conversations")
    if phase == "smoke" and pass_k != 1:
        raise ValueError("smoke is cost calibration only and must use pass_k=1")
    if phase == "smoke" and not list(task_ids or []):
        raise ValueError(
            "smoke must name its tasks with explicit task_ids; "
            "there is no default task count"
        )
    if int(seed) != seed:
        raise ValueError(f"seed must be an integer: {seed!r}")

    selected_task_ids = [str(value) for value in (task_ids or [])]
    concurrency = 1 if phase == "smoke" else 3
    if max_concurrency is not None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        concurrency = int(max_concurrency)

    return {
        "protocol": PROTOCOL,
        "phase": phase,
        "task_split_name": "train" if phase in {"smoke", "teacher"} else "test",
        "agent_name": agent_name,
        "agent_model": agent_model,
        "user_model": user_model,
        "nl_assertions_model": nl_assertions_model,
        "agent_temperature": _validate_temperature(
            "agent_temperature", agent_temperature
        ),
        "user_temperature": _validate_temperature("user_temperature", user_temperature),
        "seed": int(seed),
        "pass_k": int(pass_k),
        "max_steps": int(max_steps),
        "max_concurrency": concurrency,
        "compaction": compaction,
        "save_to": save_to,
        "task_ids": selected_task_ids or None,
        "action_constraint": False,
    }


def build_tau2_command(
    *,
    tau_python: Path,
    launcher_script: Path,
    phase: str,
    agent_name: str,
    agent_model: str,
    user_model: str,
    agent_temperature: float,
    user_temperature: float,
    seed: int,
    pass_k: int,
    save_to: str,
    max_steps: int = 200,
    task_ids: Iterable[str] | None = None,
    max_concurrency: int | None = None,
) -> list[str]:
    """Build the only command shape allowed by the frozen v1 protocol.

    No experiment-control variable has a default here. tau2's own defaults for
    ``--agent-llm-args``, ``--user-llm-args`` and ``--seed`` are never relied on:
    all three are always emitted explicitly.
    """
    requested = requested_config(
        phase=phase,
        agent_name=agent_name,
        agent_model=agent_model,
        user_model=user_model,
        nl_assertions_model="",
        agent_temperature=agent_temperature,
        user_temperature=user_temperature,
        seed=seed,
        pass_k=pass_k,
        max_steps=max_steps,
        compaction="off",
        save_to=save_to,
        task_ids=task_ids,
        max_concurrency=max_concurrency,
    )
    return command_from_requested(
        tau_python=tau_python, launcher_script=launcher_script, requested=requested
    )


def command_from_requested(
    *,
    tau_python: Path,
    launcher_script: Path,
    requested: Mapping[str, Any],
) -> list[str]:
    """Derive the tau2 argv from a requested configuration.

    ``--agent-llm-args`` / ``--user-llm-args`` are JSON documents in the pinned
    tau2 CLI (``type=json.loads``), so they are emitted as compact JSON strings.
    """
    command = [
        str(tau_python),
        str(launcher_script),
        "run",
        "--domain",
        "retail",
        "--task-set-name",
        "retail",
        "--task-split-name",
        requested["task_split_name"],
        "--num-trials",
        str(requested["pass_k"]),
        "--agent",
        requested["agent_name"],
        "--agent-llm",
        requested["agent_model"],
        "--agent-llm-args",
        json.dumps({"temperature": requested["agent_temperature"]}),
        "--user",
        "user_simulator",
        "--user-llm",
        requested["user_model"],
        "--user-llm-args",
        json.dumps({"temperature": requested["user_temperature"]}),
        "--seed",
        str(requested["seed"]),
        "--max-steps",
        str(requested["max_steps"]),
        "--save-to",
        requested["save_to"],
        "--auto-resume",
        "--verbose-logs",
        "--max-concurrency",
        str(requested["max_concurrency"]),
    ]
    task_ids = requested.get("task_ids")
    if task_ids:
        command.extend(["--task-ids", *(str(value) for value in task_ids)])
    return command


def _command_value(command: list[str], flag: str) -> str:
    if flag not in command:
        raise ValueError(f"command is missing {flag}")
    return command[command.index(flag) + 1]


def verify_command_matches_requested(
    command: list[str], requested: Mapping[str, Any]
) -> None:
    """Parse the argv back out and fail closed if it drifts from ``requested``.

    This closes the first link of the chain by reading the command rather than
    trusting that it was built correctly.
    """
    if _command_value(command, "--agent") != requested["agent_name"]:
        raise ValueError("command agent does not match requested agent_name")
    if _command_value(command, "--agent-llm") != requested["agent_model"]:
        raise ValueError("command agent-llm does not match requested agent_model")
    if _command_value(command, "--user-llm") != requested["user_model"]:
        raise ValueError("command user-llm does not match requested user_model")
    if int(_command_value(command, "--seed")) != requested["seed"]:
        raise ValueError("command seed does not match requested seed")
    if int(_command_value(command, "--num-trials")) != requested["pass_k"]:
        raise ValueError("command num-trials does not match requested pass_k")
    if int(_command_value(command, "--max-steps")) != requested["max_steps"]:
        raise ValueError("command max-steps does not match requested max_steps")
    if _command_value(command, "--task-split-name") != requested["task_split_name"]:
        raise ValueError("command task-split-name does not match requested split")
    agent_args = json.loads(_command_value(command, "--agent-llm-args"))
    if float(agent_args.get("temperature")) != requested["agent_temperature"]:
        raise ValueError("command agent temperature does not match requested")
    user_args = json.loads(_command_value(command, "--user-llm-args"))
    if float(user_args.get("temperature")) != requested["user_temperature"]:
        raise ValueError("command user temperature does not match requested")


def _message_usage(simulations: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    for simulation in simulations:
        for message in simulation.get("messages") or []:
            usage = message.get("usage") or {}
            for name in totals:
                totals[name] += int(usage.get(name) or 0)
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def _require(container: Any, key: str, label: str) -> Any:
    """Return ``container[key]`` or fail closed.

    Absence is a failure, not a pass. The previous implementation guarded its
    comparisons with ``if recorded and recorded != expected``, so a missing
    native field silently satisfied the check.
    """
    if not isinstance(container, Mapping):
        raise ValueError(f"native {label} is missing from results.json")
    if key not in container or container[key] is None:
        raise ValueError(f"native {label} is missing from results.json")
    return container[key]


def observe_native_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read the configuration tau2 itself recorded, failing closed on absence.

    ``info.git_commit`` is recorded but deliberately *not* asserted against
    :data:`TAU2_COMMIT`. tau2 fills it by running ``git rev-parse HEAD`` in the
    runner's working directory, so for a vendored snapshot it reports the
    enclosing archive repository -- not tau2. tau2 identity is established before
    the run by :func:`verify_tau2_source`.
    """
    info = _require(payload, "info", "info")
    agent_info = _require(info, "agent_info", "info.agent_info")
    user_info = _require(info, "user_info", "info.user_info")
    agent_llm_args = _require(agent_info, "llm_args", "info.agent_info.llm_args")
    user_llm_args = _require(user_info, "llm_args", "info.user_info.llm_args")
    return {
        "git_commit": info.get("git_commit"),
        "num_trials": _require(info, "num_trials", "info.num_trials"),
        "max_steps": _require(info, "max_steps", "info.max_steps"),
        "seed": _require(info, "seed", "info.seed"),
        "agent_implementation": _require(
            agent_info, "implementation", "info.agent_info.implementation"
        ),
        "agent_llm": _require(agent_info, "llm", "info.agent_info.llm"),
        "agent_llm_args": dict(agent_llm_args),
        "agent_temperature": float(
            _require(
                agent_llm_args, "temperature", "info.agent_info.llm_args.temperature"
            )
        ),
        "user_implementation": _require(
            user_info, "implementation", "info.user_info.implementation"
        ),
        "user_llm": _require(user_info, "llm", "info.user_info.llm"),
        "user_llm_args": dict(user_llm_args),
        "user_temperature": float(
            _require(user_llm_args, "temperature", "info.user_info.llm_args.temperature")
        ),
    }


def compare_native_to_requested(
    native: Mapping[str, Any], requested: Mapping[str, Any]
) -> list[str]:
    """Fail closed on any drift and return the names of the checks that passed."""
    comparisons = (
        ("agent_name", native["agent_implementation"], requested["agent_name"]),
        ("agent_model", native["agent_llm"], requested["agent_model"]),
        ("user_model", native["user_llm"], requested["user_model"]),
        (
            "agent_temperature",
            native["agent_temperature"],
            requested["agent_temperature"],
        ),
        ("user_temperature", native["user_temperature"], requested["user_temperature"]),
        ("seed", int(native["seed"]), requested["seed"]),
        ("num_trials", int(native["num_trials"]), requested["pass_k"]),
        ("max_steps", int(native["max_steps"]), requested["max_steps"]),
    )
    for name, observed, expected in comparisons:
        if observed != expected:
            raise ValueError(
                f"native {name} does not match requested: {observed!r} != {expected!r}"
            )
    return [name for name, _, _ in comparisons]


def expected_simulation_count(requested: Mapping[str, Any]) -> int:
    """Return how many simulations a complete run must contain.

    There is no phase-derived task-count guess: either the run named its tasks
    explicitly, or it ran a whole frozen split whose size is already verified by
    :func:`validate_tau2_checkout`.
    """
    task_ids = requested.get("task_ids")
    if task_ids:
        task_count = len(task_ids)
    else:
        task_count = EXPECTED_SPLITS[requested["task_split_name"]]
    return task_count * requested["pass_k"]


def annotate_results(
    result_path: Path,
    *,
    requested: Mapping[str, Any],
    command: list[str],
    tau2_source: Mapping[str, str],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Verify the run against ``requested``, then embed provenance and statistics.

    The order matters and is the point of this function:
    requested -> command -> native observed -> strict comparison -> annotation.
    Nothing is written to ``result_path`` until every comparison has passed, and
    the requested configuration is never used as evidence about the run.
    """
    verify_command_matches_requested(command, requested)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    native = observe_native_config(payload)
    checks_passed = compare_native_to_requested(native, requested)

    simulations = payload.get("simulations") or []
    expected_count = expected_simulation_count(requested)
    if len(simulations) != expected_count:
        raise ValueError(
            f"incomplete {requested['phase']} result: "
            f"{len(simulations)} != {expected_count}"
        )

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

    # payload["info"] is tau2's own record and is left exactly as written.
    payload["tau3_experiment"] = {
        "protocol": PROTOCOL,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "tau2_source": dict(tau2_source),
        "requested": dict(requested),
        "command": list(command),
        "native_observed": native,
        "checks_passed": checks_passed,
    }
    payload["experiment_summary"] = summary
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(result_path)
    return summary
