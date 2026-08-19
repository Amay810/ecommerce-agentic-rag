"""Thin adapter around the pinned tau2 AgentGymEnv.

The tau2 source is intentionally external to the main repository.  NSCC gets
it from the archive snapshot and passes its path through ``TAU_ROOT``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FROZEN_CONFIG, TAU2_COMMIT, TAU2_VERSION


class Tau3EnvironmentError(RuntimeError):
    """The environment could not be created or advanced."""


_REQUIRED_SIMULATION_RUN_FIELDS = (
    "id",
    "task_id",
    "start_time",
    "end_time",
    "duration",
    "termination_reason",
)


def _complete_simulation_run(simulation_run: Any) -> bool:
    """Check the minimum pinned tau2 fields that make a run evaluable."""
    if simulation_run is None:
        return False
    model_dump = getattr(simulation_run, "model_dump", None)
    if not callable(model_dump):
        return False
    try:
        payload = model_dump()
    except Exception:
        return False
    return all(payload.get(field) is not None for field in _REQUIRED_SIMULATION_RUN_FIELDS)


def _decode_reward_info(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return dict(raw) if isinstance(raw, dict) else None


def _snapshot_root(tau_root: str | Path) -> Path:
    root = Path(tau_root).expanduser().resolve()
    if (root / "src" / "tau2").is_dir():
        return root
    nested = root / "vendor" / "tau2-bench-fc0055dc"
    if (nested / "src" / "tau2").is_dir():
        return nested
    raise Tau3EnvironmentError(f"tau2 snapshot source not found under {root}")


def validate_snapshot(tau_root: str | Path) -> Path:
    root = _snapshot_root(tau_root)
    snapshot = root / "SNAPSHOT.json"
    if snapshot.exists():
        metadata = json.loads(snapshot.read_text(encoding="utf-8"))
        if metadata.get("upstream_commit") != TAU2_COMMIT:
            raise Tau3EnvironmentError(
                "tau2 snapshot commit mismatch: "
                f"{metadata.get('upstream_commit')!r} != {TAU2_COMMIT!r}"
            )
        if metadata.get("upstream_tag") != f"v{TAU2_VERSION}":
            raise Tau3EnvironmentError("tau2 snapshot version mismatch")
    else:
        try:
            observed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Tau3EnvironmentError(
                "tau2 root has no SNAPSHOT.json and its git commit cannot be verified"
            ) from exc
        if observed != TAU2_COMMIT:
            raise Tau3EnvironmentError(
                f"tau2 checkout commit mismatch: {observed!r} != {TAU2_COMMIT!r}"
            )
    return root


def retail_train_task_ids(tau_root: str | Path) -> list[str]:
    root = validate_snapshot(tau_root)
    split_file = root / "data" / "tau2" / "domains" / "retail" / "split_tasks.json"
    try:
        split = json.loads(split_file.read_text(encoding="utf-8"))
        task_ids = [str(task_id) for task_id in split[FROZEN_CONFIG.train_split]]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise Tau3EnvironmentError(f"cannot load retail train split: {split_file}") from exc
    if len(task_ids) != FROZEN_CONFIG.train_tasks:
        raise Tau3EnvironmentError(
            f"expected 74 retail train tasks, found {len(task_ids)}"
        )
    return task_ids


def _import_tau2(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _configure_frozen_nl_judge() -> None:
    """Use the same NL-assertion judge selection as the frozen Track-A CLI.

    The pinned evaluator imports this setting into
    ``tau2.evaluator.evaluator_nl_assertions`` at module load time.  Track-A
    explicitly overwrites that module global before launching the CLI; the
    AgentGymEnv path must do the same before its first terminal evaluation.
    """
    from tau2.evaluator import evaluator_nl_assertions

    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = (
        FROZEN_CONFIG.nl_assertions_model
    )


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "openai_schema", None)
    if schema is not None:
        return dict(schema)
    if isinstance(tool, dict):
        return dict(tool)
    raise Tau3EnvironmentError(f"tau2 tool has no OpenAI schema: {tool!r}")


@dataclass(frozen=True)
class Tau3Reset:
    observation: str
    system_prompt: str
    tools: tuple[dict[str, Any], ...]


class Tau3RetailEpisode:
    """One fresh retail episode; never reuse an environment across K samples."""

    def __init__(self, tau_root: str | Path, task_id: str, *, user_base_url: str):
        self.root = validate_snapshot(tau_root)
        _import_tau2(self.root)
        self.task_id = str(task_id)
        self.user_base_url = user_base_url
        self._env: Any | None = None
        self._closed = False

    def reset(self) -> Tau3Reset:
        if self._closed:
            raise Tau3EnvironmentError("episode already closed")
        try:
            from tau2.gym.gym_agent import AgentGymEnv

            _configure_frozen_nl_judge()
            self._env = AgentGymEnv(
                domain=FROZEN_CONFIG.domain,
                task_id=self.task_id,
                max_steps=FROZEN_CONFIG.max_steps,
                solo_mode=False,
                user_llm=FROZEN_CONFIG.user_model,
                user_llm_args={
                    "api_base": self.user_base_url,
                    "api_key": os.environ.get("DEEPSEEK_API_KEY", "EMPTY"),
                    "temperature": FROZEN_CONFIG.user_temperature,
                    "seed": FROZEN_CONFIG.seed,
                },
            )
            observation, info = self._env.reset(seed=FROZEN_CONFIG.seed)
            system_prompt = str(info.get("policy") or "")
            tools = tuple(_tool_schema(tool) for tool in (info.get("tools") or []))
            if not system_prompt or not observation:
                raise Tau3EnvironmentError("tau2 reset returned an empty policy/observation")
            return Tau3Reset(
                observation=str(observation), system_prompt=system_prompt, tools=tools
            )
        except Tau3EnvironmentError:
            raise
        except Exception as exc:
            raise Tau3EnvironmentError(
                f"tau2 retail reset failed for task {self.task_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def step(self, assistant_action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise Tau3EnvironmentError("reset() must be called before step()")
        try:
            result = self._env.step(assistant_action)
            # AgentGymEnv._get_reward() returns a raw zero both for a valid
            # evaluator result and when no SimulationRun exists.  Carry the
            # two pieces of terminal evidence explicitly so the reward
            # adapter cannot mistake infrastructure failure for a negative.
            if result[2]:
                observation, reward, terminated, truncated, info = result
                info = dict(info or {})
                simulation_run = getattr(self._env, "_simulation_run", None)
                complete = _complete_simulation_run(simulation_run)
                info["tau2_simulation_run_present"] = simulation_run is not None
                info["tau2_simulation_run_complete"] = complete
                info["tau2_official_evaluator_succeeded"] = False
                if not complete:
                    info["interaction_error"] = (
                        "AgentGymEnv terminated without a complete SimulationRun; "
                        "tau2 likely failed inside its orchestrator thread"
                    )
                else:
                    reward_info = _decode_reward_info(info.get("reward_info"))
                    try:
                        evaluated_reward = float(reward_info["reward"] if reward_info else "nan")
                        observed_reward = float(reward)
                    except (KeyError, TypeError, ValueError):
                        evaluated_reward = observed_reward = float("nan")
                    if (
                        reward_info is None
                        or not math.isfinite(evaluated_reward)
                        or not math.isfinite(observed_reward)
                        or evaluated_reward != observed_reward
                    ):
                        info["evaluator_error"] = (
                            "AgentGymEnv did not return a matching official RewardInfo"
                        )
                    else:
                        info["tau2_official_evaluator_succeeded"] = True
                result = (observation, reward, terminated, truncated, info)
            return result
        except Exception as exc:
            raise Tau3EnvironmentError(
                f"tau2 retail step failed for task {self.task_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
        self._closed = True
