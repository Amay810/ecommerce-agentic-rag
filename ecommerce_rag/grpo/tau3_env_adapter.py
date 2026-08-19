"""Thin adapter around the pinned tau2 AgentGymEnv.

The tau2 source is intentionally external to the main repository.  NSCC gets
it from the archive snapshot and passes its path through ``TAU_ROOT``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FROZEN_CONFIG, TAU2_COMMIT, TAU2_VERSION


class Tau3EnvironmentError(RuntimeError):
    """The environment could not be created or advanced."""


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
            # AgentGymEnv catches orchestrator-thread exceptions and returns a
            # terminal zero. Preserve the distinction so this is not treated
            # as a valid negative training example.
            if result[2] and getattr(self._env, "_simulation_run", None) is None:
                observation, reward, terminated, truncated, info = result
                info = dict(info or {})
                info["interaction_error"] = (
                    "AgentGymEnv terminated without a SimulationRun; "
                    "tau2 likely failed inside its orchestrator thread"
                )
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
