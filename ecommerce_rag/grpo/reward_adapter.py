"""Adapter for the pinned tau2 official terminal reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class RewardAdapterError(RuntimeError):
    """The episode did not produce a valid official terminal reward."""


@dataclass(frozen=True)
class TerminalReward:
    value: float
    source: str = "tau2.evaluator.evaluate_simulation"
    evaluation_type: str = "ALL"


class OfficialTerminalRewardAdapter:
    """Keep only tau2's final binary reward; never add dense proxies."""

    source = "tau2.evaluator.evaluate_simulation"
    evaluation_type = "ALL"

    def from_step(self, *, terminated: bool, reward: Any, info: dict[str, Any]) -> TerminalReward:
        if not terminated:
            raise RewardAdapterError("non-terminal step cannot produce a training reward")
        if info.get("interaction_error"):
            raise RewardAdapterError(
                "tau2 interaction failed; refusing to convert infrastructure failure into reward 0"
            )
        if info.get("evaluator_error"):
            raise RewardAdapterError(
                "tau2 official evaluator failed; refusing to convert the failure into reward 0"
            )
        if not info.get("tau2_simulation_run_complete"):
            raise RewardAdapterError(
                "terminal tau2 step has no complete SimulationRun; refusing to convert it into reward 0"
            )
        if not info.get("tau2_official_evaluator_succeeded"):
            raise RewardAdapterError(
                "terminal tau2 step has no successful official RewardInfo; refusing to convert it into reward 0"
            )
        try:
            value = float(reward)
        except (TypeError, ValueError) as exc:
            raise RewardAdapterError(f"invalid tau2 reward: {reward!r}") from exc
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise RewardAdapterError(f"official tau2 terminal reward must be 0/1, got {reward!r}")
        return TerminalReward(value=value)

    def from_episode(self, result: Any) -> TerminalReward:
        """Map a result while classifying project/VERL truncation separately.

        The pinned tau2 AgentGymEnv always returns ``truncated=False``.  Its
        MAX_STEPS and other policy-side termination reasons are represented by
        a complete SimulationRun and an official evaluator reward of 0, so
        they are accepted by ``from_step``.
        """
        if not isinstance(result, tuple) or len(result) != 5:
            raise RewardAdapterError("expected AgentGymEnv.step() five-tuple")
        _, reward, terminated, truncated, info = result
        if truncated:
            raise RewardAdapterError(
                "project/VERL rollout truncation has no official tau2 terminal reward"
            )
        return self.from_step(
            terminated=bool(terminated), reward=reward, info=dict(info or {})
        )
