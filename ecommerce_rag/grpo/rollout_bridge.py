"""Multi-turn bridge between VERL-generated assistant turns and tau2."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .config import FROZEN_CONFIG
from .reward_adapter import OfficialTerminalRewardAdapter, TerminalReward
from .tau3_env_adapter import Tau3RetailEpisode, Tau3Reset


@dataclass(frozen=True)
class Tau3Step:
    observation: str
    reward: float | None
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class Tau3RolloutSession:
    tau_root: str
    task_id: str
    user_base_url: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    episode: Tau3RetailEpisode = field(init=False)
    reset_result: Tau3Reset | None = field(default=None, init=False)
    step_count: int = field(default=0, init=False)
    rollout_seconds: float = field(default=0.0, init=False)
    terminal_reward: TerminalReward | None = field(default=None, init=False)
    started: bool = field(default=False, init=False)
    reward_adapter: OfficialTerminalRewardAdapter = field(
        default_factory=OfficialTerminalRewardAdapter, init=False
    )

    def __post_init__(self) -> None:
        self.episode = Tau3RetailEpisode(
            self.tau_root, self.task_id, user_base_url=self.user_base_url
        )

    def start(self) -> Tau3Reset:
        self.reset_result = self.episode.reset()
        self.started = True
        return self.reset_result

    @staticmethod
    def assistant_action(*, content: str | None = None, tool_name: str | None = None, arguments: Any = None) -> str:
        if tool_name:
            return json.dumps(
                {"name": tool_name, "arguments": arguments if arguments is not None else {}},
                ensure_ascii=False,
            )
        if not content:
            raise ValueError("assistant turn must contain content or a tool call")
        return content

    def submit(self, assistant_action: str) -> Tau3Step:
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = self.episode.step(assistant_action)
        self.rollout_seconds += time.perf_counter() - started
        self.step_count += 1
        result = Tau3Step(
            observation=str(observation),
            reward=float(reward) if terminated else None,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )
        if result.terminated:
            self.terminal_reward = self.reward_adapter.from_step(
                terminated=True, reward=result.reward, info=result.info
            )
        return result

    def close(self) -> None:
        try:
            if self.started and self.terminal_reward is None:
                raise RuntimeError(
                    "tau3 rollout ended without an official terminal reward; "
                    "do not convert an incomplete/failed episode to reward 0"
                )
        finally:
            self.episode.close()


def render_tools_for_prompt(tools: tuple[dict[str, Any], ...]) -> str:
    """Render the tau2-provided schemas without creating a second tool system."""
    if not tools:
        return ""
    return "\n\n# Available tools\n" + "\n".join(
        json.dumps(tool, ensure_ascii=False, sort_keys=True) for tool in tools
    )
