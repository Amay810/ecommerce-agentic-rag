"""Frozen configuration for the first tau3 retail GRPO pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU2_VERSION = "1.0.1"


@dataclass(frozen=True)
class FrozenTau3GRPOConfig:
    """The experiment variables that must not drift during the pilot.

    Optimizer and memory settings are deliberately absent: those are runtime
    implementation choices, not frozen scientific variables.
    """

    tau2_commit: str = TAU2_COMMIT
    tau2_version: str = TAU2_VERSION
    domain: str = "retail"
    train_split: str = "train"
    train_tasks: int = 74
    agent_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    user_model: str = "deepseek-v4-flash"
    agent_temperature: float = 0.8
    user_temperature: float = 0.0
    seed: int = 300
    max_steps: int = 200
    group_size_k: int = 8
    groups_per_step_p: int = 2
    total_steps: int = 9
    reward: str = "official_tau2_terminal_binary"
    evaluation_type: str = "ALL"

    @property
    def rollouts_per_step(self) -> int:
        return self.group_size_k * self.groups_per_step_p

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["rollouts_per_step"] = self.rollouts_per_step
        return values

    def validate(self) -> None:
        expected = {
            "tau2_commit": TAU2_COMMIT,
            "tau2_version": TAU2_VERSION,
            "domain": "retail",
            "train_split": "train",
            "train_tasks": 74,
            "agent_model": "Qwen/Qwen3-4B-Instruct-2507",
            "user_model": "deepseek-v4-flash",
            "agent_temperature": 0.8,
            "user_temperature": 0.0,
            "seed": 300,
            "max_steps": 200,
            "group_size_k": 8,
            "groups_per_step_p": 2,
            "total_steps": 9,
            "reward": "official_tau2_terminal_binary",
            "evaluation_type": "ALL",
        }
        actual = self.as_dict()
        for key, value in expected.items():
            if actual[key] != value:
                raise ValueError(
                    f"frozen tau3 variable changed: {key}={actual[key]!r}; "
                    f"expected {value!r}"
                )


FROZEN_CONFIG = FrozenTau3GRPOConfig()
