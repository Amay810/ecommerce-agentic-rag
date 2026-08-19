"""Per-step GRPO pilot artifact and provenance writer."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Iterable


def parameter_fingerprint(parameters: Iterable[Any]) -> str:
    """Hash parameter bytes when torch tensors are available."""
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().float().cpu().numpy().tobytes()
        digest.update(value)
    return digest.hexdigest()


def classify_rewards(rewards: Iterable[float]) -> str:
    values = [float(value) for value in rewards]
    if len(values) != 8 or any(value not in (0.0, 1.0) for value in values):
        raise ValueError("each GRPO group must contain exactly 8 binary rewards")
    successes = int(sum(values))
    return "0/8" if successes == 0 else "8/8" if successes == 8 else "mixed"


@dataclass(frozen=True)
class GroupArtifact:
    group_id: str
    task_id: str
    rollout_indices: tuple[int, ...]
    rewards: tuple[float, ...]
    mean_reward: float
    variance: float
    reward_class: str

    @classmethod
    def create(cls, group_id: str, task_id: str, rewards: Iterable[float]) -> "GroupArtifact":
        values = tuple(float(value) for value in rewards)
        if len(values) != 8:
            raise ValueError("GRPO group must contain K=8 rewards")
        return cls(
            group_id=group_id,
            task_id=str(task_id),
            rollout_indices=tuple(range(8)),
            rewards=values,
            mean_reward=float(mean(values)),
            variance=float(pvariance(values)),
            reward_class=classify_rewards(values),
        )


@dataclass(frozen=True)
class TimingBreakdown:
    rollout_seconds: float
    update_seconds: float
    sync_seconds: float
    other_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.rollout_seconds + self.update_seconds + self.sync_seconds + self.other_seconds


@dataclass
class StepArtifact:
    run_id: str
    step: int
    policy_version_before: str
    policy_version_after: str
    checkpoint_before: str | None
    checkpoint_after: str | None
    groups: list[GroupArtifact] = field(default_factory=list)
    training_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    optimizer_step: int | None = None
    parameter_fingerprint_before: str | None = None
    parameter_fingerprint_after: str | None = None
    timing: TimingBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepArtifactWriter:
    def __init__(self, output_dir: str | Path, *, run_id: str, config: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = self.output_dir / "steps.jsonl"
        (self.output_dir / "frozen_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write(self, artifact: StepArtifact) -> None:
        if artifact.run_id != self.run_id:
            raise ValueError("artifact run_id does not match writer")
        if artifact.timing is not None:
            values = asdict(artifact.timing)
            if any(not math.isfinite(float(value)) or value < 0 for value in values.values()):
                raise ValueError("timing values must be finite and non-negative")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact.to_dict(), ensure_ascii=False) + "\n")
