"""Validate per-step GRPO artifacts without touching model checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ecommerce_rag.grpo.config import FROZEN_CONFIG


def analyze(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != FROZEN_CONFIG.total_steps:
        raise ValueError(f"expected {FROZEN_CONFIG.total_steps} step rows, got {len(rows)}")
    classes = Counter()
    for expected_step, row in enumerate(rows, start=1):
        if row["step"] != expected_step:
            raise ValueError(f"step sequence is not contiguous at {expected_step}")
        groups = row["groups"]
        if len(groups) != FROZEN_CONFIG.groups_per_step_p:
            raise ValueError("P=2 group count drifted")
        if row["policy_version_before"] == row["policy_version_after"]:
            raise ValueError("policy version did not refresh after update")
        for group in groups:
            if group["rollout_indices"] != list(range(FROZEN_CONFIG.group_size_k)):
                raise ValueError("group rollout indices are not 0..7")
            if len(group["rewards"]) != FROZEN_CONFIG.group_size_k:
                raise ValueError("group K drifted")
            classes[group["reward_class"]] += 1
    return {
        "steps": len(rows),
        "groups": len(rows) * FROZEN_CONFIG.groups_per_step_p,
        "rollouts": len(rows) * FROZEN_CONFIG.rollouts_per_step,
        "reward_classes": dict(classes),
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.steps), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
