from __future__ import annotations

from pathlib import Path

import pytest

from ecommerce_rag.grpo.config import FROZEN_CONFIG
from ecommerce_rag.grpo.metrics import GroupArtifact
from ecommerce_rag.grpo.reward_adapter import OfficialTerminalRewardAdapter, RewardAdapterError
from ecommerce_rag.grpo.trajectory_schema import TokenSegment, assistant_only_loss_mask
from scripts.train_tau3_grpo import _verl_command


def test_frozen_pilot_shape_and_values():
    FROZEN_CONFIG.validate()
    assert FROZEN_CONFIG.train_tasks == 74
    assert FROZEN_CONFIG.group_size_k == 8
    assert FROZEN_CONFIG.groups_per_step_p == 2
    assert FROZEN_CONFIG.rollouts_per_step == 16
    assert FROZEN_CONFIG.total_steps == 9


def test_group_artifact_keeps_all_eight_rollouts_and_population_variance():
    artifact = GroupArtifact.create("g", "17", [0, 1, 0, 1, 0, 1, 0, 1])
    assert artifact.rollout_indices == tuple(range(8))
    assert artifact.reward_class == "mixed"
    assert artifact.mean_reward == 0.5
    assert artifact.variance == 0.25


def test_reward_adapter_rejects_nonterminal_and_infrastructure_zero():
    adapter = OfficialTerminalRewardAdapter()
    with pytest.raises(RewardAdapterError):
        adapter.from_step(terminated=False, reward=0.0, info={})
    with pytest.raises(RewardAdapterError):
        adapter.from_step(
            terminated=True,
            reward=0.0,
            info={"interaction_error": "orchestrator failed"},
        )


def test_assistant_only_loss_mask_excludes_environment_tokens():
    segments = [
        TokenSegment("system", (1, 2)),
        TokenSegment("user", (3,)),
        TokenSegment("assistant", (4, 5)),
        TokenSegment("tool", (6,)),
    ]
    assert assistant_only_loss_mask(segments) == [0, 0, 0, 1, 1, 0]


def test_nscc_launcher_is_synchronous_and_supports_both_step_modes(tmp_path):
    command = _verl_command(
        tmp_path / "train.parquet",
        tmp_path,
        optimizer_steps=1,
    )
    assert "verl.trainer.main_ppo_sync" in command
    assert "verl.experimental.one_step_off_policy.main_ppo" not in command
    assert "actor_rollout_ref.hybrid_engine=True" in command
    assert "actor_rollout_ref.rollout.mode=sync" in command
    assert "trainer.n_gpus_per_node=2" in command
    assert "trainer.total_training_steps=1" in command

    formal = _verl_command(
        tmp_path / "train.parquet",
        tmp_path,
        optimizer_steps=FROZEN_CONFIG.total_steps,
    )
    assert f"trainer.total_training_steps={FROZEN_CONFIG.total_steps}" in formal


def test_frozen_config_records_the_track_a_nl_judge():
    assert FROZEN_CONFIG.nl_assertions_model == "deepseek/deepseek-chat"
