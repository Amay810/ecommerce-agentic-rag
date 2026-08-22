from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_tau3_config import run_audit
from scripts.train_tau3_grpo import _verl_command


def _require_verl_hydra():
    """The resolved-config audit shells out to Hydra; skip on machines without VERL."""
    pytest.importorskip("verl")


def _overrides(tmp_path):
    command = _verl_command(tmp_path / "train.parquet", tmp_path, optimizer_steps=1)
    return command, {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in command
        if "=" in item
    }


def test_cpu_resolved_hydra_config_audit_passes():
    _require_verl_hydra()
    report = run_audit()
    assert report["status"] == "PASS"
    assert report["seed"] == 300
    assert report["p_groups"] == 2
    assert report["train_batch_size"] == 2
    assert report["k_rollouts"] == 8
    assert report["rollout_mode_override_present"] is False
    assert report["optimizer_override"] == {"foreach": False}
    assert report["rollout_free_cache_engine"] is True
    assert report["rollout_tensor_model_parallel_size"] == 1
    assert report["rollout_gpu_memory_utilization"] == 0.70
    assert report["rollout_enforce_eager"] is True
    assert report["actor_fsdp_param_offload"] is False
    assert report["actor_fsdp_optimizer_offload"] is True
    assert report["ref_fsdp_param_offload"] is False
    assert report["ref_fsdp_optimizer_offload"] is False
    assert report["checkpoint_engine_backend"] == "naive"
    assert report["vllm_sleep_level"]["value"] == 2


def test_launcher_contains_frozen_and_memory_wiring(tmp_path):
    command, overrides = _overrides(tmp_path)
    assert "+seed" in overrides and overrides["+seed"] == "300"
    assert overrides["data.train_batch_size"] == "2"
    assert overrides["actor_rollout_ref.rollout.n"] == "8"
    assert overrides["actor_rollout_ref.rollout.temperature"] == "0.8"
    assert overrides["data.max_response_length"] == "8192"
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "16384"
    assert overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "1"
    assert overrides["actor_rollout_ref.rollout.gpu_memory_utilization"] == "0.70"
    assert overrides["actor_rollout_ref.rollout.enforce_eager"] == "True"
    assert overrides["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == "1"
    assert overrides["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == "1"
    assert (
        "+actor_rollout_ref.model.override_config.attn_implementation" in overrides
        and overrides["+actor_rollout_ref.model.override_config.attn_implementation"] == "sdpa"
    )
    assert overrides["actor_rollout_ref.rollout.agent.default_agent_loop"] == "tau3_agent"
    assert overrides["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] == "True"
    assert (
        "+actor_rollout_ref.actor.optim.override_optimizer_config"
        in overrides
    )
    assert (
        overrides["+actor_rollout_ref.actor.optim.override_optimizer_config"]
        == "{foreach:false}"
    )
    assert not any(
        token.startswith("actor_rollout_ref.rollout.mode=")
        or token.startswith("+actor_rollout_ref.rollout.mode=")
        for token in command
    )


def test_cpu_memory_provenance_covers_resolved_fields():
    _require_verl_hydra()
    report = run_audit()
    sources = report["memory_config_sources"]
    required = {
        "actor_rollout_ref.rollout.free_cache_engine",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        "actor_rollout_ref.rollout.enforce_eager",
        "actor_rollout_ref.actor.fsdp_config.param_offload",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
        "actor_rollout_ref.ref.fsdp_config.param_offload",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.actor.optim.override_optimizer_config.foreach",
        "actor_rollout_ref.rollout.response_length",
        "actor_rollout_ref.rollout.max_model_len",
        "VLLM_SLEEP_LEVEL",
    }
    assert required <= sources.keys()
    assert all(isinstance(value, str) and value for value in sources.values())


def test_cpu_source_wiring_keeps_mask_and_official_reward_separate():
    root = Path(__file__).parents[1]
    loop = (root / "ecommerce_rag/grpo/verl_tau3_agent_loop.py").read_text()
    bridge = (root / "ecommerce_rag/grpo/rollout_bridge.py").read_text()
    reward = (root / "ecommerce_rag/grpo/reward_adapter.py").read_text()
    assert "agent_data.response_mask += [0] * len(observation_ids)" in loop
    assert "response_mask=agent_data.response_mask" in loop
    assert "output.reward_score = float(session.terminal_reward.value)" in loop
    assert "agent_data._active_tool_schemas = tool_schemas" in loop
    assert "render_tools_for_prompt" not in loop
    assert "OfficialTerminalRewardAdapter" in bridge
    assert "tau2.evaluator.evaluate_simulation" in reward
