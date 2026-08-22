"""CPU-only audit of the final Tau3 VERL/Hydra wiring.

This composes the exact launcher command with Hydra's --cfg job --resolve.
It never initializes Ray, vLLM, Tau2, DeepSeek, or a CUDA training worker.
"""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path

import yaml

from ecommerce_rag.grpo.config import FROZEN_CONFIG
from scripts.train_tau3_grpo import _verl_command


def _resolved_config(command: list[str]) -> dict:
    proc = subprocess.run(
        command + ["--cfg", "job", "--resolve"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Hydra composition failed with exit {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
        )
    text = proc.stdout
    start = text.find("model_engine:")
    if start < 0:
        raise RuntimeError("Hydra --cfg output did not contain model_engine")
    end_marker = "\nseed: 300"
    end = text.rfind(end_marker)
    if end < start:
        raise RuntimeError("Hydra --cfg output did not end with resolved seed")
    payload = text[start : end + len(end_marker)]
    result = yaml.safe_load(payload)
    if not isinstance(result, dict):
        raise RuntimeError("resolved Hydra payload is not a mapping")
    return result


def _command_map(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in command:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def _resolved_vllm_sleep_level() -> dict[str, object]:
    """Report VERL's source-derived sleep level without importing vLLM."""
    package_version = importlib.metadata.version("vllm")
    major, minor, patch = (
        int(part) for part in package_version.split("+", 1)[0].split(".")[:3]
    )
    version_tuple = (major, minor, patch)
    return {
        "value": 2 if version_tuple >= (0, 8, 5) else 1,
        "package_version": package_version,
        "environment_override": None,
        "source": (
            "verl/third_party/vllm/__init__.py:33,46-49; "
            "VERL derives VLLM_SLEEP_LEVEL from the installed vLLM version"
        ),
    }


def run_audit() -> dict[str, object]:
    FROZEN_CONFIG.validate()
    train_file = Path("/root/autodl-tmp/results/tau3_real_smoke_retry7/tau3_retail_train.parquet")
    command = _verl_command(
        train_file,
        Path("/root/autodl-tmp/results/tau3_config_audit"),
        optimizer_steps=1,
    )
    command_map = _command_map(command)
    resolved = _resolved_config(command)

    actor = resolved["actor_rollout_ref"]["actor"]
    optim = actor["optim"]
    fsdp = actor["fsdp_config"]
    rollout = resolved["actor_rollout_ref"]["rollout"]
    ref = resolved["actor_rollout_ref"]["ref"]
    data = resolved["data"]
    model = resolved["actor_rollout_ref"]["model"]
    checkpoint_engine = rollout["checkpoint_engine"]
    vllm_sleep_level = _resolved_vllm_sleep_level()

    mode_overrides = [
        item for item in command
        if item.startswith("actor_rollout_ref.rollout.mode=")
        or item.startswith("+actor_rollout_ref.rollout.mode=")
    ]
    assert resolved["seed"] == 300
    assert FROZEN_CONFIG.groups_per_step_p == 2
    assert data["train_batch_size"] == 2
    assert rollout["n"] == 8
    assert rollout["temperature"] == 0.8
    assert FROZEN_CONFIG.user_temperature == 0.0
    assert FROZEN_CONFIG.max_steps == 200
    assert rollout["log_prob_micro_batch_size_per_gpu"] == 1
    assert ref["log_prob_micro_batch_size_per_gpu"] == 1
    assert resolved["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"] == "sdpa"
    assert rollout["agent"]["default_agent_loop"] == "tau3_agent"
    assert not mode_overrides
    assert rollout["mode"] == "async"
    assert data["max_response_length"] == 8192
    assert rollout["response_length"] == 8192
    assert data["max_prompt_length"] == 8192
    assert rollout["max_model_len"] == 16384
    assert rollout["max_num_batched_tokens"] == 8192
    assert rollout["free_cache_engine"] is True
    assert rollout["tensor_model_parallel_size"] == 1
    assert rollout["gpu_memory_utilization"] == 0.70
    assert rollout["enforce_eager"] is True
    assert checkpoint_engine["backend"] == "naive"
    assert fsdp["strategy"] == "fsdp"
    assert fsdp["param_offload"] is False
    assert fsdp["optimizer_offload"] is True
    assert ref["fsdp_config"]["param_offload"] is False
    assert ref["fsdp_config"]["optimizer_offload"] is False
    assert optim["override_optimizer_config"]["foreach"] is False
    assert command_map["+seed"] == "300"
    assert vllm_sleep_level["value"] == 2

    loop_path = Path(__file__).resolve().parents[1] / "ecommerce_rag/grpo/verl_tau3_agent_loop.py"
    loop_source = loop_path.read_text(encoding="utf-8")
    bridge_source = (loop_path.parent / "rollout_bridge.py").read_text(encoding="utf-8")
    reward_source = (loop_path.parent / "reward_adapter.py").read_text(encoding="utf-8")
    assert "agent_data.response_mask += [0] * len(observation_ids)" in loop_source
    assert "response_mask=agent_data.response_mask" in loop_source
    assert "output.reward_score = float(session.terminal_reward.value)" in loop_source
    assert "OfficialTerminalRewardAdapter" in bridge_source
    assert "tau2.evaluator.evaluate_simulation" in reward_source

    return {
        "status": "PASS",
        "seed": resolved["seed"],
        "p_groups": FROZEN_CONFIG.groups_per_step_p,
        "train_batch_size": data["train_batch_size"],
        "k_rollouts": rollout["n"],
        "rollout_temperature": rollout["temperature"],
        "user_temperature": FROZEN_CONFIG.user_temperature,
        "max_steps": FROZEN_CONFIG.max_steps,
        "rollout_log_prob_micro_batch_size_per_gpu": rollout["log_prob_micro_batch_size_per_gpu"],
        "ref_log_prob_micro_batch_size_per_gpu": ref["log_prob_micro_batch_size_per_gpu"],
        "attention_override": model["override_config"]["attn_implementation"],
        "default_agent_loop": rollout["agent"]["default_agent_loop"],
        "rollout_mode_override_present": bool(mode_overrides),
        "resolved_verl_rollout_mode": rollout["mode"],
        "response_length": data["max_response_length"],
        "rollout_response_length": rollout["response_length"],
        "max_prompt_length": data["max_prompt_length"],
        "max_model_len": rollout["max_model_len"],
        "max_num_batched_tokens": rollout["max_num_batched_tokens"],
        "rollout_free_cache_engine": rollout["free_cache_engine"],
        "rollout_tensor_model_parallel_size": rollout["tensor_model_parallel_size"],
        "rollout_gpu_memory_utilization": rollout["gpu_memory_utilization"],
        "rollout_enforce_eager": rollout["enforce_eager"],
        "checkpoint_engine_backend": checkpoint_engine["backend"],
        "actor_fsdp_strategy": fsdp["strategy"],
        "actor_fsdp_param_offload": fsdp["param_offload"],
        "actor_fsdp_optimizer_offload": fsdp["optimizer_offload"],
        "ref_fsdp_param_offload": ref["fsdp_config"]["param_offload"],
        "ref_fsdp_optimizer_offload": ref["fsdp_config"]["optimizer_offload"],
        "optimizer_override": optim["override_optimizer_config"],
        "vllm_sleep_level": vllm_sleep_level,
        "memory_config_sources": {
            "actor_rollout_ref.rollout.free_cache_engine": (
                "resolved=True; VERL rollout.yaml:53 default; not launcher-explicit"
            ),
            "actor_rollout_ref.rollout.tensor_model_parallel_size": (
                "resolved=1; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "actor_rollout_ref.rollout.gpu_memory_utilization": (
                "resolved=0.70; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "actor_rollout_ref.rollout.enforce_eager": (
                "resolved=True; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "actor_rollout_ref.actor.fsdp_config.param_offload": (
                "resolved=False; VERL engine/fsdp.yaml:12 default; not launcher-explicit"
            ),
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": (
                "resolved=True; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "actor_rollout_ref.ref.fsdp_config.param_offload": (
                "resolved=False; ref/dp_ref.yaml composes engine/fsdp.yaml:12; not launcher-explicit"
            ),
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": (
                "resolved=1; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": (
                "resolved=1; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "actor_rollout_ref.actor.optim.override_optimizer_config.foreach": (
                "resolved=False; launcher train_tau3_grpo.py:_verl_command explicit; "
                "VERL workers/config/optimizer.py merges override into torch optimizer args"
            ),
            "actor_rollout_ref.rollout.response_length": (
                "resolved=8192; VERL rollout.yaml:31 resolves data.max_response_length"
            ),
            "actor_rollout_ref.rollout.max_model_len": (
                "resolved=16384; launcher train_tau3_grpo.py:_verl_command explicit"
            ),
            "VLLM_SLEEP_LEVEL": (
                "resolved=2 from installed vLLM 0.10.2; VERL source-derived, no env/launcher override"
            ),
        },
        "assistant_only_loss_mask_path": (
            "Tau3AgentLoop response_mask: assistant tokens=1, "
            "tau2/user observations=0; VERL postprocess forwards response_mask to PPO loss"
        ),
        "reward_adapter_path": "Tau3RolloutSession -> OfficialTerminalRewardAdapter -> tau2.evaluator.evaluate_simulation",
        "data_source_reward_registry": (
            "bypassed by precomputed AgentLoopOutput.reward_score; "
            "VERL default reward computation runs only when reward_score is None"
        ),
        "data_source": "tau3_retail",
    }


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
