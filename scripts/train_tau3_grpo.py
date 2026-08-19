"""Launch or offline-check the frozen tau3 retail GRPO pilot.

The VM-safe default is ``--check-only``.  ``--launch`` is intentionally a
thin VERL command launcher and must only be used on the NSCC training node.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ecommerce_rag.grpo.config import FROZEN_CONFIG
from ecommerce_rag.grpo.metrics import (
    GroupArtifact,
    StepArtifact,
    StepArtifactWriter,
    TimingBreakdown,
)
from ecommerce_rag.grpo.tau3_env_adapter import retail_train_task_ids, validate_snapshot
from ecommerce_rag.grpo.trajectory_schema import (
    TokenSegment,
    assert_assistant_only_mask,
    assistant_only_loss_mask,
)


@dataclass
class FakePolicy:
    version: str = "policy-0000"
    checkpoint: str | None = "fake://checkpoint-0000"
    fingerprint: str = "fake-fingerprint-0000"

    def update(self, step: int) -> None:
        self.version = f"policy-{step:04d}"
        self.checkpoint = f"fake://checkpoint-{step:04d}"
        self.fingerprint = f"fake-fingerprint-{step:04d}"


def _check_mask_contract() -> None:
    segments = [
        TokenSegment("system", (1, 2)),
        TokenSegment("user", (3,)),
        TokenSegment("assistant", (4, 5, 6)),
        TokenSegment("tool", (7, 8)),
        TokenSegment("assistant", (9,)),
        TokenSegment("evaluator", (10,)),
    ]
    expected = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
    actual = assistant_only_loss_mask(segments)
    if actual != expected:
        raise AssertionError(f"unexpected assistant-only mask: {actual}")
    assert_assistant_only_mask(segments, actual)


def _check_group_contract() -> list[GroupArtifact]:
    groups = [
        GroupArtifact.create("check-g0", "0", [0, 1, 0, 1, 0, 1, 0, 1]),
        GroupArtifact.create("check-g1", "1", [0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    if [group.rollout_indices for group in groups] != [tuple(range(8))] * 2:
        raise AssertionError("rollout indices are not 0..7")
    return groups


def _check_artifact_chain(output_dir: Path, *, steps: int) -> dict[str, object]:
    config = FROZEN_CONFIG
    config.validate()
    if steps not in (1, config.total_steps):
        raise ValueError("offline checks support exactly 1-step and frozen 9-step runs")
    _check_mask_contract()
    groups = _check_group_contract()
    run_id = f"offline-check-{uuid4().hex[:12]}"
    writer = StepArtifactWriter(output_dir, run_id=run_id, config=config.as_dict())
    policy = FakePolicy()
    for step in range(1, steps + 1):
        before = policy.version
        checkpoint_before = policy.checkpoint
        fingerprint_before = policy.fingerprint
        started = time.perf_counter()
        policy.update(step)
        artifact = StepArtifact(
            run_id=run_id,
            step=step,
            policy_version_before=before,
            policy_version_after=policy.version,
            checkpoint_before=checkpoint_before,
            checkpoint_after=policy.checkpoint,
            groups=groups,
            training_loss=1.0 / step,
            learning_rate=1e-6,
            grad_norm=0.5,
            optimizer_step=step,
            parameter_fingerprint_before=fingerprint_before,
            parameter_fingerprint_after=policy.fingerprint,
            timing=TimingBreakdown(
                rollout_seconds=0.001,
                update_seconds=max(time.perf_counter() - started, 0.0),
                sync_seconds=0.001,
                other_seconds=0.001,
            ),
        )
        writer.write(artifact)
    lines = writer.path.read_text(encoding="utf-8").splitlines()
    if len(lines) != steps:
        raise AssertionError(f"expected {steps} step artifacts, got {len(lines)}")
    return {
        "status": "PASS",
        "steps": len(lines),
        "groups_per_step": config.groups_per_step_p,
        "rollouts_per_step": config.rollouts_per_step,
        "mask": "assistant-only",
        "reward": config.reward,
        "artifact": str(writer.path),
    }


def _build_dataset(tau_root: str, output_dir: Path) -> Path:
    """Write task metadata only; episode prompts are built after env.reset()."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - NSCC dependency
        raise RuntimeError("datasets is required to build the VERL parquet") from exc
    task_ids = retail_train_task_ids(tau_root)
    rows = []
    for index, task_id in enumerate(task_ids):
        rows.append(
            {
                "data_source": "tau3_retail",
                "prompt": [{"role": "user", "content": f"tau3 retail task {task_id}"}],
                "ability": "retail",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "official_tau2_terminal_binary",
                },
                "extra_info": {
                    "split": "train",
                    "index": index,
                    "tau_root": str(tau_root),
                    "domain": "retail",
                    "task_id": task_id,
                    "seed": FROZEN_CONFIG.seed,
                    "max_steps": FROZEN_CONFIG.max_steps,
                    "user_model": FROZEN_CONFIG.user_model,
                    "user_base_url": os.environ.get("DEEPSEEK_BASE_URL", ""),
                    "user_temperature": FROZEN_CONFIG.user_temperature,
                },
            }
        )
    output = output_dir / "tau3_retail_train.parquet"
    Dataset.from_list(rows).to_parquet(str(output))
    return output


def _verl_command(
    train_file: Path, output_dir: Path, *, optimizer_steps: int
) -> list[str]:
    cfg = FROZEN_CONFIG
    if optimizer_steps not in (1, cfg.total_steps):
        raise ValueError("optimizer_steps must be 1 or the frozen 9-step count")
    model_path = os.environ.get("QWEN_MODEL_PATH", cfg.agent_model)
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo_sync",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"seed={cfg.seed}",
        f"data.train_files={train_file}",
        f"data.val_files={train_file}",
        "data.train_batch_size=2",
        "data.val_batch_size=2",
        "data.max_prompt_length=8192",
        "data.max_response_length=4096",
        "data.return_raw_chat=True",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.02",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg.group_size_k}",
        f"actor_rollout_ref.rollout.temperature={cfg.agent_temperature}",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.max_model_len=12288",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.70",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=200",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=200",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={Path(__file__).resolve().parents[1] / 'ecommerce_rag' / 'grpo' / 'agent_loop.yaml'}",
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "reward.num_workers=1",
        "trainer.project_name=tau3_grpo",
        "trainer.experiment_name=tau3_retail_grpo_v1",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.save_freq=3",
        "trainer.test_freq=-1",
        f"trainer.rollout_data_dir={output_dir / 'verl_rollouts'}",
        f"trainer.total_training_steps={optimizer_steps}",
        f"trainer.default_local_dir={output_dir / 'checkpoints'}",
        'trainer.logger=["console"]',
        "actor_rollout_ref.rollout.mode=sync",
        "actor_rollout_ref.hybrid_engine=True",
        "ray_kwargs.ray_init.num_cpus=16",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", default=os.environ.get("TAU_ROOT"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tau3_grpo_pilot"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--launch", action="store_true", help="launch VERL; NSCC-only")
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        choices=(1, FROZEN_CONFIG.total_steps),
        default=FROZEN_CONFIG.total_steps,
        help="NSCC launch length: 1 for plumbing smoke or the frozen 9-step pilot.",
    )
    args = parser.parse_args()
    if args.check_only == args.launch:
        parser.error("choose exactly one of --check-only or --launch")

    FROZEN_CONFIG.validate()
    if args.tau_root:
        validate_snapshot(args.tau_root)
        task_ids = retail_train_task_ids(args.tau_root)
        print(json.dumps({"tau_root": str(args.tau_root), "retail_train_tasks": len(task_ids)}))
    elif args.launch:
        parser.error("--launch requires --tau-root or TAU_ROOT")

    if args.launch and not os.environ.get("DEEPSEEK_BASE_URL"):
        parser.error("--launch requires DEEPSEEK_BASE_URL")
    if args.launch and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("--launch requires DEEPSEEK_API_KEY")

    if args.check_only:
        result = {
            "one_step": _check_artifact_chain(args.output_dir / "one_step", steps=1),
            "nine_step": _check_artifact_chain(args.output_dir / "nine_step", steps=FROZEN_CONFIG.total_steps),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    train_file = _build_dataset(args.tau_root, args.output_dir)
    command = _verl_command(
        train_file, args.output_dir, optimizer_steps=args.optimizer_steps
    )
    (args.output_dir / "launch_command.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "launch_command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Launching VERL; no VM/CI path should call this mode.")
    return subprocess.call(command, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
