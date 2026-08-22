from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path
from datasets import Dataset
from ecommerce_rag.grpo.config import FROZEN_CONFIG
from ecommerce_rag.grpo.tau3_env_adapter import retail_train_task_ids, validate_snapshot
from scripts.train_tau3_grpo import _verl_command

def _one_task(tau_root: str, out: Path) -> tuple[Path, str]:
    ids = retail_train_task_ids(tau_root)
    if len(ids) != FROZEN_CONFIG.train_tasks:
        raise RuntimeError(f"expected {FROZEN_CONFIG.train_tasks} tasks, found {len(ids)}")
    task_id = str(ids[0])
    row = {
        "data_source": "tau3_retail",
        "prompt": [{"role": "user", "content": f"tau3 retail task {task_id}"}],
        "ability": "retail",
        "reward_model": {"style": "rule", "ground_truth": "official_tau2_terminal_binary"},
        "extra_info": {
            "split": "train", "index": 0, "tau_root": str(tau_root),
            "domain": "retail", "task_id": task_id, "seed": FROZEN_CONFIG.seed,
            "max_steps": FROZEN_CONFIG.max_steps, "user_model": FROZEN_CONFIG.user_model,
            "user_base_url": os.environ.get("DEEPSEEK_BASE_URL", ""),
            "user_temperature": FROZEN_CONFIG.user_temperature,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tau3_retail_one_task.parquet"
    Dataset.from_list([row]).to_parquet(str(path))
    return path, task_id

def _replace(cmd: list[str], prefix: str, value: str) -> None:
    for i, arg in enumerate(cmd):
        if arg.startswith(prefix):
            cmd[i] = value
            return
    raise RuntimeError(f"missing launcher arg: {prefix}")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=int, choices=(1, 2), required=True)
    p.add_argument("--tau-root", default=os.environ.get("TAU_ROOT"))
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()
    if not a.tau_root:
        p.error("TAU_ROOT is required")
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("DEEPSEEK_BASE_URL"):
        p.error("DeepSeek environment is required")
    out = a.output_dir or Path(f"/root/autodl-tmp/results/tau3_staged_rollout_1x{a.rollouts}")
    if out.exists() and any(out.iterdir()):
        p.error(f"output directory must be empty: {out}")
    root = validate_snapshot(a.tau_root)
    path, task_id = _one_task(str(root), out)
    cmd = _verl_command(path, out, optimizer_steps=1)
    _replace(cmd, "data.train_batch_size=", "data.train_batch_size=1")
    _replace(cmd, "data.val_batch_size=", "data.val_batch_size=1")
    _replace(cmd, "actor_rollout_ref.actor.ppo_mini_batch_size=", "actor_rollout_ref.actor.ppo_mini_batch_size=1")
    _replace(cmd, "actor_rollout_ref.rollout.n=", f"actor_rollout_ref.rollout.n={a.rollouts}")
    _replace(cmd, "actor_rollout_ref.rollout.gpu_memory_utilization=", "actor_rollout_ref.rollout.gpu_memory_utilization=0.55")
    _replace(cmd, "trainer.n_gpus_per_node=", "trainer.n_gpus_per_node=1")
    _replace(cmd, "trainer.save_freq=", "trainer.save_freq=-1")
    _replace(cmd, "trainer.test_freq=", "trainer.test_freq=-1")
    cmd.extend(['actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'trainer.critic_warmup=2', 'trainer.total_training_steps=1'])
    (out / "launch_command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc:
        return rc
    records = []
    for file in sorted((out / "verl_rollouts").glob("*.jsonl")):
        records += [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != a.rollouts:
        raise RuntimeError(f"expected {a.rollouts} records, found {len(records)}")
    uids = [str(x.get("uid", "")) for x in records]
    scores = [x.get("score") for x in records]
    if any(not x for x in uids) or len(set(uids)) != a.rollouts:
        raise RuntimeError(f"uid evidence incomplete: {uids}")
    if any(float(x) not in (0.0, 1.0) for x in scores):
        raise RuntimeError(f"non-binary terminal reward: {scores}")
    if any(x.get("gts") != "official_tau2_terminal_binary" for x in records):
        raise RuntimeError("official tau2 ground-truth marker missing")
    if any(not str(x.get("output", "")).strip() for x in records):
        raise RuntimeError("empty staged trajectory output")
    result = {
        "status": "PASS", "task_count": 1, "task_id": task_id,
        "rollouts": a.rollouts, "unique_rollout_uids": len(set(uids)),
        "trajectory_records": len(records),
        "official_terminal_reward": {
            "scores": scores, "binary": True,
            "ground_truth_marker": "official_tau2_terminal_binary",
            "source_wiring": "Tau3RolloutSession -> OfficialTerminalRewardAdapter -> tau2.evaluator.evaluate_simulation",
        },
        "agent_loop": "tau3_agent",
        "actor_update": "skipped_for_staged_rollout_only",
        "gpu_visible_to_stage": "0",
    }
    (out / "staged_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
