"""Synthetic two-GPU optimizer-only smoke for the resolved Tau3 FSDP stack.

This deliberately does not import Tau2, call DeepSeek, start Ray, or start
vLLM. It loads the real Qwen model through VERL's FSDPEngine, uses the same
bfloat16/FSDP/optimizer-offload/foreach=false settings, and records memory at
each training stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from verl.trainer.config.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _memory(stage: str) -> dict[str, object]:
    torch.cuda.synchronize()
    return {
        "stage": stage,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _stage(records: list[dict[str, object]], name: str, fn=None):
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    value = fn() if fn is not None else None
    torch.cuda.synchronize()
    records.append(_memory(name))
    dist.barrier()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ["QWEN_MODEL_PATH"])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/results/tau3_optimizer_only_smoke.json"),
    )
    args = parser.parse_args()

    if args.seq_len <= 0 or args.seq_len > 16384:
        raise ValueError("seq_len must be in 1..16384")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this optimizer-only smoke")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    records: list[dict[str, object]] = []
    engine = None
    try:
        model_config = HFModelConfig(
            path=args.model_path,
            trust_remote_code=True,
            override_config={"attn_implementation": "sdpa"},
            enable_gradient_checkpointing=True,
            enable_activation_offload=False,
            use_remove_padding=True,
        )
        engine_config = FSDPEngineConfig(
            param_offload=False,
            optimizer_offload=True,
            strategy="fsdp",
            dtype="bfloat16",
            use_dynamic_bsz=True,
            max_token_len_per_gpu=16384,
            micro_batch_size_per_gpu=1,
            use_remove_padding=True,
            wrap_policy={"min_num_params": 0},
            use_torch_compile=True,
        )
        optimizer_config = FSDPOptimizerConfig(
            lr=1e-6,
            weight_decay=0.01,
            total_training_steps=1,
            optimizer="AdamW",
            optimizer_impl="torch.optim",
            override_optimizer_config={"foreach": False},
        )
        checkpoint_config = CheckpointConfig(
            save_contents=["model", "optimizer", "extra"],
            load_contents=["model", "optimizer", "extra"],
            strict=True,
        )

        engine = FSDPEngine(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        engine.initialize()
        records.append(_memory("initialized_optimizer_offloaded"))

        input_ids = torch.randint(
            low=0,
            high=10000,
            size=(1, args.seq_len),
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        attention_mask = torch.ones_like(input_ids)
        with engine.train_mode():
            before_checksum = float(
                sum(p.detach().float().sum().item() for p in engine.module.parameters())
            )
            _stage(records, "before_forward")

            def forward_loss():
                output = engine.module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
                return output.logits.float().square().mean()

            loss = _stage(records, "forward_loss", forward_loss)
            loss_value = float(loss.detach().item())
            _stage(records, "backward", loss.backward)

            def grad_norm():
                value = engine.module.clip_grad_norm_(1.0)
                return float(value.item() if hasattr(value, "item") else value)

            grad_norm_value = _stage(records, "grad_norm", grad_norm)
            _stage(records, "optimizer_step", engine.optimizer.step)
            after_checksum = float(
                sum(p.detach().float().sum().item() for p in engine.module.parameters())
            )
            _stage(records, "zero_grad", engine.optimizer_zero_grad)

        _stage(records, "after_train_context")
        local_result = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "world_size": world_size,
            "seq_len": args.seq_len,
            "micro_batch_size_per_gpu": 1,
            "dtype": "bfloat16",
            "fsdp": True,
            "optimizer": "torch.optim.AdamW",
            "optimizer_offload": True,
            "optimizer_override": {"foreach": False},
            "loss": loss_value,
            "grad_norm": grad_norm_value,
            "parameter_checksum_before": before_checksum,
            "parameter_checksum_after": after_checksum,
            "parameter_changed": before_checksum != after_checksum,
            "memory": records,
        }
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_result)
        if rank == 0:
            report = {
                "status": "PASS",
                "world_size": world_size,
                "gpus": gathered,
                "optimizer_step_completed": True,
                "all_parameters_changed": all(
                    bool(item and item["parameter_changed"]) for item in gathered
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        dist.barrier()
        return 0
    finally:
        if engine is not None:
            del engine
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
