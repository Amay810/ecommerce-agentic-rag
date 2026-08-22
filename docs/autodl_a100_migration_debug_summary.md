# AutoDL A100 migration debug summary

- Generated: 2026-08-21
- Source instance hostname: autodl-container-a3d5118ffa-6a9d8e7b
- Purpose: internal troubleshooting record for migration; not a project showcase.
- Migration preparation did not start CUDA, vLLM, VERL, rollout, training, or GPU memory tests.

## Source software record

- ecommerce-agentic-rag HEAD: 48301f3eecf0dc80754dd03589f00a7381e21629
- tau2-bench HEAD: fc0055dc4e0a316c3f83133267fbd6faaa770992
- VERL commit: 7aed6b230776f963fa09509c10d9c3a767d1102c
- Python: 3.12.13
- PyTorch: 2.8.0+cu128
- vLLM: 0.10.2
- Ray: 2.56.1
- Transformers: 4.57.6
- NumPy: 1.26.4

## Previously recorded PASS items

The prior audit record reports successful validation of the basic 2-GPU CUDA/NCCL/Ray/FSDP/vLLM/VERL construction path, custom tau3 agent-loop wiring, fresh tau2 episodes, DeepSeek connectivity, multi-turn simulation, Qwen generation in a real tau2 episode, official terminal reward wiring, and 16-rollout completion to the actor-update entry point.

These items were not re-run during migration preparation.

## Retry5/6/7 conclusions

- retry5: real GRPO actor optimizer peak OOM; the failing GPU had about 162.88 MiB remaining.
- retry6: optimizer_offload=True still reached optimizer peak OOM.
- retry7: optimizer_offload=True plus foreach=False; the run stopped during rollout and did not prove a real GRPO optimizer-step pass.

## Current blocker

The unresolved issue is real-GRPO GPU residency and peak memory across rollout sleep, old/ref log-prob computation, actor backward, and optimizer.step. The fixed VERL source shows that the reference FSDP1 forward-only path uses CPU offload internally, but ref.param_offload=True itself is not a proven guarantee that all CUDA storage is released before actor update. Historical main_ppo_sync.log files were empty, so stage-level runtime memory evidence is incomplete.

## Migration follow-up checks

After restoring the staged data to the original /root/autodl-tmp paths on A800, verify:

1. conda environment activation and executable shebangs;
2. exact Python/package versions and pinned VERL commit;
3. ecommerce and tau2 git HEAD/dirty changes;
4. Qwen safetensor index and all referenced weight files;
5. hard-coded /root/autodl-tmp references;
6. persistent stdout/stderr/traceback logging;
7. only then, under separate authorization, the minimum required GPU validation.

No API keys, passwords, or authentication material are recorded here.
