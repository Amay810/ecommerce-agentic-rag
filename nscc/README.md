# NSCC jobs

Workdir: `/scratch/users/ntu/s250045/ecommerce-agentic-rag-main`
(`qsub`/`qstat` in `/opt/pbs/bin`). Do not submit from sibling scratch copies.

Windows is the internet-connected orchestrator (DeepSeek user/judge). NSCC
serves the agent. Current stop: [docs/current_status.md](../docs/current_status.md).

| Job | Use |
|---|---|
| `serve_tau3_agent_v1.pbs` | Qwen3-4B Base |
| `run_tau3_retail_base_v1.pbs` | cluster-side Base eval helper |
| `train_tau3_grpo_v1.pbs` | two-GPU G0 GRPO implementation smoke/pilot |

30B teacher serve/rollout jobs, M1 LoRA/serve jobs, and the legacy
action-constraint PBS are in the archive, not this tree.

The GRPO job requires the archive's `vendor/tau2-bench-fc0055dc` snapshot via
`TAU_ROOT`, a dedicated VERL/vLLM environment, and a DeepSeek OpenAI-compatible
endpoint. It is not a claim that the local VM can run the pilot.
