# NSCC jobs

Workdir: `/scratch/users/ntu/s250045/ecommerce-agentic-rag-main`
(`qsub`/`qstat` in `/opt/pbs/bin`). Do not submit from sibling scratch copies.

Windows is the internet-connected orchestrator (DeepSeek user/judge). NSCC
serves the agent. Current stop: [docs/current_status.md](../docs/current_status.md).

| Job | Use |
|---|---|
| `serve_tau3_agent_v1.pbs` | Qwen3-4B Base |
| `run_tau3_retail_base_v1.pbs` | cluster-side Base eval helper |
| `run_ecommerce_m1_lora_v1.pbs` | fail-closed LoRA; do not submit until the 400-train gate passes |
| `serve_ecommerce_m1_v1.pbs` | serve a future M1 adapter |

30B teacher serve/rollout jobs and the legacy action-constraint PBS are in the
archive, not this tree.
