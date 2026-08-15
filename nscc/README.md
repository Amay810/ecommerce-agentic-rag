# NSCC jobs

Workdir: `/scratch/users/ntu/s250045/ecommerce-agentic-rag-main`
(`qsub`/`qstat` in `/opt/pbs/bin`). Do not submit from sibling scratch copies.

Windows is the internet-connected orchestrator (DeepSeek user/judge). NSCC
serves the agent. Current stop: [docs/current_status.md](../docs/current_status.md).

| Job | Use |
|---|---|
| `serve_tau3_agent_v1.pbs` | Qwen3-4B Base |
| `run_tau3_retail_base_v1.pbs` | cluster-side Base eval helper |
| `serve_teacher_v1.pbs` / `run_ecommerce_m1_lora_v1.pbs` | kept in tree; **do not run** |

Current compiled artifact: `data/compiled_retail_m1/` (47 structures / 376 tasks).
The ~222 compiled 4B results belong to the archived 48-structure snapshot, not this set.
