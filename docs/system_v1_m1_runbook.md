# System-v1 runbook

Pinned τ³: `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
Active NSCC checkout: `/scratch/users/ntu/s250045/ecommerce-agentic-rag-main`.

Current stop: see [current status](current_status.md). Do not serve the 30B teacher or submit M1 LoRA.

## Windows orchestrator (internet-connected)

DeepSeek user/judge stay on Windows. Agent traffic goes through the vLLM tunnel.

```powershell
python -m scripts.run_tau3_retail_v1 `
  --tau-root E:\cv_codex\external\tau2-bench `
  --phase base `
  --agent-name ecommerce_native `
  --compaction off `
  --agent-model hosted_vllm/Qwen3-4B-Instruct-2507 `
  --user-model $env:TAU3_USER_MODEL `
  --nl-assertions-model $env:TAU3_NL_ASSERTIONS_MODEL `
  --pass-k 4 `
  --max-steps 200 `
  --max-concurrency 1 `
  --save-to tau3_system_v1_base_qwen3_4b_k4
```

Use `--max-concurrency 1` when remaining long-horizon tasks stall. Enable Windows long paths before `--verbose-logs` on compiled task ids.

Compiled tasks (current 47/376 artifact):

```powershell
python -m scripts.run_compiled_retail_teacher `
  --tasks data/compiled_retail_m1/tasks.json `
  --splits data/compiled_retail_m1/split_tasks.json `
  --task-split-name train `
  --max-concurrency 1 `
  --save-to compiled_retail_m1_qwen3_4b
```

Phase 1 write-gate probe (no training):

```powershell
python -m scripts.run_phase1_write_gate --check-only
```

## NSCC

Submit from `ecommerce-agentic-rag-main`. Put `/opt/pbs/bin` on `PATH`.
`serve_tau3_agent_v1.pbs` serves Qwen3-4B. M1 LoRA/serve jobs stay fail-closed
until the 400-train gate passes. 30B teacher jobs are in the archive.

Do not `qsub` from `ecommerce-agentic-rag-m1` or sibling scratch copies.
