# Current status

Last updated: 2026-08-17

## Decision

`retail_task_compiler` / `compiled_retail_m1` is **No-Go** and has been removed from
this repository. Do not regenerate compiled tasks, rerun the 30B teacher, or start
M1 LoRA.

The learning path is official τ²/τ³ tasks → on-policy rollout → environment
reward / verifier → rejection sampling → RFT → GRPO. RFT/GRPO training code is
not implemented here yet.

## Frozen evidence

| Run | Result | Do not |
|---|---|---|
| τ³ train job 89 | 73/74 valid | rerun |
| Remaining miss | `ecommerce_native` mixed content+tool_call after infra was fixed | treat as DNS/proxy |
| Compiled 4B (~222 usable) | unconfirmed-write 1.4%; reward ~18% from incomplete gold paths | treat as a live compiler |

Closed compiler source, the 47-structure/376-task artifact, and the 48-structure
compiler-report snapshot live in
[`Amay810/ecommerce-agentic-rag-archive`](https://github.com/Amay810/ecommerce-agentic-rag-archive).

## Next

Phase 1 write-gate (`ecommerce_rag/phase1_write_gate.py`) remains a **measurement**
on the local harness: count `ask_user` vs premature write. It is not compiled-task
training.

Official τ³ Retail evaluation (`scripts/run_tau3_retail_v1.py`) is the external
eval path.
