# Current status

Last updated: 2026-08-15

## Decision

Do not run a 30B full teacher rollout. Do not start M1 LoRA.
The 400-train gate is not met.

## Frozen evidence

| Run | Result | Do not |
|---|---|---|
| τ³ train job 89 | 73/74 valid | rerun |
| Remaining miss | `ecommerce_native` mixed content+tool_call after infra was fixed | treat as DNS/proxy |
| Compiled 4B (~222 usable) | unconfirmed-write 1.4%; reward ~18% from incomplete gold paths | treat as the current 47-structure set |

τ³ unconfirmed-write is 23.3% and still teacher-shaped, but 74 tasks cannot pass the 400-train gate.

## Compiler versions

| Version | Where | Structures | Tasks | S47 |
|---|---|---:|---:|---|
| Historical 4B rollout | archive `data/compiled_retail_m1_v0_48_structures/` | 48 | 384 | yes |
| Current source + committed artifact | `ecommerce_rag/retail_task_compiler/` and `data/compiled_retail_m1/` | 47 | 376 | no |

Future compiled experiments must regenerate under a new version name.

## Next

Phase 1 is a **measurement**, not training: count `ask_user` vs premature write on a frozen probe set (`ecommerce_rag/phase1_write_gate.py`). Train only if that error is common and the label is stable.

Closed planning text and the 48-structure snapshot live in `Amay810/ecommerce-agentic-rag-archive`.
