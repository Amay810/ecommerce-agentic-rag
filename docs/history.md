# Project history and closed experiments

Retired code and raw artifacts live in the private
[`Amay810/ecommerce-agentic-rag-archive`](https://github.com/Amay810/ecommerce-agentic-rag-archive).
Current decisions are in [current status](current_status.md).

| Stage | Frozen result | Status |
|---|---|---|
| RAG/Streamlit MVP | hybrid retrieval baseline | orchestration archived; retrieval retained |
| Agent v2 | 84.17% operational success on 120×3; 0 illegal state changes | harness/tools retained |
| Terminal grounding | 34/40 → 34/40 fact pass | archived |
| Return workflow fixes | 34/40 → 40/40 from protocol then runtime constraint | constraint retained; not model gain |
| SQL Memory probe | 0 repaired actions | archived |
| S0 LoRA | Base 85/160, S0 86/160 on τ³ Retail test | negative control; archived |
| Hint self-distillation | terminal success up, process compliance failed | archived |
| 30B full teacher / M1 LoRA | compiled 4B already rarely writes unconfirmed; 74 τ³ tasks cannot pass the 400-train gate | **No-Go**; not started |

Compiler versioning: the ~222 compiled 4B trajectories belong to a 48-structure
set that was never committed as `tasks.json`. The archive keeps the compiler
*report* (IDs and splits only). Current source and `data/compiled_retail_m1/`
are 47 structures / 376 tasks.
