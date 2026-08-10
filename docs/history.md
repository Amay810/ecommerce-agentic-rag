# Project history and closed experiments

The active repository contains the current tool Agent, retrieval, safety,
native function calling, MCP, and verified-learning path. Retired code and raw
artifacts are kept in the owner's private repository
`Amay810/ecommerce-agentic-rag-archive` rather than in the active import path.

| Stage | Frozen result | Current status |
|---|---|---|
| RAG/Streamlit MVP | established hybrid retrieval baseline | orchestration archived; retrieval retained |
| Agent v2 | 84.17% operational success on 120 fixed internal tasks; 0 illegal state changes | core harness/tools retained |
| Terminal grounding | 34/40 → 34/40 fact pass | negative/inconclusive; archived |
| Return workflow fixes | 34/40 → 38/40 from protocol fixes; 40/40 with runtime constraint | constraint retained; not model gain |
| SQL Memory probe | 0 repaired actions; retrieval 12/24 | negative/inconclusive; runtime path removed |
| S0 LoRA | Base 85/160, S0 86/160 on τ³ Retail test | negative control; training/deployment chain proven, artifacts archived |
| Hint self-distillation | improved terminal success but failed process-compliance requirement | stopped; archived |

These outcomes explain design decisions; they are not presented as current
features or as model-capability improvements. Exact code and artifacts remain
recoverable from the private archive and the original repository's Git history.
