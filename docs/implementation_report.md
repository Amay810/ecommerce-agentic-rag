# Credible implementation and evaluation report

## Delivered engineering

- Amazon Reviews 2023 pinned 5,000-product corpus: 2,500 Electronics and
  2,500 Home and Kitchen products, 1,125 reviews and 43,953 child chunks.
- Multilingual dense retrieval, sparse inverted BM25, RRF, constraint filtering,
  optional cross-encoder reranking and parent product cards.
- Deterministic retail environment with 1,000 users, 10,000 orders and eight
  typed tools.
- Identity, policy eligibility and confirmation guardrails for every write.
- TaskSpec, AgentObservation, ToolCall, Trajectory and GradeResult contracts.
- run/replay/compare, terminal-state diff, failure taxonomy, pass@1 and pass^3.
- Fail-closed Agent RL gate that rejects Oracle/Rule traces as training evidence.

## Retrieval evidence

### NSCC scale set

| Products | Recall@5 | nDCG@5 | P95 | Backend |
|---:|---:|---:|---:|---|
| 40 | 1.0000 | 0.9901 | 9.96ms | FAISS FlatIP |
| 1,000 | 0.9958 | 0.9888 | 28.93ms | FAISS FlatIP |
| 5,000 | 0.9917 | 0.9826 | 125.73ms | FAISS FlatIP |

At 5k, BGE reranking raises Recall@1 from 0.9542 to 0.9750 and nDCG@5
from 0.9826 to 0.9958, but P95 rises from 108.37ms to 219.12ms. It is not
enabled globally.

### Hard retrieval

The v2 benchmark removes complete-title leakage. Its locked raw Recall@5 is
0.8029, below the 0.85 target. A dev-calibrated threshold correctly abstains on
75% of no-answer cases but drops Recall@5 to 0.4964, so threshold abstention is
a recorded negative result.

A new v3 holdout excludes all v2 gold products and uses seed 20260722. Its
Recall@5 is 0.6889:

| Kind | Recall@5 |
|---|---:|
| Multi-constraint | 1.0000 |
| Near-SKU multi-gold | 1.0000 |
| Attribute without title | 0.6625 |
| Alias/typo | 0.2500 |
| No-answer accuracy | 0.0000 |

The v3 result confirms that hard retrieval remains the primary limitation.

## Agent evaluation

The original 100% score is retained only as an Oracle feasibility upper bound.
Normal policies receive no hidden TaskSpec fields.

| Policy | Tasks × repeats | Success | pass^3 | Tool F1 | Compliance | Terminal state |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 60×3 | 1.000 | 1.000 | 0.944 | 1.000 | 1.000 |
| Rule Policy | 60×3 | 0.950 | 0.950 | 0.917 | 1.000 | 1.000 |
| LLMPolicy | pending | pending | pending | pending | pending | pending |

All nine Rule failures are three policy tasks repeated three times. The policy
treated ambiguous “物流/退款” words as personal-order or return requests before
checking explicit “规则/规定” language. The routing priority was fixed and a
fresh focused v3 holdout passed 30/30. Because the prior locked failures were
inspected, the same locked set will not be represented as unseen after tuning.

## Original regression

A freshly rebuilt index reproduced the NSCC regression exactly:
Recall@1=0.889, Recall@5=1.000 and MRR=0.963. Two genuine ranking regressions
were isolated:

1. a dense rank-1 pet-cleaning product lost an RRF tie to a generic vacuum;
2. explicit comparison entities were present but ranked below a fused distractor.

A small natural-language dense tie-break adjustment and explicit comparison
entity priority restore the complete local 28-query result to Recall@1=0.944,
Recall@3/5=1.000 and MRR=1.000. An NSCC confirmation run remains required.

## Audits

- The evidence builder turns the stratified 50-row set into a self-contained
  HTML panel with structured constraints, complete proposed-gold facts, Hybrid
  Top 10/20 candidates, per-constraint failures and editable adjudication. It
  remains `AI-assisted pending human adjudication` until decisions are exported.
- The real LLM job exports 40 readable trajectory rows with user request, tool
  arguments, guardrails, terminal-state diff and grader decision.
- Human/grader agreement must reach 90% before the gate can pass.

## RL decision

Current gate: `eligible=false`.

Required before next-action SFT/DPO:

- 360 real LLMPolicy trajectories;
- 40 human-audited trajectories;
- at least 90% grader/human agreement;
- 200 successful/failed preference pairs;
- base model success below 95%, otherwise training is unnecessary.

PPO, multi-step GRPO and Agent Lightning training are outside the current
delivery and are not claimed.
