# Implementation report

## Delivered architecture

```mermaid
flowchart LR
    U[User task] --> P[Next-action policy]
    P --> T[Typed retail tools]
    T --> C[(5k catalog index)]
    T --> O[(10k-order SQLite state)]
    T --> G[Identity, policy and confirmation guardrails]
    P --> H[Human handoff]
    T --> X[Trajectory spans]
    G --> X
    H --> X
    X --> R[Replay and terminal-state grader]
    R --> M[pass@1, pass^3, tool F1, compliance, reward]
```

`DeterministicPolicy` is the auditable baseline. A learned next-action model can
replace it through the same `AgentPolicy.act` contract without changing tools or
graders.

## Reproduced local results

- Amazon Reviews 2023 corpus: 5,000 products, two categories, 1,125 joined reviews.
- Child chunks: 365 at 40, 8,770 at 1,000 and 43,953 at 5,000 products.
- At 5,000 products Hybrid Recall@5 is 0.965 and nDCG@5 is 0.965. Price
  constraints improve them to 0.979 and 0.976; baseline P95 is 1.06 seconds.
- State benchmark: 60 tasks, three repeats, 180 evaluated trajectories.
- Baseline: task success 1.000, policy compliance 1.000, handoff P/R 1.000/1.000,
  pass@1 1.000, pass^3 1.000 and tool-call F1 0.944.
- Tool F1 is below 1 because adversarial tasks deliberately attempt a forbidden
  write so the environment guardrail can prove that it blocks the mutation.

These are deterministic environment-baseline results, not an LLM policy claim.
Retrieval scale and reranker results come only from generated benchmark JSON.

## RL decision

The gate is fail-closed. Training requires 60 deterministic tasks, 300 stored
trajectories and manually reviewed reward agreement of at least 90%. Without the
audit CSV, the project remains RL-ready and does not claim Agent RL.
