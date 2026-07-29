# Agent v2 architecture

## Design goal

Agent v2 coordinates knowledge retrieval and transactional tools without allowing the language model to become the source of truth for changing business state.

## Runtime loop

1. `HarnessRunner` builds an `AgentObservation` from the user turn, prior actions, evidence ledger and session state.
2. `LLMPolicy` returns one structured action: tool call, user question, final answer or handoff.
3. The action envelope and tool arguments are validated before execution.
4. `RetailTools` reads or updates the catalog, policy corpus and order database.
5. Tool results become the next observation; the loop continues until answer or handoff.
6. The complete `Trajectory` and `GradeResult` are stored for replay and comparison.

RAG is one action path inside this loop. It supplies product and policy facts; it does not own orchestration or transactional state.

## Public contracts

- `TaskSpec`: user goal, split, expected operations and terminal-state contract;
- `AgentObservation`: information visible to the policy at one decision step;
- `ToolCall`: validated typed action request;
- `Trajectory`: turns, raw model traces, tool calls, evidence and final state;
- `GradeResult`: action, policy and terminal-state metrics.

The Oracle, Rule and LLM policies share these contracts but are reported separately. Ordinary policies cannot read hidden gold fields.
