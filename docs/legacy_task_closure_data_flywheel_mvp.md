# Data Flywheel MVP

Close the loop without expanding the eval suite:

```text
Agent run → Trajectory → AgentCase → admission → Case Memory
→ similar-state retrieval → memory_advice prior → Policy
→ Action Constraint → candidate writeback
```

## Components

| Step | Module | Notes |
|---|---|---|
| 6A Store | `ecommerce_rag/agent_case_store.py` | SQLite only; no Chroma |
| 6B Admission | `admit_for_memory` / `sanitize_case` | `dev`/`locked` → quarantined; credentials / hidden grader fields rejected |
| 6C Retrieval | `query_memory_candidates` + `build_memory_advice` | workflow + pending/guard + allowed intersection |
| 6D Prior | `HarnessRunner` session `memory_advice` | fail-open; never expands allowlist |
| 6E Constraint | existing `apply_action_constraint` | final legality |
| 6F Writeback | `candidate_from_trajectory` | status `candidate` only unless explicitly approved |
| 6G Demo | `scripts/demo_agent_case_flywheel.py` + `tests/test_agent_case_flywheel.py` | one closed loop |

## Safety rails

- Memory off → previous Agent behavior
- Memory failure does not block the task
- Memory cannot authorize write tools
- Formal `dev`/`locked` never `memory_approved`
- Credentials never enter Memory
- Illegal state change remains a hard reject on success cases

## Flags

```text
ERAG_AGENT_CASE_DB          # default: logs/agent_cases.db
ERAG_AGENT_CASE_MEMORY=1    # inject memory_advice
ERAG_AGENT_CASE_WRITEBACK=1 # persist candidates after each run
```

## Done when

One approved train case is retrieved on a similar task, Policy observation receives
an action prior inside `allowed_next_actions`, Constraint still adjudicates, and
the run writes a new candidate case. No large-scale score chase required.
