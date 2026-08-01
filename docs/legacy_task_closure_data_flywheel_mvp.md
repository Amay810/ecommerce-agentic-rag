# Data Flywheel MVP (v1.1)

```text
status: engineering_loop_complete
policy_gain: unproven
default: off
```

Frozen at commit lineage including `47459f9` + attribution hardening + probe
scoring freeze `0bd58c5`.

**Policy capability gain from Memory:** tested by `memory_policy_probe_v1` →
**negative_or_inconclusive** (`docs/memory_policy_probe_v1_closeout.md`).
Runtime Memory remains default off.

```text
Agent run → Trajectory → decision-level AgentCase → admission → Case Memory
→ similar-state retrieval → memory_advice prior → Policy
→ Action Constraint → candidate writeback
```

## Components

| Step | Module | Notes |
|---|---|---|
| Store | `agent_case_store.py` | SQLite source of truth |
| Admission | `admit_for_memory` | candidates loose; **approved** strict |
| Retrieval | `build_memory_advice` | SQL only; never expands allowlist |
| Prior | harness `memory_advice` | fail-open |
| Constraint | `apply_action_constraint` | final legality |
| Writeback | `candidates_from_trajectory` | **one case per decision step** |
| Demo | `python -m scripts.demo_agent_case_flywheel` | seed → query → list |

## Attribution (v1.1)

Memory spans record, in order:

```text
raw_policy_action
memory_preferred_actions
policy_followed_advice   # computed BEFORE constraint
constrained_action
constraint_remapped
executed_action
```

`advice_used` is an alias of `policy_followed_advice`. Constraint remaps must not
be counted as Memory adoption.

Decision cases store `causal_credit`. Remapped steps are never marked
`success=true` for Memory preferred experience.

## Approval gate (strict)

Approved cases require provenance hash, attribution source, verifiable terminal
state, `approved_by` / `approved_at` / `approval_reason`, and
`paired_replay_result`. Failed cases cannot use `failure_owner=none`. Remapped
raw policy actions cannot be approved as policy success memory.

## Credential hygiene

Redaction/scanning skips system fields (`created_at`, hashes, ids). Six-digit
matches after `.` (ISO fractional seconds) are ignored.

## Flags

```text
ERAG_AGENT_CASE_DB
ERAG_AGENT_CASE_MEMORY=1
ERAG_AGENT_CASE_WRITEBACK=1
```

## Demo

```powershell
python -m scripts.demo_agent_case_flywheel --db logs/demo_cases.db seed-train-identity
python -m scripts.demo_agent_case_flywheel --db logs/demo_cases.db query-identity
python -m scripts.demo_agent_case_flywheel --db logs/demo_cases.db list
```

## Claim boundary

- Storage / isolation / retrieval / inject / writeback: **accepted as MVP**.
- “Memory improved Policy”: **not claimed** until Memory off/on shows
  `policy_followed_advice` gains without constraint credit laundering.
