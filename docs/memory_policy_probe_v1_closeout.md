# memory_policy_probe_v1 closeout

## Decision

Status: **negative_or_inconclusive**.

```text
protocol: memory_policy_probe_v1
verdict: negative_or_inconclusive
policy_memory_gain: negative_or_inconclusive   # = not proven / not admitted
illegal_state_change: 0
```

**Only admissible claim:**

> On this frozen 24-pair independent probe, SQL Memory advice **cannot** be
> claimed to increase raw Policy adherence to preferred legal actions.

This closeout does **not** reopen Action Constraint `40/40`, does **not** rewrite
Data Flywheel MVP v1.1 engineering-loop acceptance, and does **not** authorize
semantic retrieval or LoRA.

Raw artifacts remain in the external formal run directory (gitignored `logs/`):

```text
logs/memory_policy_probe_v1/
  report.json
  pairs.jsonl
  trajectories/{memory_off,memory_on}/
```

No JSON report is reconstructed from this narrative. Numbers below are taken
from the operator-accepted formal run against protocol commit lineage
`0bd58c5` (scoring/guard freeze) as executed on the clean worktree.

## Claim boundary (mandatory)

| Layer | Status | Claim |
|---|---|---|
| Protocol / tool / Action Constraint | accepted | runtime contract, not base-model gain |
| Data Flywheel MVP v1.1 | engineering_loop_complete | store / admit / retrieve / inject / writeback |
| Memory → Policy capability | **not proven** | this probe |

```text
experiment_succeeded: true          # 24×2 completed; auto-verdict emitted
hypothesis_succeeded: false         # Memory improves raw Policy — rejected by gate
```

Do not conflate “probe failed to run” with “hypothesis failed the gate.”

## Main paired result

| Metric | Value |
|---|---:|
| Pairs | 24 |
| Retrieval coverage | **12/24** (gate requires 24/24) |
| Off-arm raw Policy errors | 12 |
| `repaired_by_memory` | **0** |
| `regressed_by_memory` | 0 |
| Constraint remap off / on | 0 / 0 |
| Terminal success off / on | 0 / 0 |
| Illegal state change | 0 |

Automatic failure reason recorded in `report.json`:

```text
retrieval coverage 12/24 < 24/24
```

Even if coverage were waived: **repair count remains 0** on the 12 retrieved
pairs — Memory did not causally move raw Policy onto scoring preferred actions
with `policy_followed_advice=true`.

## Mechanism (why the gate closed)

Preregistered Positive required **all** of:

- retrieval **24/24**
- ≥4 off raw errors; repair ≥ half; no regression
- remap down; terminal success not down; illegal = 0
- repair bound to `retrieval_matched ∧ policy_followed_advice`

Observed structure:

1. **`missing_return_reason` / `awaiting_confirmation` (12 tasks)**  
   Scripted tool prefixes leave probe-step progress signatures that do not
   match curated seed signatures → SQL miss → coverage stuck at 12/24.

2. **`missing_order_id` / `missing_verification` (12 tasks)**  
   Retrieval can hit, but `repaired_by_memory=0` (off already near-correct
   and/or on lacked `policy_followed_advice`).

3. **Constraint remap 0/0**  
   Constraint did not “launder” Memory credit; also did not create a
   Constraint-rescued success narrative for Memory.

Train priors remain labeled:

```text
source_kind=curated_contract_seed
validation_type=deterministic_contract_check
experience_case=false
```

They were valid **contract seeds for this probe**, not flywheel experience cases.

## Disposition (Negative path — frozen)

```text
AgentCase Store:        retain (audit / attribution asset)
Runtime Memory inject:  default OFF  (ERAG_AGENT_CASE_MEMORY=0)
Semantic retrieval:     do not start
Action-only LoRA:       do not start
memory_policy_probe_v1: CLOSED — do not patch and re-run to flip verdict
```

Re-opening Memory→Policy requires a **new protocol id** with a new freeze
(e.g. first-step-reachable states only, or progress-aligned seeds *before*
preregistration). Tip-SHA churn or post-hoc coverage relaxation on this id is
not an admissible rescue.

## What this does not say

- It does not say the Agent stack is broken (protocol/constraint already held).
- It does not say Case Memory storage is useless (audit/writeback remain).
- It does not say every future Memory design must fail — only that **this**
  frozen SQL prior under **this** gate did not prove Policy gain.

## Next work (outside this protocol)

See README Task Closure section. Priority shifts away from Memory prior tuning
toward either:

1. shipping/operating the accepted runtime stack (progress + constraint) with
   Memory off; or
2. a separately named probe if and only if the research question is redesigned
   *before* any new formal run.
