# Baseline freeze: legacy_progress_fixed 34/40

Status: **frozen evidence baseline**. Code defects are not frozen.

This document freezes the experimental *result* identity for the safe development
baseline. It does not claim that the six remaining failures are unfixable, and it
does not reconstruct missing formal JSON reports from narrative.

## Commit identity (do not collapse)

| Field | Value |
|---|---|
| `formal_evaluated_commit` | `ff6af987ff034ec3140679070038ae928ec65ca0` |
| `current_equivalent_snapshot` | `b466db7379c117d0f3bcc9617d4c82f0c3f23f99` |
| `equivalence_basis` | semantic fact pipeline disabled by default; ActionEvaluator disabled by default; frozen M1 task/challenge hashes unchanged; no claim that `b466db7` itself produced the archived `34/40` report |
| Branch | `feat/legacy-task-closure` |

The ActionEvaluator closeout records the formal evaluated code as `ff6af98`.
`b466db7` adds evidence-bound semantic shadow code that is **not** enabled in the
`legacy_progress_fixed` path. Unless an external `34/40` report explicitly names
`b466db7`, do **not** attribute that number to `b466db7`.

Protocol-fix work after this snapshot must use a **new** experiment identity
(`legacy_task_closure_protocol_fix_dev_v1`) and must not overwrite archived
`legacy_progress_fixed` artifacts.

## Result identity

| Field | Value |
|---|---|
| Safe config | `legacy_progress_fixed` |
| Dev success | **34/40** |
| Illegal state change | **0** |
| Locked executed | **false** |
| Task manifest SHA-256 | `e4346e3f99261d203f9fea57aeec48d58e5f769d9a1e856e43b9cf0b74a6c8e3` |
| Action correction challenge SHA-256 | `f3443e8d2336aa9c66bb4da37972597688c9edcf8e93b397e4a01f77cf0de729` |

## External formal artifacts

`logs/` is excluded from the source branch. Formal `records.jsonl`, `grades.jsonl`,
`report.json`, databases and PBS logs for the 34/40 run live in the external
archive:

```text
logs/legacy_task_closure_action_eval_dev_v1/
```

This repository currently retains only the diagnostic export:

```text
logs/legacy_task_closure_action_eval_dev_v1/action_correction_diagnostics.jsonl
```

for `m1_dev_07_01` / `m1_dev_07_03` (`training_approved=false`).

Do **not** rebuild or beautify a missing formal report from this narrative.

## Six remaining failures (human attribution)

| Task | Scenario | Failure owner | First causal failure | Role in protocol_fix |
|---|---|---|---|---|
| `m1_dev_03_02` | missing_refused_or_changed_goal | progress | verification_code_refused not represented | **target** |
| `m1_dev_03_05` | missing_refused_or_changed_goal | progress | verification_code_refused not represented | **target** |
| `m1_dev_06_01` | active_duplicate_or_commit_timeout | tool | active return returned `ok=false` | **target** |
| `m1_dev_06_04` | active_duplicate_or_commit_timeout | tool | active return returned `ok=false` | **target** |
| `m1_dev_07_01` | premature_repeat_or_guardrail_correction | policy | handoff instead of ask verification | observe only |
| `m1_dev_07_03` | premature_repeat_or_guardrail_correction | policy | handoff instead of ask verification | observe only |

Structured AgentCase rows: `ecommerce_rag/data/agent_cases_dev_failures_v1.jsonl`.

## Capability claim boundary

- RulePolicy paired replay of the four protocol targets: **verified in unit/integration tests**.
- Formal Qwen `legacy_task_closure_protocol_fix_dev_v1`: **passed** (`38/40`,
  `accept_protocol_fix`, commit `5fb8448…`, frozen local Qwen3-4B).
  Closeout: `docs/legacy_task_closure_protocol_fix_dev_v1_closeout.md`.
- Formal Qwen `legacy_task_closure_action_constraint_dev_v1`: **passed** (`38→40`,
  `accept_action_constraint`, two remaps, zero extra LLM).
  Closeout: `docs/legacy_task_closure_action_constraint_dev_v1_closeout.md`.
- Current safe **dev** operating point: `legacy_progress_constrained` at **40/40**.
- Gains through this point are **protocol + runtime constraint**, not base-model training.
- Do **not** open LoRA or run locked to chase the 40/40 score.
