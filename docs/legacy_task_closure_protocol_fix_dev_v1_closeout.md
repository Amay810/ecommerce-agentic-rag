# Legacy Task Closure Protocol Fix dev v1 closeout

## Decision

Status: **positive**. The frozen ProtocolFix gate passed:
`accept_protocol_fix`.

Locked was not run. ActionEvaluator and SemanticFactExtractor stayed disabled.

Raw artifacts remain in the external formal run directory:

```text
logs/legacy_task_closure_protocol_fix_dev_v1/
```

Because `logs/` is excluded from the source branch, formal JSON, databases and
PBS logs must remain in the external/artifact archive. No report is reconstructed
here beyond the narrative identity below.

## Main result

| Identity | Value |
|---|---|
| Protocol | `legacy_task_closure_protocol_fix_dev_v1` |
| Config | `legacy_task_closure_protocol_fix` |
| Code commit | `5fb8448c59f0f93f3477c944a2fed16cf02f0e39` |
| Formal baseline commit (34/40 archive) | `ff6af987ff034ec3140679070038ae928ec65ca0` |
| Dev success | **38/40** |
| Illegal / protocol / duplicate | **0 / 0 / 0** |
| Locked executed | **false** |
| Model | frozen local `Qwen3-4B-Instruct-2507` |

## What was recovered

| Task | Owner | Intervention |
|---|---|---|
| `m1_dev_03_02`, `m1_dev_03_05` | progress/protocol | `verification_code_refused` event + handoff-only allowlist |
| `m1_dev_06_01`, `m1_dev_06_04` | tool/grader | active return `ok=true, changed=false, idempotent_replay=true` |

## What remained

| Task | Owner | Note |
|---|---|---|
| `m1_dev_07_01`, `m1_dev_07_03` | policy | Still inappropriate handoff; observe-only in this protocol |

## Claim boundary

Recovered tasks are **system protocol / tool-contract** gains, not base-model
learning. The remaining two failures were deferred to
`legacy_task_closure_action_constraint_dev_v1`.

## Disposition

- Treat `38/40` as the accepted protocol-fixed baseline for the constraint pair.
- Do not overwrite archived `legacy_progress_fixed` `34/40` artifacts.
- Do not admit LoRA from these formal `dev` recoveries.
