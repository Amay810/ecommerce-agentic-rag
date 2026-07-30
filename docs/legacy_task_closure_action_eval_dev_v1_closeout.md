# Legacy Task Closure ActionEvaluator dev v1 closeout

## Decision

Status: **negative**. The frozen ActionEvaluator v1 gate did not pass.
The locked split was not run and CompletionEvaluator was not admitted.

This closeout does not rewrite the formal report or retroactively change its
gate. Raw artifacts remain in the external formal run directory:

```text
logs/legacy_task_closure_action_eval_dev_v1/
```

The evaluated code commit was
`ff6af987ff034ec3140679070038ae928ec65ca0`. Because `logs/` is excluded from
the source branch, `records.jsonl`, `grades.jsonl`, `report.json`, databases and
the PBS log must remain in the external/artifact archive. No JSON report is
reconstructed from this narrative summary.

## Main paired result

The 40-task main comparison was:

| Configuration | Success | Illegal state changes |
|---|---:|---:|
| `legacy_progress_fixed` | 34/40 | 0 |
| `legacy_progress_action_eval` | 34/40 | 0 |

All 40 tasks were paired. ActionEvaluator produced two semantic corrections,
on `m1_dev_07_01` and `m1_dev_07_03`, and recovered neither task.

## Failure mechanism

In both tasks the first action was an inappropriate handoff while progress
required identity verification. The evaluator correctly returned:

```text
pending: identity_verification
allowed_next_actions: ask_user:verification_code
```

Qwen then replaced the handoff with a terminal answer claiming that the return
had already succeeded. `LegacyActionEvaluator` v1 checked tool calls, handoff
and user-input finals, but did not check a terminal
`requires_user_response=false` final against progress. The second action was
therefore recorded as accepted even though it was premature and unsupported.

The reduction in inappropriate handoff is not a capability gain: the failure
changed type from handoff to premature final, while task success remained
unchanged.

## Challenge boundary

Four correction-challenge failures were caused by free-language return reasons
not being extracted by the regex-based reducer. They are evidence about fact
extraction coverage, not valid evidence that action correction failed. The
challenge expressions are retained for SemanticFactExtractor development and
validation; they are not used to rescue or reinterpret the v1 gate.

## Disposition

- Keep `legacy_progress_fixed` as the current 34/40 safe development baseline.
- Keep `HarnessRunner.action_evaluator` disabled by default.
- Do not run ActionEvaluator v2 merely to chase the failed gate.
- Do not enter CompletionEvaluator or locked evaluation.
- Stop extending regexes for free-language reasons, refusal, goal changes or
  compound intent.
- Move those fields to an evidence-bound semantic extraction pipeline.
- Preserve the two main failures as diagnostic correction exemplars only; they
  must never enter training because their lineage is the formal dev split. Use
  them to define the schema for independent train-split examples in which an
  inappropriate handoff plus identity-verification progress must map to a
  verification-code question, never to a terminal success claim.

If ActionEvaluator is reconsidered after the extraction and policy-data work,
every action type, including terminal final, must be checked against a unified
progress allowlist. That future change is not part of v1 and cannot alter this
closeout.
