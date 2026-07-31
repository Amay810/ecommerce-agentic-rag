# Legacy Task Closure Action Constraint dev v1 closeout

## Decision

Status: **positive**. The frozen ActionConstraint gate passed:
`accept_action_constraint`.

Locked was not run. ActionEvaluator, SemanticFactExtractor, CompletionEvaluator
and LoRA were not admitted. This closeout does not rewrite earlier negative
ActionEvaluator evidence.

Raw artifacts remain in the external formal run directory:

```text
logs/legacy_task_closure_action_constraint_dev_v1/
```

Because `logs/` is excluded from the source branch, `records.jsonl`,
`grades.jsonl`, `report.json`, databases and the PBS log must remain in the
external/artifact archive. No JSON report is reconstructed from this narrative.

## Claim boundary (mandatory)

```text
runtime_system_constraint_not_base_model
```

The gain is a **runtime action contract** over `TaskProgress` allowlists.
It must not be described as Qwen3-4B base-model improvement, SFT, DPO, or Agent RL.

## Main paired result

| Configuration | Success | Illegal state changes |
|---|---:|---:|
| `legacy_progress_protocol_fixed` | 38/40 | 0 |
| `legacy_progress_constrained` | 40/40 | 0 |

Paired gain: **+2**. Only the preregistered observe-policy tasks moved.

| Check | Result |
|---|---|
| All gate checks | `true` |
| `observe_policy_recovered` | `m1_dev_07_01`, `m1_dev_07_03` |
| `observe_premature_final` | `[]` |
| `constraint_remap_count` | `2` |
| Extra constraint LLM calls | `0` |
| ActionEvaluator correction spans | `0` |
| Locked executed | `false` |

## Mechanism (contrast with ActionEvaluator v1)

ActionEvaluator v1 rejected inappropriate handoffs on the same two tasks, then
allowed a second model action that claimed terminal return success. Success
stayed 34/40: **error migration**, not recovery.

ActionConstraint v1 remapped each illegal first action **once** to the preferred
allowlist action (`ask_user:verification_code`) with **no second LLM call**.
Both tasks then completed under the ordinary progress/tool loop: **true task
completion**, not metric transfer.

## One-line result

> 在协议修复基线 38/40 上，动态动作约束以 2 次 remap、0 额外 LLM 调用恢复
> `m1_dev_07_01/03`，达到 40/40；增益归属运行时契约，不归属基座模型。

## Disposition

- Keep `legacy_progress_constrained` as the current safe **dev** operating point
  for return-resolution Task Closure (`40/40`, illegal state change `0`).
- Keep `HarnessRunner.action_evaluator` disabled by default.
- Keep `enforce_action_constraint` as the recommended runtime for formal
  return-resolution configs; do not conflate it with ActionEvaluator correction.
- Do **not** open LoRA, DPO, GRPO, or preference training to chase this score.
- Do **not** run locked merely because dev reached 40/40.
- Do **not** treat formal `dev` trajectories (`07_01/03` remaps included) as
  training data (`training_approved` remains false).
- Preserve ActionEvaluator v1 negative closeout unchanged as contrast evidence.
- Next engineering work, if any, is independent: semantic-dev validation,
  AgentCase memory accumulation, interview/GitHub packaging — not score chasing.

## Lineage

- Protocol-fix baseline: `legacy_task_closure_protocol_fix_dev_v1` (`38/40`)
- Constraint protocol: `legacy_task_closure_action_constraint_dev_v1`
- Task manifest SHA-256:
  `e4346e3f99261d203f9fea57aeec48d58e5f769d9a1e856e43b9cf0b74a6c8e3`
- Model: frozen local `Qwen3-4B-Instruct-2507`
