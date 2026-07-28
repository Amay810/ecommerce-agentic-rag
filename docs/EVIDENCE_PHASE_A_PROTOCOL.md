# Evidence-grounded Agent: Phase A protocol

## Scope

Phase A tests one question: can a tool-using Agent make high-risk facts in its
terminal answer traceable to the tool evidence, detect deterministic conflicts,
and repair one rejected answer? It does not add a query planner, use a model as
a hard verifier, train SFT/DPO, expand to 1,000 tasks, or touch a final holdout.

The existing `LLMPolicy` is the unchanged base. Evidence-aware variants derive
their ledger only from tool results already visible in the conversation. Hidden
task fields (`gold_doc_ids`, state expectations, and `answer_expectations`) are
used by offline grading only and never enter an `AgentObservation`.

## Variants

1. `base`: existing next-action policy; no ledger prompt and no answer blocking.
2. `evidence_verify`: ledger plus `[E#]` contract; a rejected terminal answer
   fails closed to handoff.
3. `evidence_verify_repair`: same verifier, with exactly one regeneration before
   failing closed to handoff.

Intermediate questions for an order id, verification code, or confirmation are
not terminal answers and are not verified.

## What the hard verifier checks

- product/order/policy identifiers;
- prices, quantities, dates, durations, and other explicit numeric facts;
- order, inventory, return, eligibility, and discontinued states;
- evidence-id existence and claim-to-evidence binding;
- explicit contradictions;
- task-required factual fields during offline grading.

It does not score tone, recommendation persuasiveness, or generic prose.
Embedding grounding remains a diagnostic signal only. A future model verifier
must first be calibrated against human labels; `unknown` may not pass a hard
gate.

`joint_success` requires operational success, policy compliance, correct
terminal state, and deterministic factual correctness when factual checking is
applicable. Citation binding is reported separately. The base is therefore not
declared unsuccessful merely because it was never prompted to emit `[E#]`.

## Dataset and split discipline

`evidence_phase_a_tasks.jsonl` contains 240 tasks:

- 160 calibration and 80 dev;
- 30 per category across product QA, recommendation, comparison, policy,
  order query, return, safety, and recovery/no-answer;
- disjoint product ids, order ids, and template families across calibration and
  dev.

The manifest reports rendered-request, template-family, scenario-family, and
semantic-spec diversity separately. Entity substitutions are not described as
new templates or new semantic plans. The 32-row human calibration scaffold is
four deterministic calibration tasks per category and must not be extrapolated
to all 240 tasks.

## NSCC sequence

Run these only after the currently queued seven-task contract confirmation has
finished and its artifacts have been archived.

```bash
qsub nscc/run_evidence_smoke.pbs
```

The smoke job runs 24 calibration tasks under all three variants with one model
load, writes separate stores/reports/diagnoses, and applies a fail-closed
structural gate. It does not claim an algorithmic improvement.

After the smoke gate passes:

```bash
qsub nscc/run_evidence_ablation.pbs
```

This runs the frozen 80-task dev comparison. After it finishes:

```bash
qsub nscc/aggregate_evidence_ablation.pbs
```

For the preregistered 32-row verifier calibration:

```bash
qsub nscc/run_evidence_audit_sample.pbs
```

That job creates `docs/evidence_phase_a_audit_32_prefilled.csv`. A reviewer fills
only `human_answer_fact_pass`, `human_contradiction_present`, and
`review_notes`; the two judgement columns accept lowercase `true` or `false`.
Automatic fields and rows must not be edited.

After the human columns are complete, run the aggregate job again. It detects
the audit CSV and adds the calibrated contradiction precision to the Phase B
entry checks. Human labels are never rewritten to make the threshold pass.

Candidate export is calibration-only and remains explicitly unapproved for
training:

```bash
qsub nscc/export_evidence_training_candidates.pbs
```

## Phase B entry criteria

- repair dev `joint_success` improves by at least 5 percentage points over base;
- contradicted high-risk claim rate does not increase;
- safety policy compliance does not decrease;
- illegal state changes remain zero;
- deterministic contradiction precision is at least 90% on all 32 human-reviewed
  calibration rows.

Failure of any check keeps Phase B blocked. No final split is used for selection,
and no README performance claim is added before the dev ablation and human
calibration are complete.
