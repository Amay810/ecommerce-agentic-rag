# Terminal-only grounding main experiment

## Frozen scope

The Qwen3-4B model, terminal-grounding prompt, deterministic decoding settings,
base trajectory store, 12-task smoke manifest, and 40-task blinded-audit selection
are frozen before any new grounded output is inspected. The verifier is diagnostic
only and is not an admission gate, answer selector, repair mechanism, or handoff
trigger.

`verifier_challenge_locked_v2` is closed as a constructed regression fixture. Do
not confirm its prefilled labels, run its formal evaluator, or create locked_v3.

## Run order

1. Submit `nscc/run_answer_postprocess_smoke_v1.pbs` once. Both structural gates
   must pass. Inspecting the smoke may stop the experiment for generation or
   structural failures, but must not change the frozen prompt or configuration.
2. Submit `nscc/run_answer_postprocess_dev_v1.pbs` once. It refuses configuration
   drift from smoke and creates paired shadow and terminal-grounded sidecars for
   all 80 frozen trajectories.
3. Create the blinded package without opening the grounded sidecar:

```powershell
python -m scripts.prepare_answer_postprocess_audit `
  --selection-manifest docs/answer_postprocess_blind_audit_v1_manifest.json `
  --tasks ecommerce_rag/data/evidence_phase_a_tasks_v2.jsonl `
  --base-store logs/evidence_phase_a_dev_v3_base.sqlite `
  --shadow logs/answer_postprocess_dev_v1_shadow.jsonl `
  --terminal-grounded logs/answer_postprocess_dev_v1_terminal_grounded.jsonl `
  --output-dir docs/answer_postprocess_blind_audit_v1
```

Give the reviewer only `answer_postprocess_blind_audit_v1_review.jsonl`. Keep the
mapping and package manifest out of the review view. For every row, fill:

- `fact_pass`: `true`, `false`, or `unclear`;
- `answer_complete`: `true`, `false`, or `unclear`;
- `contradiction_present`: `true`, `false`, or `unclear`;
- `review_notes`: text, which may be empty.

Do not edit any other field or change row order/count.

4. Aggregate only after all 80 blinded answers are reviewed:

```powershell
python -m scripts.aggregate_answer_postprocess_audit `
  --review docs/answer_postprocess_blind_audit_v1/answer_postprocess_blind_audit_v1_review.jsonl `
  --mapping docs/answer_postprocess_blind_audit_v1/answer_postprocess_blind_audit_v1_mapping.jsonl `
  --package-manifest docs/answer_postprocess_blind_audit_v1/answer_postprocess_blind_audit_v1_package_manifest.json `
  --shadow logs/answer_postprocess_dev_v1_shadow.jsonl `
  --terminal-grounded logs/answer_postprocess_dev_v1_terminal_grounded.jsonl `
  --shadow-gate docs/answer_postprocess_dev_v1_shadow_gate.json `
  --terminal-grounded-gate docs/answer_postprocess_dev_v1_terminal_grounded_gate.json `
  --output docs/answer_postprocess_blind_audit_v1_aggregate.json
```

## Decision rule

A positive result requires a positive paired fact-pass difference whose 10,000
sample paired-bootstrap 95% lower bound is above zero, no more than a five-point
completeness drop, no increase in human-labelled contradictions, and both
trajectory-immutability gates passing. Verifier diagnostics are reported in a
separate block and never enter this decision.

If the decision is negative or inconclusive, close the experiment without more
verifier, canonical-product, external-benchmark, SFT, or DPO work. Only a positive
result permits the later 120-product canonical evidence phase and one separately
reported external benchmark.
