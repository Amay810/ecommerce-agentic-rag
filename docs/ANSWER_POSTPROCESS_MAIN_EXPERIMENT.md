# Terminal-only grounding main experiment

## Final status

This experiment is closed as `negative_or_inconclusive`. The blinded 40-pair
review found identical fact-pass rates for base and terminal-grounded answers
(34/40, 85.0%), with a paired difference of 0 and a 10,000-sample paired-bootstrap
95% interval of [-7.5 pp, +7.5 pp]. Completeness changed from 80.0% to 82.5%, and
the total contradiction rate remained 15.0%. Both immutability gates passed.

The reviewer was a dedicated Codex context restricted to the blinded review file;
this is not reported as external human annotation. The full result and provenance
are recorded in `docs/answer_postprocess_blind_audit_v1_closeout.md`. Do not run
v3, retune this method, or start the conditional canonical-product, external
benchmark, verifier, SFT, or DPO work.

## Frozen scope

The Qwen3-4B model, terminal-grounding prompt, deterministic decoding settings,
base trajectory store, 12-task smoke manifest, and 40-task blinded-audit selection
are frozen before any new grounded output is inspected. The verifier is diagnostic
only and is not an admission gate, answer selector, repair mechanism, or handoff
trigger.

`verifier_challenge_locked_v2` is closed as a constructed regression fixture. Do
not confirm its prefilled labels, run its formal evaluator, or create locked_v3.

## Run order

The v1 dev run is archived as `aborted_generation_truncated`: its shadow sidecar
completed, but no complete terminal-grounded sidecar was published. It is not an
algorithmic result and must not be rerun, deleted, or used for human review. V2
changes only the real generation ceiling from 512 to 1024 tokens.

1. Submit `nscc/run_answer_postprocess_smoke_v2.pbs` once. Both structural gates
   must pass, the report must record `max_new_tokens=1024`, and there must be no
   generation error or truncation. The prompt, model, decoding, tasks, and blind
   audit selection remain frozen.
2. Submit `nscc/run_answer_postprocess_dev_v2.pbs` once. It refuses model, token,
   prompt-hash, generation-config, or verifier drift from smoke and creates paired
   shadow and terminal-grounded sidecars for all 80 frozen trajectories.
3. Create the blinded package without opening the grounded sidecar:

```powershell
python -m scripts.prepare_answer_postprocess_audit `
  --selection-manifest docs/answer_postprocess_blind_audit_v1_manifest.json `
  --tasks ecommerce_rag/data/evidence_phase_a_tasks_v2.jsonl `
  --base-store logs/evidence_phase_a_dev_v3_base.sqlite `
  --shadow logs/answer_postprocess_dev_v2_shadow.jsonl `
  --terminal-grounded logs/answer_postprocess_dev_v2_terminal_grounded.jsonl `
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
  --shadow logs/answer_postprocess_dev_v2_shadow.jsonl `
  --terminal-grounded logs/answer_postprocess_dev_v2_terminal_grounded.jsonl `
  --shadow-gate docs/answer_postprocess_dev_v2_shadow_gate.json `
  --terminal-grounded-gate docs/answer_postprocess_dev_v2_terminal_grounded_gate.json `
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

Any v2 truncation aborts the experiment. Do not rerun v2, increase the ceiling to
2048, create v3, alter the prompt, or replace a frozen audit task.
