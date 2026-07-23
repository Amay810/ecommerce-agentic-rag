# Completion status against the implementation plan

Status date: 2026-07-22.

## Completed locally

1. Documentation: README, scale table, implementation report, failure analysis,
   resume bullets and cleanup manifest have been updated.
2. Original 28 regression: exact compound-entity ordering and dense-score tie
   handling were fixed. The local result improved from Recall@1 0.889 to 0.944;
   Recall@3/5 are both 1.0.
3. RulePolicy routing: explicit policy questions now take priority over ambiguous
   order/refund keywords. A fresh 30-query v3 routing holdout scored 1.0.
4. Retrieval audit: a stratified 50-row assisted sheet was generated with 10
   budget, 10 multi-constraint, 10 typo/alias, 10 near-SKU and 10 no-answer
   cases. Proposed product IDs were checked against the corpus; human sign-off
   remains intentionally blank.
5. Index timing: the loader now reports model-load, embedding, persistence,
   total build time and index size. A local 40-product smoke measurement passed.
6. Fresh v3 retrieval set: 150 queries use product IDs excluded from v2. The
   honest local baseline has Recall@5 0.6889; typo/alias queries are the main
   weakness and no-answer calibration remains unresolved.
7. NSCC jobs: paths, environment and model cache settings were corrected. Jobs
   now exist for formal index timing, v3 validation and 360 real LLMPolicy
   trajectories.

## Requires NSCC execution

- `nscc/measure_index_builds.pbs`: formal 40/1,000/5,000 build timings.
- `nscc/run_v3_validation.pbs`: FAISS confirmation of the v3 retrieval and
  routing results.
- `nscc/run_regression_only.pbs`: formal NSCC confirmation of the 28-query fix.
- `nscc/build_retrieval_audit_evidence.pbs`: Hybrid Top 10/20 candidates,
  constraint checks and a self-contained HTML adjudication panel.
## Executed, but invalid

- `nscc/run_llm_policy_v2.pbs`: executed. Produced 360 trajectories and the
  40-row audit template, but the batch is an **invalid integration run** —
  `model_action_parse_failure` on 360/360, every step degraded to
  `escalate_to_human`. Outputs are retained and marked via `run_validity` in
  `docs/harness_v2_llm_*_pass3.json`. Blocked on: persist raw model output and
  parse attempts in the trace, then a 5–10 task smoke run, then a full re-run.

## Requires human judgement

- Review `docs/retrieval_audit_panel_50.html` and export the completed decisions;
  until then it is not a human-labelled benchmark. Known blocker: the model-code
  regex does not extract `ZX-xxx` from Chinese text, so the no-answer candidate
  constraint checks are empty and that stratum cannot be adjudicated yet.
- Review all 40 rows in `docs/trajectory_audit_40.csv` and rerun the fail-closed
  RL gate. Currently **0 of 40 rows carry a human verdict**; the gate now reports
  `human_audit_reviewed: 0`, `human_audit_status: "not_started"` and
  `human_reward_agreement: null` instead of the earlier misleading `0.0`.
  A row counts only when both `human_success` and `human_policy_compliant` are
  strict booleans; anything else is tallied as `human_audit_malformed_rows`.
  Note this is a *trajectory* audit and feeds `human_reward_agreement` only; the
  separate `docs/retrieval_human_audit_50.csv` backs the retrieval gold and does
  not contribute to the RL gate.

## RL decision

No Agent RL is claimed. Next-action SFT/DPO is allowed only if every gate check
passes, including 360 real LLM trajectories, 40 audited rows, at least 90%
grader agreement and at least 200 valid preference pairs.
