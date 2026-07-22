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
- `nscc/run_llm_policy_v2.pbs`: 360 real local-LLM trajectories and audit export.

## Requires human judgement

- Review `docs/retrieval_audit_panel_50.html` and export the completed decisions;
  until then it is not a human-labelled benchmark.
- After the LLM job, review all 40 rows in `docs/trajectory_audit_40.csv` and
  rerun the fail-closed RL gate.

## RL decision

No Agent RL is claimed. Next-action SFT/DPO is allowed only if every gate check
passes, including 360 real LLM trajectories, 40 audited rows, at least 90%
grader agreement and at least 200 valid preference pairs.
