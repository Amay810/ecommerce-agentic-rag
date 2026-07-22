# Directory and cleanup manifest

No files were deleted while preparing this manifest.

## Canonical source directories

| Path | Purpose | Recommendation |
|---|---|---|
| `ecommerce_rag/` | runtime source, tools, retrieval, Agent Harness | KEEP |
| `scripts/` | reproducible generators, audits and exports | KEEP |
| `nscc/` | formal PBS and model-cache helpers | KEEP |
| `tests/` | deterministic regression and safety tests | KEEP |
| `docs/` | reports and machine-readable results | KEEP |

## Canonical result files

Keep these as the primary evidence:

- `retrieval_40/1000/5000.json`;
- `retrieval_5000_constraints.json`;
- `retrieval_5000_constraints_reranker.json`;
- `retrieval_v2_locked_raw.json` and `retrieval_v2_locked.json`;
- `retrieval_v3_locked_raw.json`;
- `harness_nscc_rule_v2.json`;
- `harness_v2_oracle_locked_pass3.json`;
- `agent_rl_gate_nscc_v2.json`;
- `original_28_regression_nscc.json` and
  `original_28_regression_after_fix.json`;
- `routing_holdout_v3_rule.json`;
- `retrieval_audit_evidence_50.json` and `retrieval_audit_panel_50.html` after
  the evidence PBS completes;
- `implementation_report.md`, `final_failure_analysis.md`,
  `resume_bullets.md`.

## Reproducible large local artifacts

These are ignored by Git and may be deleted only after results are backed up:

| Path | Approx local size | Purpose | If deleted |
|---|---:|---|---|
| `ecommerce_rag/data/amazon_products_5k.jsonl` | ~9 MB | formal source corpus | regenerate/upload again |
| `ecommerce_rag/index_5000/` | ~134 MB local | formal 5k local index | rebuild embeddings |
| `ecommerce_rag/index_1000/` | ~21 MB | scale index | rebuild |
| `ecommerce_rag/index_40/` | <1 MB | scale index | rebuild |
| `ecommerce_rag/index_regression_local_rebuild/` | <1 MB | temporary regression reproduction | safe to delete after NSCC confirmation |
| `ecommerce_rag/index_timing_*/` | up to ~200 MB on NSCC | build-time measurement only | safe after stats JSON is saved |
| `logs/*.db, logs/*.sqlite` | ~15 MB currently | trajectory replay and diagnosis | archive before deleting |

## Historical/intermediate tracked reports

These are useful for experiment history but redundant for a concise public
repository. The user may move them to an archive branch or delete them after
confirming the canonical reports above:

- `docs/agent_rl_gate.json`: early permissive gate;
- `docs/harness_baseline.json`: early smoke/Oracle-like baseline;
- `docs/harness_v2_oracle.json`, `docs/harness_v2_rule.json`: single-run
  intermediates superseded by locked pass^3 reports;
- `docs/original_28_regression.json`,
  `docs/original_28_regression_v2.json`,
  `docs/original_28_regression_local_rebuild.json`: prior/reproduction copies;
- `docs/index_build_regression_40_local.json`: local timing smoke result;
  retain until the three formal NSCC timing reports are available;
- `docs/retrieval_v2_dev_raw.json`,
  `retrieval_v2_dev_after_sparse.json`,
  `retrieval_v2_dev_after_dedup.json`,
  `retrieval_v2_dev_docfusion.json`,
  `retrieval_v2_dev_final.json`: dev tuning history;
- `docs/replay_smoke.json`: replay smoke output;
- root `rerank_3way.log`, `rerank_final_v2.log`,
  `smoke_supportcase.log`: ignored local logs.
- `docs/audit_panel_ui_smoke.html`: 6 KB UI-only smoke artifact with zero
  audit cases; safe to delete after reviewing the panel shell.
- `nscc/download_agent_model.py`: no longer needed for the current NSCC setup
  because Qwen3-4B-Instruct-2507 already exists under `/scratch/.../models`;
  retain only as an optional reproducibility helper.

Recommended action: keep them until the final README and interview evidence are
accepted, then archive historical reports rather than mixing them with canonical
results.

## Tracked generated database

`ecommerce_rag/data/agent_env_v2.db` is a deterministic generated database.
It is convenient for no-key reproduction but can be regenerated with
`scripts.generate_hidden_tasks`. It may be removed from Git later if repository
size becomes a concern; do not remove it before confirming the generator and
NSCC jobs reproduce the same task set.
