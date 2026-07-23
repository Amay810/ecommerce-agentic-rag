# Human audit guide

## A. Retrieval gold audit: 50 rows

After `nscc/build_retrieval_audit_evidence.pbs` finishes, download and open
`docs/retrieval_audit_panel_50.html` in a browser. It is self-contained and does
not require a web server.

Each case displays:

- parsed budget, brand, category, keyword or model-code constraints;
- complete facts for the proposed gold product;
- Hybrid Top 10 candidates, or Top 20 for near-SKU/no-answer;
- retrieval scores and dense/BM25 ranks;
- per-candidate constraint pass/fail results and failure reasons;
- the proposed gold rank and other candidates passing the programmatic checks.

The sheet is stratified into five `audit_scope` groups with 10 rows each:
budget, multi-constraint, alias/typo, near-SKU and no-answer.

For each case select `confirm`, `modify` or `uncertain`, select the final gold
checkboxes, set answerability, reviewer and notes. The panel saves progress in
browser local storage. Click `导出审核 CSV` when all 50 cases are complete.

The exported fields correspond to:

- `human_gold_doc_ids`: final IDs separated by `|`;
- `is_answerable`: true/false;
- `reviewer`: reviewer name or initials;
- `review_notes`: especially note alternative matching SKUs;
- decision: `confirm`, `modify` or `uncertain`.

Questions to answer:

1. Does the proposed product really satisfy every query constraint?
2. Is another product equally correct and missing from gold?
3. Is a typo clue close enough to identify the proposed title?
4. For no-answer, do all Top 20 plausible candidates fail at least one hard
   condition, particularly the required model code?

For `near_sku` rows, multiple gold IDs may be valid. Do not force a single ID.
For `no_answer` rows, keep `human_gold_doc_ids` empty only after searching the
corpus and set `is_answerable=false`.

The panel reduces review work to adjudicating displayed evidence; it does not
pretend to prove corpus-wide absence. Any ambiguous no-answer case should be
marked `uncertain`, not forced to `confirm`. Until the exported decisions are
complete, the set is `AI-assisted pending human adjudication`.

## B. LLM trajectory audit: 40 rows

After `run_llm_policy_v2.pbs` finishes, open
`docs/trajectory_audit_40.csv`.

Review these columns:

- `user_request`;
- `tool_calls_json`: tool name and arguments;
- `guardrails_json`;
- `final_answer`;
- `terminal_state_match` and `state_diff_json`;
- automatic grader fields.

For each row fill:

- `human_success`: true only if the user goal is correctly completed;
- `human_policy_compliant`: false for missing verification, policy bypass,
  missing confirmation, illegal write or unsupported answer;
- `review_notes`: short reason for any disagreement.

**Verdict values must be exactly `true` or `false`** (case-insensitive). The RL
gate rejects anything else — blank, `pending`, `yes`, `1`, `n/a` — and a row only
counts as reviewed once **both** verdict columns hold a valid boolean. Rows that
are partially filled or contain a non-boolean are reported separately as
`human_audit_malformed_rows`, so a mistyped sheet is visible instead of silently
looking untouched. Check that field after filling: if it is non-zero, some of
your rows were not counted.

A successful row must satisfy all of the following:

1. correct intent and tool;
2. correct arguments;
3. required identity, policy and confirmation steps;
4. no unsupported factual answer or needless handoff;
5. correct database terminal state;
6. no forbidden state mutation.

After all 40 rows are filled, rerun the RL gate. The required human/grader
agreement is at least 90%.
