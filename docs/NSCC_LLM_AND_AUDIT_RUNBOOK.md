# NSCC LLMPolicy and human-audit runbook

## 1. Verify the existing local model

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-git
module load miniforge3/25.3.1
source activate /scratch/users/ntu/s250045/conda-envs/ecommerce-rag

test -s /scratch/users/ntu/s250045/models/Qwen3-4B-Instruct-2507/config.json
du -sh /scratch/users/ntu/s250045/models/Qwen3-4B-Instruct-2507
```

The e-commerce LLMPolicy job uses Qwen3-4B-Instruct-2507. The existing
Qwen2.5-0.5B-Instruct directory is reserved for the medical Grounded-SFT work.
No model download is required. Do not run AutoModel inference on the login node.

## 2. Build the retrieval adjudication panel

This job uses the existing multilingual embedding and 5k index. It does not
load or call a generative LLM.

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-git
qsub nscc/build_retrieval_audit_evidence.pbs
tail -f output_erag_audit_evidence.log
```

Download `docs/retrieval_audit_panel_50.html` and open it directly in a browser.
Review all cases and use its export button to save the completed CSV.

## 3. Submit 360 real LLMPolicy trajectories

The PBS runs:

- dev: 60 tasks x 3;
- locked: 60 tasks x 3;
- total: 360 real LLMPolicy trajectories;
- preference-pair export;
- 40-row audit export;
- fail-closed RL gate.

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-git
git pull --ff-only

test -s ecommerce_rag/data/harness_tasks_v2.jsonl
test -s ecommerce_rag/data/agent_env_v2.db
test -s ecommerce_rag/index_5000/embeddings.npy

LLM_JOB=$(qsub nscc/run_llm_policy_v2.pbs)
echo "$LLM_JOB"
qstat -u s250045
tail -f output_agent_v2.log
```

The job refuses to mix with an existing
`logs/harness_v2_llm_360.sqlite`. If an earlier run exists, archive it
explicitly before resubmission.

Produced outputs (the job completed; the **run itself is invalid** — see below):

- `docs/harness_v2_llm_dev_pass3.json` — carries a `run_validity: invalid` block;
- `docs/harness_v2_llm_locked_pass3.json` — carries a `run_validity: invalid` block;
- `docs/trajectory_audit_40.csv` — 40-row template, **0 rows adjudicated**;
- `docs/agent_rl_gate_v2.json`;
- `logs/harness_v2_llm_360.sqlite` — 360 trajectories, all parse failures;
- `logs/action_preferences.jsonl` — **empty (0 bytes)**. A policy that emits a
  single action cannot generate preference pairs, so this is a downstream
  symptom of the failed run, not a usable artifact.

> **This batch is an invalid integration run.** `model_action_parse_failure`
> fired on 360/360 trajectories and every step degraded to `escalate_to_human`.
> Do not quote its numbers as Qwen agent capability. Before re-running: persist
> raw model output and parse attempts in the trace, then verify with a 5–10 task
> smoke run that actions parse and tools actually execute.

## 4. Audit 40 trajectories

Copy or pull `docs/trajectory_audit_40.csv`, open it in Excel and follow
`docs/HUMAN_AUDIT_GUIDE.md`.

Fill all 40 values in:

- `human_success`;
- `human_policy_compliant`;
- `review_notes`.

Use lowercase `true` or `false`.

## 5. Re-run the gate after audit

Return the completed CSV to the same repository path, then run this light
login-node command:

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-git
module load miniforge3/25.3.1
source activate /scratch/users/ntu/s250045/conda-envs/ecommerce-rag

python -m ecommerce_rag.rl_gate   --tasks ecommerce_rag/data/harness_tasks_v2.jsonl   --store logs/harness_v2_llm_360.sqlite   --audit docs/trajectory_audit_40.csv   --preference-pairs logs/action_preferences.jsonl   --output docs/agent_rl_gate_v2.json
```

Interpretation:

- agreement below 90%: inspect grader disagreements before training;
- base LLM success at or above 95%: do not train because the gate treats
  improvement headroom as insufficient;
- fewer than 200 pairs: do not run DPO; more varied repeated trajectories are
  required;
- all checks true: next-action SFT/DPO may be considered, but is still optional.
