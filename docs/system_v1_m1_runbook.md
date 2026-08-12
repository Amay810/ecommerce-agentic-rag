# System-v1 → M1 execution runbook

Status date: 2026-08-11

## Objective

Measure model improvement with a frozen runtime:

```text
System-v1 + Qwen3-4B Base
vs
System-v1 + E-commerce M1 LoRA
```

The prompt, native function-calling protocol, τ³ Retail environment, user
simulator, judge, task × seed, max steps, and compaction setting remain fixed.
Only the checkpoint changes. Context compaction is a separate off/on ablation.

## Implemented locally

- `ecommerce_rag/agent_runtime.py`: frozen provider-facing Runtime and trace.
- `ecommerce_rag/tau3_agent_adapter.py`: runtime registration as
  `ecommerce_native` without modifying the pinned τ³ checkout.
- `scripts/run_tau3_retail_v1.py`: System-v1 Base/SFT and full train-only
  teacher phases.
- `scripts/audit_tau3_process.py`: deterministic minimum process audit.
- `scripts/build_verified_ecommerce_sft.py`: strict reward/DB/process filtering,
  structure deduplication, split isolation, provenance, and manifest.
- `scripts/validate_verified_sft.py`: fail-closed 400–1,200 train-record gate.
- NSCC jobs for teacher rollout, one M1 LoRA, and M1 serving.

Local verification:

```text
223 passed, 1 skipped
τ³ v1.0.1 ecommerce_native registration: passed
official Retail tool discovery: 16 tools
teacher command freeze/check-only: passed (74 train tasks × k)
```

Historical S0 data was used only to exercise the new pipeline. Strict filtering
accepted 45/296 trajectories; the structure-isolated train split contained only
33 records, so it correctly fails the formal 400-record M1 gate. It must not be
used as the M1 dataset.

## NSCC sequence

### 1. Synchronize the active repository

Synchronize the working tree to the active NSCC checkout without copying `.env`,
secrets, local logs, virtual environments, or external benchmark data. Confirm
the pinned τ³ checkout is exactly:

```text
fc0055dc4e0a316c3f83133267fbd6faaa770992
```

### 2. Freeze System-v1 Base

Serve Qwen3-4B-Instruct-2507, then run from the internet-connected Windows
orchestrator:

```powershell
python -m scripts.run_tau3_retail_v1 `
  --tau-root E:\cv_codex\external\tau2-bench `
  --phase base `
  --agent-name ecommerce_native `
  --compaction off `
  --agent-model hosted_vllm/Qwen3-4B-Instruct-2507 `
  --user-model $env:TAU3_USER_MODEL `
  --nl-assertions-model $env:TAU3_NL_ASSERTIONS_MODEL `
  --pass-k 4 `
  --max-steps 200 `
  --save-to tau3_system_v1_base_qwen3_4b_k4
```

### 3. Generate and gate teacher data

Submit `nscc/run_tau3_teacher_rollout_v1.pbs` only after selecting an immutable,
approved open-weight teacher that supports native tool calls. Required variables:

```text
TEACHER_MODEL
TEACHER_MODEL_ID
TAU3_USER_MODEL
TAU3_NL_ASSERTIONS_MODEL
TAU3_USER_API_KEY
```

The job runs all 74 τ³ Retail train tasks, writes the official result, performs
the process audit, and builds `data/verified_ecommerce_sft_v1/`.

Do not train unless `scripts.validate_verified_sft` reports:

- 400–1,200 train records;
- non-empty dev and held-out splits;
- no structure signature shared across splits;
- official reward 1 and DB match for every admitted record;
- process audit present for every admitted record;
- approved teacher usage rights.

If τ³ train cannot reach the gate, add Task Compiler tasks before training. Do
not lower the 400-record gate or relabel repeated rollouts as new structures.

### 4. Train one M1 LoRA

Submit `nscc/run_ecommerce_m1_lora_v1.pbs`. It uses one frozen configuration:

```text
Qwen3-4B-Instruct-2507
LoRA rank 16 / alpha 32
learning rate 1e-4
2 epochs
max length 32768
4 GPUs
```

No multi-arm hyperparameter search is part of M1.

### 5. Paired Base/M1 acceptance

Serve the selected M1 checkpoint with `nscc/serve_ecommerce_m1_v1.pbs`, then run
the same τ³ Retail 40×4 configuration used for Base with:

```text
agent_model = hosted_vllm/Qwen3-4B-Ecommerce-M1
agent_name = ecommerce_native
compaction = off
```

Report separately:

- τ³ official reward, DB match, action checks, and NL assertions;
- raw write failures and process violations;
- prompt/completion tokens;
- generation and end-to-end latency;
- Runtime intervention rate.

### 6. Compaction ablation

Only after the Base/M1 model comparison, rerun the chosen checkpoint with
`--compaction on`. Attribute the off/on difference to context management, not
model training.

## Current external blocker

At the status date, local implementation is ready but this Codex task cannot
authenticate to NSCC. `ntu.nscc.sg:2222` times out, and no persistent listener is
present on `127.0.0.1:2222`. A transient local connection rejected the configured
`codex_nscc_ed25519` key and offered password authentication only. Remote sync and
PBS submission therefore remain pending; no NSCC job is claimed as submitted.
