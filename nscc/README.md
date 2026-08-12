# NSCC reproduction jobs

The active NSCC surface contains the frozen System-v1 Base service/evaluation,
an approved open-weight teacher serve/rollout path, strict verified-dataset
construction, one formal M1 LoRA training job, and Base/M1 serving for paired
evaluation. The Windows runner remains the preferred internet-connected
orchestration entrypoint (DeepSeek user/judge stay on Windows).

Current model-learning sequence:

1. Download/serve teacher with `nscc/serve_teacher_v1.pbs`
   (`Qwen3-30B-A3B-Instruct-2507`, Apache-2.0, hermes tool parser).
2. Windows `scripts/run_m1_teacher_pipeline.py` runs train-only calibration,
   τ³ Retail train teacher rollouts, and compiled_retail teacher rollouts via
   the SSH tunnel to the teacher endpoint.
3. `scripts/build_m1_verified_dataset.py` merges audited results into
   `data/verified_ecommerce_sft_v1/` and `scripts/validate_verified_sft.py`
   enforces the 400-train gate.
4. `run_ecommerce_m1_lora_v1.pbs` fails closed unless the strict train split has
   400–1,200 records and isolated dev/held-out structures.
5. `serve_tau3_agent_v1.pbs` serves Base; `serve_ecommerce_m1_v1.pbs` serves M1.

Teacher provenance: `docs/teacher_m1_provenance.json`.
Compiled task set: `data/compiled_retail_m1/` (48 structures, 12 families).

S0 LoRA and older harness jobs remain in the private research archive.
