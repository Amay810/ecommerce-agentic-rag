# G0 tau3 retail GRPO pilot implementation

The implementation is a thin VERL-first integration. The pinned tau2 source is
external and is supplied through `TAU_ROOT`; it is not copied into the main
repository.

Frozen variables are encoded in
`ecommerce_rag/grpo/config.py`: tau2 commit `fc0055dc`, retail train split
(74 tasks), Qwen3-4B-Instruct-2507, DeepSeek user simulator and
`deepseek/deepseek-chat` NL judge, agent temperature 0.8, user temperature
0.0, seed 300, max steps 200, K=8, P=2, and 9 optimizer steps. The only
reward passed to GRPO is the pinned tau2 `EvaluationType.ALL` terminal binary
reward.

Run the VM-safe checks with:

```bash
python -m scripts.train_tau3_grpo --check-only --output-dir /tmp/tau3-grpo-check
python scripts/analyze_tau3_grpo_pilot.py \
  --steps /tmp/tau3-grpo-check/nine_step/steps.jsonl
```

The VM checks cover group construction, 0..7 rollout indices, binary reward
mapping, the adapter's mocked pinned-tau2 contract, assistant-only loss
masking, fake optimizer/checkpoint/version refresh, and one-to-nine-step
artifact generation. They do not run a real tau2 episode, VERL, vLLM,
DeepSeek, or a model.

On NSCC, first verify the pinned snapshot and then use
`nscc/train_tau3_grpo_v1.pbs`. The launcher targets VERL v0.8.0's
`verl.trainer.main_ppo_sync` with a colocated two-GPU hybrid actor/rollout
worker. `GRPO_OPTIMIZER_STEPS=1` is the plumbing smoke; the default `9` is the
formal pilot. Memory feasibility and actual episode throughput remain
NSCC-only unknowns.

The VERL integration targets the API used by VERL 0.8's native
`ToolAgentLoop`; do not replace it with a local GRPO loss or a second rollout
engine. The real run must confirm the exact versions in
`requirements-grpo-nscc.txt`, the native AgentLoop response mask, and that
the synchronous checkpoint manager refreshes rollout weights after each
optimizer update.
