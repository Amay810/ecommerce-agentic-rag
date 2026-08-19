# Current status

Last updated: 2026-08-19

## Decision

`P0 experiment control plane` and `P0 real-run provenance` are **CLOSED**.
Do not reopen the tau3 runner for this.

`G0-E exploration config` is **FROZEN**. Do not leave sampling knobs unspecified.
Do not start GRPO training from this freeze. Do not put optimizer `P` (prompts per
step) into the exploration command.

`retail_task_compiler` / `compiled_retail_m1` remains **No-Go**.
Do not add tasks, rerun S0, retune SFT, edit reward, or touch Track B.

## P0 freeze

Smoke `tau3_retail_v1_p0_smoke` (Retail train tasks `0`, `1`; `pass_k=1`):

| Native `info` | Observed |
|---|---|
| `agent_info.implementation` | `llm_agent` |
| `agent_info.llm` | `hosted_vllm/Qwen3-4B-Instruct-2507` |
| `agent_info.llm_args.temperature` | `0.0` |
| `user_info.llm` | `deepseek/deepseek-chat` |
| `user_info.llm_args.temperature` | `0.0` |
| `seed` | `300` |
| `num_trials` | `1` |
| `max_steps` | `200` |
| `checks_passed` | 8 |
| `experiment_summary.valid` | `true` |
| `infrastructure_errors` | `0` |

`task1=1`, `task0=0` is protocol smoke, not a Base-quality or GRPO claim.

## G0-E frozen exploration config

```text
group target K         = 8
exploration num_trials = 8
agent_temperature      = 0.8
user_temperature       = 0.0
seed                   = 300
max_steps              = 200
agent_name             = llm_agent
agent_model            = hosted_vllm/Qwen3-4B-Instruct-2507
user_model             = deepseek/deepseek-chat
task_split             = train
tasks                  = official Retail train 74
trajectories           = 74 × 8 = 592
```

`num_trials=8` estimates, before training, how many prompts would yield
non-zero-variance groups if each prompt were later sampled at GRPO group size
K=8. It is not a claim that tau2 `num_trials` is the GRPO loss object.

`agent_temperature=0.8` is the first-round engineering choice from the Slime
Qwen3-4B RL recipe (`rollout-temperature=0.8`). It is **not** claimed to be
optimal for τ² Retail + Qwen3-4B. Whether it is usable is decided by this
74×8 group-variance measurement.

Relative to historical S0 (`temperature=0.0`, `num_trials=4`, `seed=300`),
the intended deltas are only:

```text
agent temp: 0.0 → 0.8
trials:     4   → 8
```

User simulator stays DeepSeek at `user_temperature=0.0` so Agent sampling is
the only new entropy source.

## After the first 74×8 artifact

1. Native provenance in that run's `results.json` (`info`, not the submit command):
   implementation, agent model, agent temp `0.8`, user model, user temp `0.0`,
   seed `300`, `num_trials=8`, `max_steps=200`.
2. Infrastructure validity (`experiment_summary.valid`, `infrastructure_errors`).
3. Only then reward: per-task `0/8`, `1/8`–`7/8` mixed, `8/8`; compare to S0
   classes (21 all-zero, 37 mixed, 16 all-one); report
   `non-zero-variance groups / 74`.

Do not invent a “enough mixed groups” threshold before that distribution exists.

## G0-E exploration result (COMPLETE)

Artifact: [`data/simulations/tau3_g0e_train_qwen3_4b_temp08_k8/results.json`](../data/simulations/tau3_g0e_train_qwen3_4b_temp08_k8/results.json)

Summary: [`docs/experiments/tau3_g0e_train_qwen3_4b_temp08_k8_summary.json`](experiments/tau3_g0e_train_qwen3_4b_temp08_k8_summary.json)

| Check | Observed |
|---|---|
| `simulations` | `592` |
| `experiment_summary.valid` | `true` |
| `infrastructure_errors` | `0` |
| `mean_reward` | `0.439` |
| native agent temp | `0.8` |
| native user temp | `0.0` |
| native seed / `num_trials` | `300` / `8` |
| `tau3_experiment.checks_passed` | 8/8 |

Per-task group classes (74 train tasks × 8 trials):

| Class | G0-E (`temp=0.8`, K=8) | S0 reference (`temp=0.0`, K=4) |
|---|---:|---:|
| `0/8` all-fail | 17 | 21 |
| mixed (`1/8`–`7/8`) | 47 | 37 |
| `8/8` all-pass | 10 | 16 |
| non-zero-variance groups | **57 / 74** | 53 / 74 |

This distribution exists; no GRPO-start threshold is asserted here.

## Next

```text
P0                       CLOSED
G0-E exploration         COMPLETE (592/592, valid)
NEXT → decide whether to start GRPO from this group-variance snapshot
```

RFT/GRPO training code is still not implemented.
Closed compiler artifacts live in
[`Amay810/ecommerce-agentic-rag-archive`](https://github.com/Amay810/ecommerce-agentic-rag-archive).
