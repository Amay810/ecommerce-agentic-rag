# NSCC jobs

Workdir: `/scratch/users/ntu/s250045/ecommerce-agentic-rag-main`
(`qsub`/`qstat` in `/opt/pbs/bin`). Do not submit from sibling scratch copies.

Windows is the internet-connected orchestrator (DeepSeek user/judge). NSCC
serves the agent. Current stop: [docs/current_status.md](../docs/current_status.md).

| Job | Use |
|---|---|
| `serve_tau3_agent_v1.pbs` | Qwen3-4B Base |
| `run_tau3_retail_base_v1.pbs` | cluster-side Base eval helper |
| `train_tau3_grpo_v1.pbs` | two-GPU G0 GRPO implementation smoke/pilot |

30B teacher serve/rollout jobs, M1 LoRA/serve jobs, and the legacy
action-constraint PBS are in the archive, not this tree.

The verified historical G0-E topology was Windows tau2/DeepSeek reaching the
NSCC Qwen vLLM through an SSH tunnel. It did **not** verify an NSCC-to-DeepSeek
relay. The GRPO job introduces a new, unresolved pre-smoke requirement:
the allocated PBS compute node must reach an externally provided
OpenAI-compatible DeepSeek relay/proxy.

`DEEPSEEK_BASE_URL` is required and has no public-DeepSeek fallback; its value
must be supplied and verified separately. The PBS maps it to
`DEEPSEEK_API_BASE` for tau2's LiteLLM judge. A localhost endpoint on an NSCC
login node is not assumed to be reachable from the PBS compute node. The relay
is external to the GRPO conda environment and is never started by PBS.

Prepare the environment by cloning the known-good Qwen/vLLM environment; do
not modify it in place:

```bash
module load cuda/12.8.1
module load miniforge3/25.3.1
module load git/2.39.2
conda create --prefix /scratch/users/ntu/s250045/conda-envs/tau3-grpo-v1 \
  --clone /scratch/users/ntu/s250045/conda-envs/ecommerce-vllm
conda activate /scratch/users/ntu/s250045/conda-envs/tau3-grpo-v1
python -m pip install -r requirements-grpo-nscc.txt
python -m pip check
```

The install command is preparation only; do not run it from a PBS job. The
runtime preflight verifies Python 3.12, the retained Qwen/vLLM stack, the
missing tau2/VERL packages, the exact VERL commit, the frozen tau2 snapshot,
and 74 Retail train tasks before launching anything.

To run only that allocated-node preflight, submit with
`GRPO_PREFLIGHT_ONLY=1`; it exits before the VERL command. Use
`GRPO_OPTIMIZER_STEPS=1` for the first actual plumbing smoke and do not submit
the formal nine-step pilot until that smoke has been inspected.

It is not a claim that the local VM or this preflight has run the pilot.
