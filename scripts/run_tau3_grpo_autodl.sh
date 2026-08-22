#!/usr/bin/env bash
set -euo pipefail

source /root/tau3_env.sh
conda activate tau3-grpo

: "${TAU3_ROOT:?TAU3_ROOT is not set}"
: "${TAU_ROOT:?TAU_ROOT is not set}"
: "${QWEN_MODEL_PATH:?QWEN_MODEL_PATH is not set}"
: "${DEEPSEEK_BASE_URL:=https://api.deepseek.com}"

[[ -d "${TAU3_ROOT}" ]] || { echo "TAU3_ROOT does not exist" >&2; exit 2; }
[[ -d "${TAU_ROOT}/src/tau2" ]] || { echo "TAU_ROOT is not a tau2 checkout" >&2; exit 2; }
[[ -f "${QWEN_MODEL_PATH}/config.json" ]] || { echo "Qwen config.json is missing" >&2; exit 2; }
[[ -f "${QWEN_MODEL_PATH}/model.safetensors.index.json" ]] || { echo "Qwen weight index is missing" >&2; exit 2; }
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is missing; inject it into the AutoDL environment before launch" >&2; exit 2; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable; GPU phase has not started" >&2; exit 2; }
gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)"
[[ "${gpu_count}" -ge 2 ]] || { echo "expected at least 2 visible GPUs, found ${gpu_count}" >&2; exit 2; }

exec python "${TAU3_ROOT}/scripts/train_tau3_grpo.py" --launch "$@"
