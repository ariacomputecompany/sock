#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source vllm/.venv/bin/activate

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_TARGET_DEVICE=rocm
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

python scripts/rocm_wsl_preflight.py --build-dlpack
python scripts/rocm_wsl_qwen_smoke.py "$@"
