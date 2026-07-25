#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ENV_FILE="${ENV_FILE:-docker/inference/.env.inference}"
if [[ -z "${QUILT_CLI:-}" && -x /home/deepsaint/work/quilt-oss/quilt-core/target/release/cli ]]; then
  QUILT_CLI=/home/deepsaint/work/quilt-oss/quilt-core/target/release/cli
else
  QUILT_CLI="${QUILT_CLI:-cli}"
fi
QUILT_SERVER_ADDR="${QUILT_SERVER_ADDR:-http://127.0.0.1:50051}"
QUILT_IMAGE_PATH="${QUILT_IMAGE_PATH:-}"
QUILT_CONTAINER_NAME="${QUILT_CONTAINER_NAME:-sock-inference}"

if [[ ! -f "${ENV_FILE}" ]]; then
  docker/inference/run-rocm.sh --prepare-only
fi

if [[ -z "${QUILT_IMAGE_PATH}" ]]; then
  echo "QUILT_IMAGE_PATH is required for Quilt execution." >&2
  echo "Build/export the ROCm image or point this at a Quilt-compatible rootfs tarball." >&2
  exit 64
fi

if [[ ! -f "${QUILT_IMAGE_PATH}" ]]; then
  echo "QUILT_IMAGE_PATH does not exist: ${QUILT_IMAGE_PATH}" >&2
  exit 66
fi

if ! command -v "${QUILT_CLI}" >/dev/null 2>&1; then
  echo "Quilt CLI not found: ${QUILT_CLI}" >&2
  echo "Set QUILT_CLI=/path/to/cli or install quilt-core's cli binary." >&2
  exit 127
fi

env_args=()
while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
  env_args+=(--env "${line}")
done < "${ENV_FILE}"

source "${ENV_FILE}"
mkdir -p "${HF_HOME:-/home/deepsaint/.cache/huggingface}" "${SOCK_DATA_DIR:-/home/deepsaint/.sock-inference}"

"${QUILT_CLI}" --server-addr "${QUILT_SERVER_ADDR}" create \
  --name "${QUILT_CONTAINER_NAME}" \
  --async-mode \
  --image-path "${QUILT_IMAGE_PATH}" \
  --memory-limit 0 \
  --cpu-limit 0 \
  --volume "${HF_HOME:-/home/deepsaint/.cache/huggingface}:/root/.cache/huggingface" \
  --volume "${SOCK_DATA_DIR:-/home/deepsaint/.sock-inference}:/data" \
  "${env_args[@]}" \
  -- sock-inference-entrypoint
