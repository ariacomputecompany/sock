#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ENV_FILE="${ENV_FILE:-docker/inference/.env.inference}"
WHEEL_SRC="${WHEEL_SRC:-/home/deepsaint/wheelhouse/pytorch-gfx1151/torch-2.11.0+gfx1151-cp312-cp312-linux_x86_64.whl}"
WHEEL_DST_DIR="docker/inference/wheelhouse"

mkdir -p "${WHEEL_DST_DIR}"
if [[ -f "${WHEEL_SRC}" && ! -e "${WHEEL_DST_DIR}/$(basename "${WHEEL_SRC}")" ]]; then
  cp "${WHEEL_SRC}" "${WHEEL_DST_DIR}/"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp docker/inference/.env.example "${ENV_FILE}"
  api_key="$(docker/inference/generate-api-key.sh)"
  linx_secret="$(docker/inference/generate-api-key.sh)"
  python3 - "${ENV_FILE}" "${api_key}" "${linx_secret}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
api_key = sys.argv[2]
linx_secret = sys.argv[3]
text = path.read_text()
text = text.replace("SOCK_API_KEY=\n", f"SOCK_API_KEY={api_key}\n")
text = text.replace(
    "LINX_JWT_SECRET=change-this-long-random-secret\n",
    f"LINX_JWT_SECRET={linx_secret}\n",
)
path.write_text(text)
PY
  echo "created ${ENV_FILE} with generated SOCK_API_KEY and LINX_JWT_SECRET"
fi

if [[ "${1:-}" == "--prepare-only" ]]; then
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed on this host" >&2
  exit 127
fi

if docker compose version >/dev/null 2>&1; then
  docker compose -f docker/inference/compose.rocm.yml --env-file "${ENV_FILE}" up --build
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose -f docker/inference/compose.rocm.yml --env-file "${ENV_FILE}" up --build
else
  echo "Docker is installed, but neither 'docker compose' nor 'docker-compose' is available" >&2
  exit 127
fi
