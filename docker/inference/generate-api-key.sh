#!/usr/bin/env bash
set -euo pipefail

if command -v openssl >/dev/null 2>&1; then
  printf 'sk-sock-%s\n' "$(openssl rand -hex 24)"
else
  python3 - <<'PY'
import secrets
print("sk-sock-" + secrets.token_hex(24))
PY
fi
