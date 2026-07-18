from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sock_runtime_env import (
    apply_python_runtime_contract,
    apply_rocm_wsl_runtime_defaults,
)


def apply_sock_runtime_profile() -> None:
    profile = os.environ.get("SOCK_RUNTIME_PROFILE", "").strip().lower()
    if profile in {"rocm", "rocm-wsl", "amd"}:
        apply_rocm_wsl_runtime_defaults()
    else:
        apply_python_runtime_contract()


def main() -> None:
    apply_sock_runtime_profile()
    sys.argv = ["vllm", *sys.argv[1:]]

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
