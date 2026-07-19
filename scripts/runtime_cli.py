from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sock_runtime_env import (
    apply_runtime_profile,
)


def apply_sock_runtime_profile() -> None:
    profile = os.environ.get("SOCK_RUNTIME_PROFILE", "").strip().lower()
    apply_runtime_profile(profile)


def apply_sock_tmh_cli_defaults() -> None:
    policy = os.environ.get("SOCK_TMH_KV_POLICY")
    if policy and "--tmh-kv-policy" not in sys.argv:
        sys.argv.extend(["--tmh-kv-policy", policy])
    hot_budget = os.environ.get("SOCK_TMH_HOT_BUDGET_PCT")
    if hot_budget and "--tmh-hot-budget-pct" not in sys.argv:
        sys.argv.extend(["--tmh-hot-budget-pct", hot_budget])
    log_allocations = os.environ.get("SOCK_TMH_LOG_ALLOCATIONS")
    if log_allocations and "VLLM_TMH_LOG_ALLOCATIONS" not in os.environ:
        os.environ["VLLM_TMH_LOG_ALLOCATIONS"] = log_allocations


def main() -> None:
    apply_sock_runtime_profile()
    apply_sock_tmh_cli_defaults()
    sys.argv = ["vllm", *sys.argv[1:]]

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
