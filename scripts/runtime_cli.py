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


def _has_cli_flag(name: str) -> bool:
    return name in sys.argv


def apply_sock_tmh_cli_defaults() -> None:
    policy = os.environ.get("SOCK_TMH_KV_POLICY")
    if policy and not _has_cli_flag("--tmh-kv-policy"):
        sys.argv.extend(["--tmh-kv-policy", policy])
    hot_budget = os.environ.get("SOCK_TMH_HOT_BUDGET_PCT")
    if hot_budget and not _has_cli_flag("--tmh-hot-budget-pct"):
        sys.argv.extend(["--tmh-hot-budget-pct", hot_budget])
    log_allocations = os.environ.get("SOCK_TMH_LOG_ALLOCATIONS")
    if log_allocations and "VLLM_TMH_LOG_ALLOCATIONS" not in os.environ:
        os.environ["VLLM_TMH_LOG_ALLOCATIONS"] = log_allocations


def apply_sock_modality_cli_defaults() -> None:
    modality = os.environ.get("SOCK_INFERENCE_MODALITY", "").strip().lower()
    if modality in {"", "auto"}:
        return
    if modality not in {"text", "language", "language-only"}:
        raise ValueError(
            "SOCK_INFERENCE_MODALITY must be one of: auto, text, "
            f"language, language-only; got {modality!r}"
        )
    if not _has_cli_flag("--language-model-only"):
        sys.argv.append("--language-model-only")
    if not _has_cli_flag("--skip-mm-profiling"):
        sys.argv.append("--skip-mm-profiling")


def main() -> None:
    apply_sock_runtime_profile()
    apply_sock_tmh_cli_defaults()
    apply_sock_modality_cli_defaults()
    sys.argv = ["vllm", *sys.argv[1:]]

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
