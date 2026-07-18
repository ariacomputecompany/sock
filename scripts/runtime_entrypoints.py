from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sock_runtime_env import apply_python_runtime_contract

apply_python_runtime_contract()


def ensure_runner_compatibility() -> None:
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None:
        return
    try:
        from vllm.utils.platform_utils import is_uva_available
    except Exception:
        return
    if not is_uva_available():
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def runtime_config_from_env() -> dict[str, Any]:
    return {
        "model": os.environ.get("SOCK_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "max_model_len": _env_int("SOCK_MAX_MODEL_LEN", 512),
        "gpu_memory_utilization": _env_float(
            "SOCK_GPU_MEMORY_UTILIZATION", 0.5
        ),
        "enforce_eager": _env_bool("SOCK_ENFORCE_EAGER", True),
        "trust_remote_code": _env_bool("SOCK_TRUST_REMOTE_CODE", False),
        "distributed_executor_backend": os.environ.get(
            "SOCK_EXECUTOR_BACKEND", "uni"
        ),
    }


def _cleanup_worker_context(executor: Any) -> None:
    try:
        if hasattr(executor, "shutdown"):
            executor.shutdown()
    finally:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()


def build_worker_context(document: dict[str, Any]):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.executor.abstract import Executor

    ensure_runner_compatibility()
    config = runtime_config_from_env()
    engine_args = EngineArgs(**config)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.LLM_CLASS
    )
    executor_class = Executor.get_class(vllm_config)
    executor = executor_class(vllm_config)

    driver_worker = getattr(executor, "driver_worker", None)
    worker = getattr(driver_worker, "worker", None)
    if worker is None:
        if hasattr(executor, "shutdown"):
            executor.shutdown()
        raise RuntimeError(
            "Unable to resolve vendored vLLM worker from the initialized engine."
        )

    # Keep the owning executor/config alive for the lifetime of the returned worker context.
    setattr(worker, "_sock_executor_owner", executor)
    setattr(worker, "_sock_engine_config", vllm_config)
    setattr(worker, "_sock_entrypoint_document", document)
    setattr(worker, "_sock_cleanup", lambda: _cleanup_worker_context(executor))
    return worker


def describe_worker_context(document: dict[str, Any]) -> dict[str, Any]:
    worker = build_worker_context(document)
    return {
        "worker_type": f"{type(worker).__module__}.{type(worker).__qualname__}",
        "has_model_runner": hasattr(worker, "model_runner"),
        "has_scheduler_config": hasattr(worker, "scheduler_config"),
        "runtime_config": runtime_config_from_env(),
        "v2_model_runner_env": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
        "scope_name": document.get("scope_name"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        document = json.load(f)
    print(json.dumps(describe_worker_context(document), sort_keys=True))
