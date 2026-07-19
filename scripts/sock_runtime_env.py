from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_VLLM_ROOT = REPO_ROOT / "vllm"
VENDORED_VENV_ROOT = VENDORED_VLLM_ROOT / ".venv"
VENDORED_PYTHON = VENDORED_VENV_ROOT / "bin" / "python"


def prepend_env_path(name: str, value: Path) -> None:
    value_str = str(value)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value_str not in parts:
        os.environ[name] = os.pathsep.join([value_str, *parts])


def prepend_env_value(env: dict[str, str], name: str, value: Path) -> None:
    value_str = str(value)
    parts = [part for part in env.get(name, "").split(os.pathsep) if part]
    if value_str not in parts:
        env[name] = os.pathsep.join([value_str, *parts])


def set_default_env(name: str, value: str) -> None:
    current = os.environ.get(name)
    if current is None or current == "":
        os.environ[name] = value


def apply_python_runtime_contract() -> None:
    """Apply the deterministic repo-local Python/vLLM runtime contract."""
    if str(VENDORED_VLLM_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDORED_VLLM_ROOT))

    prepend_env_path("PYTHONPATH", VENDORED_VLLM_ROOT)
    set_default_env("PYTHONNOUSERSITE", "1")
    set_default_env("PYTHONHASHSEED", "0")
    set_default_env("TOKENIZERS_PARALLELISM", "false")


def apply_rocm_wsl_runtime_defaults() -> None:
    apply_python_runtime_contract()
    set_default_env("VLLM_TARGET_DEVICE", "rocm")
    set_default_env("VLLM_USE_V2_MODEL_RUNNER", "0")
    set_default_env("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
    set_default_env("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def apply_cuda_runtime_defaults() -> None:
    apply_python_runtime_contract()
    set_default_env("VLLM_TARGET_DEVICE", "cuda")
    set_default_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    set_default_env("CUDA_MODULE_LOADING", "LAZY")
    set_default_env("VLLM_USE_V2_MODEL_RUNNER", "1")
    set_default_env("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def apply_runtime_profile(profile: str) -> None:
    normalized = profile.strip().lower()
    if normalized in {"rocm", "rocm-wsl", "amd"}:
        apply_rocm_wsl_runtime_defaults()
    elif normalized in {"cuda", "nvidia", "nva"}:
        apply_cuda_runtime_defaults()
    else:
        apply_python_runtime_contract()


def subprocess_env(*, rocm_wsl: bool = True) -> dict[str, str]:
    if rocm_wsl:
        apply_rocm_wsl_runtime_defaults()
    else:
        apply_python_runtime_contract()
    return os.environ.copy()


def tool_subprocess_env(*, rocm_wsl: bool = False) -> dict[str, str]:
    env = subprocess_env(rocm_wsl=rocm_wsl)
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        prepend_env_value(env, "PATH", cargo_bin)
    return env


def isolated_process_kwargs() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    terminate_timeout_s: int = 15,
) -> None:
    if process.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=terminate_timeout_s)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()
            return

    process.terminate()
    try:
        process.wait(timeout=terminate_timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
