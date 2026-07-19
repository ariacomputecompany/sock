from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sock_runtime_env import apply_runtime_profile


def test_cuda_runtime_profile_sets_production_cuda_defaults(monkeypatch) -> None:
    for key in [
        "CUDA_DEVICE_ORDER",
        "CUDA_MODULE_LOADING",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "SOCK_RUNTIME_PROFILE",
        "TOKENIZERS_PARALLELISM",
        "VLLM_TARGET_DEVICE",
        "VLLM_USE_V1",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_WORKER_MULTIPROC_METHOD",
    ]:
        monkeypatch.delenv(key, raising=False)

    apply_runtime_profile("cuda")

    assert os.environ["VLLM_TARGET_DEVICE"] == "cuda"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_MODULE_LOADING"] == "LAZY"
    assert os.environ["VLLM_USE_V1"] == "1"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["PYTHONNOUSERSITE"] == "1"
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_cuda_runtime_profile_preserves_explicit_user_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_MODULE_LOADING", "EAGER")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    apply_runtime_profile("nvidia")

    assert os.environ["CUDA_MODULE_LOADING"] == "EAGER"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
