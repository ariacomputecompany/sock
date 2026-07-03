# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest


def _load_build_profiles():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "vllm" / "build_profiles.py"
    spec = importlib.util.spec_from_file_location("vllm_build_profiles", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_profiles = _load_build_profiles()


def test_default_profile_is_full() -> None:
    resolution = build_profiles.resolve_build_profile(None)

    assert resolution.profile == "full"
    assert "triton_kernels" in resolution.enabled_components
    assert "-DVLLM_BUILD_FLASH_ATTN=ON" in resolution.cmake_defines


def test_minimal_dev_profile_disables_optional_cuda_packs() -> None:
    resolution = build_profiles.resolve_build_profile("minimal-dev")

    assert resolution.profile == "minimal-dev"
    assert resolution.enabled_components == ("triton_kernels",)
    assert "flash_attn" in resolution.disabled_components
    assert "-DVLLM_BUILD_TRITON_KERNELS=ON" in resolution.cmake_defines
    assert "-DVLLM_BUILD_DEEPGEMM=OFF" in resolution.cmake_defines


def test_build_profile_normalization_accepts_underscores() -> None:
    resolution = build_profiles.resolve_build_profile("hopper_flashinfer")

    assert resolution.profile == "hopper-flashinfer"
    assert "flash_attn" in resolution.enabled_components


def test_unknown_build_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported VLLM build profile"):
        build_profiles.resolve_build_profile("mystery-pack")
