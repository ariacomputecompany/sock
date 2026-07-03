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
    assert resolution.profile_family == "production"
    assert "triton_kernels" in resolution.enabled_components
    assert "vllm._flashmla_C" in resolution.extension_targets
    assert "-DVLLM_BUILD_FLASH_ATTN=ON" in resolution.cmake_defines


def test_minimal_dev_profile_disables_optional_cuda_packs() -> None:
    resolution = build_profiles.resolve_build_profile("minimal-dev")

    assert resolution.profile == "minimal-dev"
    assert resolution.profile_family == "developer"
    assert resolution.developer_friendly is True
    assert resolution.enabled_components == ("triton_kernels",)
    assert "flash_attn" in resolution.disabled_components
    assert resolution.optional_backend_packs == ("triton_kernels",)
    assert resolution.experimental_packs == ()
    assert resolution.enabled_native_families == ("base_runtime",)
    assert "marlin" in resolution.disabled_native_families
    assert resolution.editable_sync_roots == ("vllm/third_party/triton_kernels",)
    assert "-DVLLM_BUILD_TRITON_KERNELS=ON" in resolution.cmake_defines
    assert "-DVLLM_BUILD_DEEPGEMM=OFF" in resolution.cmake_defines
    assert "-DVLLM_BUILD_FAMILY_MARLIN=OFF" in resolution.cmake_defines


def test_build_profile_normalization_accepts_underscores() -> None:
    resolution = build_profiles.resolve_build_profile("hopper_flashinfer")

    assert resolution.profile == "hopper-flashinfer"
    assert "flash_attn" in resolution.enabled_components


def test_targeted_profiles_select_only_requested_backend_targets() -> None:
    resolution = build_profiles.resolve_build_profile("deepgemm")

    assert resolution.profile_family == "targeted"
    assert resolution.cuda_arches == ("9.0", "10.0")
    assert resolution.enabled_components == ("triton_kernels", "deepgemm")
    assert resolution.enabled_native_families == (
        "base_runtime",
        "model_fused_ops",
        "cutlass_scaled_mm",
        "cutlass_moe",
        "fp4",
        "hadamard",
    )
    assert resolution.extension_targets == (
        "vllm.triton_kernels",
        "vllm._deep_gemm_C",
    )
    assert resolution.editable_sync_roots == (
        "vllm/third_party/triton_kernels",
        "vllm/third_party/deep_gemm",
    )


def test_blackwell_profile_marks_experimental_pack() -> None:
    resolution = build_profiles.resolve_build_profile("blackwell-fa3")

    assert resolution.cuda_arches == ("10.0",)
    assert resolution.optional_backend_packs == ("triton_kernels", "flash_attn")
    assert resolution.experimental_packs == ("fmha_sm100",)
    assert "vllm.fmha_sm100" in resolution.extension_targets
    assert "vllm/third_party/fmha_sm100" in resolution.editable_sync_roots


def test_editable_sync_and_component_helpers_follow_profile() -> None:
    resolution = build_profiles.resolve_build_profile("flashattn")

    assert build_profiles.component_enabled(resolution, "flash_attn") is True
    assert build_profiles.component_enabled(resolution, "deepgemm") is False
    assert build_profiles.native_family_enabled(resolution, "base_runtime") is True
    assert build_profiles.native_family_enabled(resolution, "marlin") is False
    assert build_profiles.editable_sync_enabled(
        resolution, "vllm/vllm_flash_attn"
    ) is True
    assert build_profiles.editable_sync_enabled(
        resolution, "vllm/third_party/deep_gemm"
    ) is False


def test_unknown_build_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported VLLM build profile"):
        build_profiles.resolve_build_profile("mystery-pack")
