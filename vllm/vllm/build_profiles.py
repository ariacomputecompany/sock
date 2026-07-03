# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

_FLAG_COMPONENTS = {
    "VLLM_BUILD_TRITON_KERNELS": "triton_kernels",
    "VLLM_BUILD_DEEPGEMM": "deepgemm",
    "VLLM_BUILD_FMHA_SM100": "fmha_sm100",
    "VLLM_BUILD_FLASHMLA": "flashmla",
    "VLLM_BUILD_QUTLASS": "qutlass",
    "VLLM_BUILD_FLASH_ATTN": "flash_attn",
}

_PROFILE_FLAGS = {
    "full": {
        "VLLM_BUILD_TRITON_KERNELS": True,
        "VLLM_BUILD_DEEPGEMM": True,
        "VLLM_BUILD_FMHA_SM100": True,
        "VLLM_BUILD_FLASHMLA": True,
        "VLLM_BUILD_QUTLASS": True,
        "VLLM_BUILD_FLASH_ATTN": True,
    },
    "core": {
        "VLLM_BUILD_TRITON_KERNELS": False,
        "VLLM_BUILD_DEEPGEMM": False,
        "VLLM_BUILD_FMHA_SM100": False,
        "VLLM_BUILD_FLASHMLA": False,
        "VLLM_BUILD_QUTLASS": False,
        "VLLM_BUILD_FLASH_ATTN": False,
    },
    "minimal-dev": {
        "VLLM_BUILD_TRITON_KERNELS": True,
        "VLLM_BUILD_DEEPGEMM": False,
        "VLLM_BUILD_FMHA_SM100": False,
        "VLLM_BUILD_FLASHMLA": False,
        "VLLM_BUILD_QUTLASS": False,
        "VLLM_BUILD_FLASH_ATTN": False,
    },
    "hopper-flashinfer": {
        "VLLM_BUILD_TRITON_KERNELS": True,
        "VLLM_BUILD_DEEPGEMM": False,
        "VLLM_BUILD_FMHA_SM100": False,
        "VLLM_BUILD_FLASHMLA": False,
        "VLLM_BUILD_QUTLASS": False,
        "VLLM_BUILD_FLASH_ATTN": True,
    },
}


@dataclass(frozen=True)
class BuildProfileResolution:
    profile: str
    enabled_components: tuple[str, ...]
    disabled_components: tuple[str, ...]
    cmake_defines: tuple[str, ...]


def supported_build_profiles() -> tuple[str, ...]:
    return tuple(_PROFILE_FLAGS)


def normalize_build_profile(value: str | None) -> str:
    normalized = (value or "full").strip().lower().replace("_", "-")
    if normalized not in _PROFILE_FLAGS:
        supported = ", ".join(supported_build_profiles())
        raise ValueError(
            f"Unsupported VLLM build profile '{value}'. Supported profiles: {supported}"
        )
    return normalized


def resolve_build_profile(value: str | None) -> BuildProfileResolution:
    profile = normalize_build_profile(value)
    flags = _PROFILE_FLAGS[profile]
    enabled = tuple(
        component
        for flag, component in _FLAG_COMPONENTS.items()
        if flags.get(flag, False)
    )
    disabled = tuple(
        component
        for flag, component in _FLAG_COMPONENTS.items()
        if not flags.get(flag, False)
    )
    defines = tuple(
        f"-D{flag}={'ON' if enabled else 'OFF'}"
        for flag, enabled in flags.items()
    )
    return BuildProfileResolution(
        profile=profile,
        enabled_components=enabled,
        disabled_components=disabled,
        cmake_defines=defines,
    )
