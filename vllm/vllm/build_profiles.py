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

_OPTIONAL_BACKEND_COMPONENTS = (
    "triton_kernels",
    "deepgemm",
    "flashmla",
    "qutlass",
    "flash_attn",
)
_EXPERIMENTAL_COMPONENTS = ("fmha_sm100",)
_BENCHMARK_ONLY_COMPONENTS: tuple[str, ...] = ()

_COMPONENT_EXTENSION_TARGETS = {
    "triton_kernels": ("vllm.triton_kernels",),
    "deepgemm": ("vllm._deep_gemm_C",),
    "fmha_sm100": ("vllm.fmha_sm100",),
    "flashmla": ("vllm._flashmla_C", "vllm._flashmla_extension_C"),
    "qutlass": ("vllm._qutlass_C",),
    "flash_attn": (
        "vllm.vllm_flash_attn._vllm_fa2_C",
        "vllm.vllm_flash_attn._vllm_fa3_C",
        "vllm.vllm_flash_attn._vllm_fa4_cutedsl_C",
    ),
}

_COMPONENT_EDITABLE_SYNC_ROOTS = {
    "triton_kernels": ("vllm/third_party/triton_kernels",),
    "deepgemm": ("vllm/third_party/deep_gemm",),
    "fmha_sm100": ("vllm/third_party/fmha_sm100",),
    "flash_attn": ("vllm/vllm_flash_attn",),
}

_NATIVE_FAMILY_FLAGS = {
    "VLLM_BUILD_FAMILY_MODEL_FUSED_OPS": "model_fused_ops",
    "VLLM_BUILD_FAMILY_MACHETE": "machete",
    "VLLM_BUILD_FAMILY_MARLIN": "marlin",
    "VLLM_BUILD_FAMILY_ALLSPARK": "allspark",
    "VLLM_BUILD_FAMILY_CUTLASS_SCALED_MM": "cutlass_scaled_mm",
    "VLLM_BUILD_FAMILY_CUTLASS_MOE": "cutlass_moe",
    "VLLM_BUILD_FAMILY_FP4": "fp4",
    "VLLM_BUILD_FAMILY_W4A8": "w4a8",
    "VLLM_BUILD_FAMILY_ATTENTION_MLA": "attention_mla",
    "VLLM_BUILD_FAMILY_HADAMARD": "hadamard",
    "VLLM_BUILD_FAMILY_MOE_MARLIN": "moe_marlin",
}

_NATIVE_FAMILY_CATEGORIES = {
    "always_needed": ("base_runtime",),
    "model_specific": ("model_fused_ops",),
    "backend_specific": (
        "machete",
        "marlin",
        "cutlass_scaled_mm",
        "cutlass_moe",
        "fp4",
        "w4a8",
        "attention_mla",
        "hadamard",
        "moe_marlin",
    ),
    "legacy_compatibility": ("allspark",),
    "test_benchmark_only": (),
}

_PROFILE_SPECS = {
    "full": {
        "profile_family": "production",
        "developer_friendly": False,
        "cuda_arches": (),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": True,
            "VLLM_BUILD_FMHA_SM100": True,
            "VLLM_BUILD_FLASHMLA": True,
            "VLLM_BUILD_QUTLASS": True,
            "VLLM_BUILD_FLASH_ATTN": True,
        },
        "native_families": (
            "base_runtime",
            "model_fused_ops",
            "machete",
            "marlin",
            "allspark",
            "cutlass_scaled_mm",
            "cutlass_moe",
            "fp4",
            "w4a8",
            "attention_mla",
            "hadamard",
            "moe_marlin",
        ),
    },
    "core": {
        "profile_family": "production",
        "developer_friendly": False,
        "cuda_arches": (),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": False,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": False,
        },
        "native_families": ("base_runtime",),
    },
    "minimal-dev": {
        "profile_family": "developer",
        "developer_friendly": True,
        "cuda_arches": (),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": False,
        },
        "native_families": ("base_runtime",),
    },
    "flashattn": {
        "profile_family": "targeted",
        "developer_friendly": False,
        "cuda_arches": (),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": True,
        },
        "native_families": ("base_runtime",),
    },
    "deepgemm": {
        "profile_family": "targeted",
        "developer_friendly": False,
        "cuda_arches": ("9.0", "10.0"),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": True,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": False,
        },
        "native_families": (
            "base_runtime",
            "model_fused_ops",
            "cutlass_scaled_mm",
            "cutlass_moe",
            "fp4",
            "hadamard",
        ),
    },
    "flashmla": {
        "profile_family": "targeted",
        "developer_friendly": False,
        "cuda_arches": ("9.0", "10.0"),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": True,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": False,
        },
        "native_families": (
            "base_runtime",
            "model_fused_ops",
            "attention_mla",
        ),
    },
    "qutlass": {
        "profile_family": "targeted",
        "developer_friendly": False,
        "cuda_arches": ("9.0", "10.0"),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": True,
            "VLLM_BUILD_FLASH_ATTN": False,
        },
        "native_families": (
            "base_runtime",
            "cutlass_scaled_mm",
            "fp4",
            "hadamard",
        ),
    },
    "hopper-flashinfer": {
        "profile_family": "targeted",
        "developer_friendly": False,
        "cuda_arches": ("9.0",),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": False,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": True,
        },
        "native_families": ("base_runtime",),
    },
    "blackwell-fa3": {
        "profile_family": "targeted",
        "developer_friendly": False,
        "cuda_arches": ("10.0",),
        "flags": {
            "VLLM_BUILD_TRITON_KERNELS": True,
            "VLLM_BUILD_DEEPGEMM": False,
            "VLLM_BUILD_FMHA_SM100": True,
            "VLLM_BUILD_FLASHMLA": False,
            "VLLM_BUILD_QUTLASS": False,
            "VLLM_BUILD_FLASH_ATTN": True,
        },
        "native_families": ("base_runtime",),
    },
}


@dataclass(frozen=True)
class BuildProfileResolution:
    profile: str
    profile_family: str
    developer_friendly: bool
    cuda_arches: tuple[str, ...]
    enabled_components: tuple[str, ...]
    disabled_components: tuple[str, ...]
    optional_backend_packs: tuple[str, ...]
    experimental_packs: tuple[str, ...]
    benchmark_only_packs: tuple[str, ...]
    enabled_native_families: tuple[str, ...]
    disabled_native_families: tuple[str, ...]
    extension_targets: tuple[str, ...]
    editable_sync_roots: tuple[str, ...]
    cmake_defines: tuple[str, ...]


def supported_build_profiles() -> tuple[str, ...]:
    return tuple(_PROFILE_SPECS)


def supported_build_profile_csv() -> str:
    return ", ".join(supported_build_profiles())


def normalize_build_profile(value: str | None) -> str:
    normalized = (value or "full").strip().lower().replace("_", "-")
    if normalized not in _PROFILE_SPECS:
        raise ValueError(
            "Unsupported VLLM build profile "
            f"'{value}'. Supported profiles: {supported_build_profile_csv()}"
        )
    return normalized


def component_enabled(
    resolution: BuildProfileResolution, component: str
) -> bool:
    return component in resolution.enabled_components


def editable_sync_enabled(
    resolution: BuildProfileResolution, root: str
) -> bool:
    return root in resolution.editable_sync_roots


def native_family_enabled(
    resolution: BuildProfileResolution, family: str
) -> bool:
    return family in resolution.enabled_native_families


def resolve_build_profile(value: str | None) -> BuildProfileResolution:
    profile = normalize_build_profile(value)
    spec = _PROFILE_SPECS[profile]
    flags = spec["flags"]
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
    optional_backend_packs = tuple(
        component for component in enabled if component in _OPTIONAL_BACKEND_COMPONENTS
    )
    experimental_packs = tuple(
        component for component in enabled if component in _EXPERIMENTAL_COMPONENTS
    )
    benchmark_only_packs = tuple(
        component for component in enabled if component in _BENCHMARK_ONLY_COMPONENTS
    )
    enabled_native_families = tuple(spec["native_families"])
    disabled_native_families = tuple(
        family
        for family in ("base_runtime", *tuple(_NATIVE_FAMILY_FLAGS.values()))
        if family not in enabled_native_families
    )
    extension_targets = tuple(
        target
        for component in enabled
        for target in _COMPONENT_EXTENSION_TARGETS.get(component, ())
    )
    editable_sync_roots = tuple(
        root
        for component in enabled
        for root in _COMPONENT_EDITABLE_SYNC_ROOTS.get(component, ())
    )
    defines = tuple(
        f"-D{flag}={'ON' if enabled else 'OFF'}"
        for flag, enabled in flags.items()
    ) + tuple(
        f"-D{flag}={'ON' if family in enabled_native_families else 'OFF'}"
        for flag, family in _NATIVE_FAMILY_FLAGS.items()
    )
    return BuildProfileResolution(
        profile=profile,
        profile_family=spec["profile_family"],
        developer_friendly=spec["developer_friendly"],
        cuda_arches=tuple(spec["cuda_arches"]),
        enabled_components=enabled,
        disabled_components=disabled,
        optional_backend_packs=optional_backend_packs,
        experimental_packs=experimental_packs,
        benchmark_only_packs=benchmark_only_packs,
        enabled_native_families=enabled_native_families,
        disabled_native_families=disabled_native_families,
        extension_targets=extension_targets,
        editable_sync_roots=editable_sync_roots,
        cmake_defines=defines,
    )
