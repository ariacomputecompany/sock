# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_env_override_module(
    *,
    is_torch_equal=None,
    is_torch_equal_or_newer=None,
):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "vllm" / "env_override.py"

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.9.0-light"
    torch_mod._C = types.SimpleNamespace()

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )

    torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_mod.is_torch_equal = is_torch_equal or (lambda version: False)
    torch_utils_mod.is_torch_equal_or_newer = is_torch_equal_or_newer or (
        lambda version: False
    )

    sys.modules["torch"] = torch_mod
    sys.modules["vllm.logger"] = logger_mod
    sys.modules["vllm.utils.torch_utils"] = torch_utils_mod

    spec = importlib.util.spec_from_file_location(
        "vllm_env_override_light", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["vllm_env_override_light"] = module
    spec.loader.exec_module(module)
    return module


def test_patch_profile_manifest_lightweight() -> None:
    env_override = _load_env_override_module()
    manifest = env_override.patch_profile_manifest()
    empty_digest = hashlib.sha256(
        json.dumps([], separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert manifest["schema_version"] == 1
    assert manifest["torch_version"] == "2.9.0-light"
    assert manifest["obsolete_patch_count"] == 0
    assert manifest["obsolete_patch_ids"] == []
    assert manifest["compile_surface_widening_count"] == 0
    assert manifest["compile_surface_widening_patch_ids"] == []
    assert manifest["fallback_namespace_coverage"] == {
        "schema_version": 1,
        "allow_list_proxy_active": False,
        "graph_binding_rebound": False,
        "namespaces": [
            {
                "namespace": "vllm",
                "prefix": "vllm::",
                "registered_op_count": 0,
                "registered_ops_digest": empty_digest,
                "registered_ops_preview": [],
            },
            {
                "namespace": "vllm_aiter",
                "prefix": "vllm_aiter::",
                "registered_op_count": 0,
                "registered_ops_digest": empty_digest,
                "registered_ops_preview": [],
            },
        ],
    }
    patch_ids = {patch["patch_id"] for patch in manifest["patches"]}
    assert "fallback_allow_list" in patch_ids
    assert "triton_force_first_config" in patch_ids
    assert all(patch["obsolete"] is False for patch in manifest["patches"])
    assert all(patch["obsolete_reason"] is None for patch in manifest["patches"])
    assert all(
        patch["compile_surface_effect"] == "neutral"
        for patch in manifest["patches"]
    )
    assert all(
        patch["compile_surface_reason"] is None for patch in manifest["patches"]
    )


def test_patch_profile_manifest_marks_obsolete_patches_on_newer_torch() -> None:
    env_override = _load_env_override_module(
        is_torch_equal=lambda version: False,
        is_torch_equal_or_newer=lambda version: version in (
            "2.10.0",
            "2.11.0",
            "2.11.0.dev",
            "2.12.0.dev",
        ),
    )
    manifest = env_override.patch_profile_manifest()

    assert manifest["obsolete_patch_count"] == 8
    assert manifest["obsolete_patch_ids"] == [
        "torch_2_9_memory_plan_reuse",
        "torch_2_9_graph_partition_signature",
        "torch_2_9_scheduler_partition",
        "torch_2_9_get_raw_stream",
        "graph_capture_runtime_env",
        "constrain_to_fx_strides",
        "fxgraphcache_pickle",
        "cpp_indirect_assert",
    ]
    assert manifest["compile_surface_widening_count"] == 0
    assert manifest["compile_surface_widening_patch_ids"] == []

    obsolete_by_id = {
        patch["patch_id"]: patch
        for patch in manifest["patches"]
        if patch["patch_id"] in set(manifest["obsolete_patch_ids"])
    }
    assert all(patch["obsolete"] is True for patch in obsolete_by_id.values())
    assert all(
        patch["obsolete_reason"] == "torch_version_outside_patch_window"
        for patch in obsolete_by_id.values()
    )
    assert all(
        patch["compile_surface_effect"] == "neutral"
        for patch in manifest["patches"]
    )


def test_fallback_creation_evidence_manifest_tracks_hits() -> None:
    env_override = _load_env_override_module()
    proxy = env_override._VllmFallbackAllowList(set())

    assert proxy.evidence_manifest() == {
        "schema_version": 1,
        "proxy_active": True,
        "total_hit_count": 0,
        "total_unique_op_count": 0,
        "namespaces": [
            {
                "namespace": "vllm",
                "prefix": "vllm::",
                "hit_count": 0,
                "unique_op_count": 0,
                "ops_preview": [],
            },
            {
                "namespace": "vllm_aiter",
                "prefix": "vllm_aiter::",
                "hit_count": 0,
                "unique_op_count": 0,
                "ops_preview": [],
            },
        ],
    }

    assert "vllm::all_reduce" in proxy
    assert "vllm::all_reduce" in proxy
    assert "vllm_aiter::rocm_aiter_fused_moe" in proxy

    assert proxy.evidence_manifest() == {
        "schema_version": 1,
        "proxy_active": True,
        "total_hit_count": 3,
        "total_unique_op_count": 2,
        "namespaces": [
            {
                "namespace": "vllm",
                "prefix": "vllm::",
                "hit_count": 2,
                "unique_op_count": 1,
                "ops_preview": [
                    {"op_name": "vllm::all_reduce", "hit_count": 2},
                ],
            },
            {
                "namespace": "vllm_aiter",
                "prefix": "vllm_aiter::",
                "hit_count": 1,
                "unique_op_count": 1,
                "ops_preview": [
                    {
                        "op_name": "vllm_aiter::rocm_aiter_fused_moe",
                        "hit_count": 1,
                    },
                ],
            },
        ],
    }

    proxy.reset_evidence()
    assert proxy.evidence_manifest()["total_hit_count"] == 0
