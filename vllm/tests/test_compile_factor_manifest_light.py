# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
import types
from pathlib import Path


def _normalize_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, bytearray):
        return bytes(value).hex()
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalize_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_normalize_value(v) for v in value))
    return str(value)


def _load_envs_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "vllm" / "envs.py"

    vllm_pkg = types.ModuleType("vllm")
    config_pkg = types.ModuleType("vllm.config")
    utils_mod = types.ModuleType("vllm.config.utils")
    utils_mod.normalize_value = _normalize_value

    sys.modules["vllm"] = vllm_pkg
    sys.modules["vllm.config"] = config_pkg
    sys.modules["vllm.config.utils"] = utils_mod

    spec = importlib.util.spec_from_file_location("vllm_envs_light", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compile_factor_manifest_lightweight() -> None:
    envs = _load_envs_module()
    manifest = envs.compile_factor_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["categories"]["VLLM_BUILD_PROFILE"] == "host_only"
    assert manifest["categories"]["VLLM_CACHE_ROOT"] == "cache_location_only"
    assert manifest["categories"]["VLLM_DISABLE_COMPILE_CACHE"] == "compile_affecting"
    assert (
        manifest["policies"]["VLLM_DISABLE_COMPILE_CACHE"]["included_in_compile_key"]
        is True
    )
    assert "compile identity" in manifest["policies"]["VLLM_BUILD_PROFILE"]["reason"] or (
        "must not change shared compile identity"
        in manifest["policies"]["VLLM_BUILD_PROFILE"]["reason"]
    )
    assert "VLLM_BUILD_PROFILE" in manifest["ignored_keys"]
    assert "VLLM_DISABLE_COMPILE_CACHE" in manifest["included_keys"]
    assert (
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
        in manifest["ambient_included_keys"]
    )
    assert (
        manifest["ambient_policies"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"][
            "source"
        ]
        == "ambient_environment"
    )
    assert manifest["validation"]["ok"] is True
    assert (
        manifest["validation"]["compile_affecting_key_digest"]
        == envs._EXPECTED_COMPILE_AFFECTING_ENV_VARS_DIGEST
    )
    assert manifest["identity"]["schema_version"] == 1
    assert manifest["identity"]["declared_factor_count"] > 0
    assert manifest["identity"]["ambient_factor_count"] > 0

    rendered = envs.render_compile_factor_manifest()
    reparsed = json.loads(rendered)
    assert reparsed["schema_version"] == manifest["schema_version"]
    assert reparsed["categories"] == manifest["categories"]
    assert reparsed["policies"] == manifest["policies"]
    assert reparsed["ambient_policies"] == manifest["ambient_policies"]
    assert reparsed["included_keys"] == manifest["included_keys"]
    assert reparsed["ambient_included_keys"] == manifest["ambient_included_keys"]
    assert reparsed["ignored_keys"] == manifest["ignored_keys"]
    assert reparsed["validation"] == manifest["validation"]

    rendered_identity = envs.render_compile_factor_identity_manifest()
    reparsed_identity = json.loads(rendered_identity)
    assert reparsed_identity["schema_version"] == manifest["identity"]["schema_version"]
    assert (
        reparsed_identity["combined_factor_digest"]
        == manifest["identity"]["combined_factor_digest"]
    )
    assert (
        reparsed_identity["declared_factor_digest"]
        == manifest["identity"]["declared_factor_digest"]
    )
    assert (
        reparsed_identity["ambient_factor_digest"]
        == manifest["identity"]["ambient_factor_digest"]
    )


def test_compile_factor_policy_detects_unexpected_compile_affecting_set() -> None:
    envs = _load_envs_module()
    original_digest = envs._EXPECTED_COMPILE_AFFECTING_ENV_VARS_DIGEST
    envs._EXPECTED_COMPILE_AFFECTING_ENV_VARS_DIGEST = "bad-digest"
    try:
        validation = envs.validate_compile_factor_policy(hard_fail=False)
        assert validation["ok"] is False
        assert validation["reasons"] == ["compile_affecting_env_var_set_changed"]
    finally:
        envs._EXPECTED_COMPILE_AFFECTING_ENV_VARS_DIGEST = original_digest
