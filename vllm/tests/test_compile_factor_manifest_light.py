# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch


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
    assert manifest["categories"]["VLLM_API_KEY"] == "runtime_non_compile"
    assert manifest["categories"]["VLLM_CONFIGURE_LOGGING"] == "debug_only"
    assert manifest["categories"]["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] == "debug_only"
    assert manifest["categories"]["VLLM_SKIP_MODEL_NAME_VALIDATION"] == "runtime_non_compile"
    assert manifest["categories"]["VLLM_ENFORCE_STRICT_TOOL_CALLING"] == "runtime_non_compile"
    assert manifest["categories"]["VLLM_ENABLE_RESPONSES_API_STORE"] == "runtime_non_compile"
    assert manifest["categories"]["VLLM_LOG_MODEL_INSPECTION"] == "debug_only"
    assert manifest["categories"]["VLLM_DISABLE_LOG_LOGO"] == "debug_only"
    assert manifest["categories"]["VLLM_COMPUTE_NANS_IN_LOGITS"] == "debug_only"
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
    assert "VLLM_API_KEY" in manifest["ignored_keys"]
    assert "VLLM_CONFIGURE_LOGGING" in manifest["ignored_keys"]
    assert "VLLM_SKIP_MODEL_NAME_VALIDATION" in manifest["ignored_keys"]
    assert "VLLM_ENFORCE_STRICT_TOOL_CALLING" in manifest["ignored_keys"]
    assert "VLLM_LOG_MODEL_INSPECTION" in manifest["ignored_keys"]
    assert "VLLM_COMPUTE_NANS_IN_LOGITS" in manifest["ignored_keys"]
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
    assert manifest["validation"]["overlap_keys"] == {}
    assert manifest["identity"]["schema_version"] == 1
    assert manifest["identity"]["declared_factor_count"] > 0
    assert manifest["identity"]["ambient_factor_count"] > 0
    assert manifest["audit"]["category_counts"]["debug_only"] >= 1
    assert manifest["audit"]["category_counts"]["runtime_non_compile"] >= 1
    assert manifest["audit"]["overlap_keys"] == {}
    assert (
        manifest["normalization"]["declared_factor_normalization"][
            "VLLM_DISABLED_KERNELS"
        ]["normalizer"]
        == "_normalize_unordered_string_list_compile_factor"
    )
    assert (
        manifest["normalization"]["ambient_factor_normalization"][
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
        ]["normalizer"]
        == "_normalize_boolean_toggle_compile_factor"
    )
    assert (
        "VLLM_DISABLED_KERNELS"
        not in manifest["normalization"]["declared_raw_normalization"]
    )
    assert (
        manifest["normalization"]["declared_factor_normalization"][
            "VLLM_DISABLE_COMPILE_CACHE"
        ]["strategy"]
        == "raw"
    )
    assert manifest["validation"]["missing_normalization_policy_keys"] == []
    assert manifest["validation"]["extra_normalization_policy_keys"] == []
    assert manifest["validation"]["invalid_normalizer_keys"] == []
    assert (
        "VLLM_DISABLE_COMPILE_CACHE"
        in manifest["normalization"]["declared_raw_normalization"]
    )
    assert "VLLM_CONFIGURE_LOGGING" in manifest["audit"]["category_keys"]["debug_only"]
    assert "VLLM_API_KEY" in manifest["audit"]["category_keys"]["runtime_non_compile"]
    assert "VLLM_LOG_MODEL_INSPECTION" in manifest["audit"]["category_keys"]["debug_only"]
    assert (
        "VLLM_ENFORCE_STRICT_TOOL_CALLING"
        in manifest["audit"]["category_keys"]["runtime_non_compile"]
    )

    rendered = envs.render_compile_factor_manifest()
    reparsed = json.loads(rendered)
    assert reparsed["schema_version"] == manifest["schema_version"]
    assert reparsed["categories"] == manifest["categories"]
    assert reparsed["policies"] == manifest["policies"]
    assert reparsed["ambient_policies"] == manifest["ambient_policies"]
    assert reparsed["normalization"] == manifest["normalization"]
    assert reparsed["included_keys"] == manifest["included_keys"]
    assert reparsed["ambient_included_keys"] == manifest["ambient_included_keys"]
    assert reparsed["ignored_keys"] == manifest["ignored_keys"]
    assert reparsed["audit"] == manifest["audit"]
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


def test_compile_factor_policy_detects_missing_normalizer_registration() -> None:
    envs = _load_envs_module()
    original = envs._compile_factor_normalization_policy

    def bad_policy(factor: str):
        policy = original(factor)
        if factor == "VLLM_DISABLED_KERNELS":
            return {
                **policy,
                "normalizer": "_missing_normalizer",
            }
        return policy

    envs._compile_factor_normalization_policy = bad_policy
    try:
        validation = envs.validate_compile_factor_policy(hard_fail=False)
        assert validation["ok"] is False
        assert "compile_factor_normalizer_missing" in validation["reasons"]
        assert validation["invalid_normalizer_keys"] == ["VLLM_DISABLED_KERNELS"]
    finally:
        envs._compile_factor_normalization_policy = original


def test_case_insensitive_choice_envs_canonicalize_for_identity() -> None:
    envs = _load_envs_module()

    with patch.dict(
        os.environ,
        {
            "VLLM_FLOAT32_MATMUL_PRECISION": "HIGHEST",
            "VLLM_BUILD_PROFILE": "MINIMAL-DEV",
        },
        clear=False,
    ):
        factors = envs.compile_factors()
        manifest = envs.compile_factor_manifest()
        build_profile = envs.environment_variables["VLLM_BUILD_PROFILE"]()

    assert factors["VLLM_FLOAT32_MATMUL_PRECISION"] == "highest"
    assert manifest["categories"]["VLLM_BUILD_PROFILE"] == "host_only"
    assert "VLLM_BUILD_PROFILE" not in factors
    assert build_profile == "minimal-dev"


def test_case_insensitive_choice_helpers_return_canonical_spelling() -> None:
    envs = _load_envs_module()

    with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}, clear=False):
        env_func = envs.env_with_choices(
            "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
        )
        assert env_func() == "option1"

    with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}, clear=False):
        env_list_func = envs.env_list_with_choices(
            "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
        )
        assert env_list_func() == ["option1", "option2"]

    with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}, clear=False):
        env_set_func = envs.env_set_with_choices(
            "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
        )
        assert env_set_func() == {"option1", "option2"}


def test_disabled_kernels_compile_factor_is_set_canonicalized() -> None:
    envs = _load_envs_module()

    with patch.dict(
        os.environ,
        {"VLLM_DISABLED_KERNELS": "MarlinLinearKernel, ExllamaLinearKernel,MarlinLinearKernel"},
        clear=False,
    ):
        factors = envs.compile_factors()
        identity = envs.compile_factor_identity_manifest()

    with patch.dict(
        os.environ,
        {"VLLM_DISABLED_KERNELS": " ExllamaLinearKernel ,MarlinLinearKernel "},
        clear=False,
    ):
        equivalent_factors = envs.compile_factors()
        equivalent_identity = envs.compile_factor_identity_manifest()

    assert factors["VLLM_DISABLED_KERNELS"] == (
        "ExllamaLinearKernel",
        "MarlinLinearKernel",
    )
    assert equivalent_factors["VLLM_DISABLED_KERNELS"] == (
        "ExllamaLinearKernel",
        "MarlinLinearKernel",
    )
    assert (
        identity["combined_factor_digest"]
        == equivalent_identity["combined_factor_digest"]
    )


def test_ambient_boolean_compile_factors_are_canonicalized() -> None:
    envs = _load_envs_module()

    with patch.dict(
        os.environ,
        {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "TRUE"},
        clear=False,
    ):
        factors = envs.compile_factors()
        identity = envs.compile_factor_identity_manifest()

    with patch.dict(
        os.environ,
        {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"},
        clear=False,
    ):
        equivalent_factors = envs.compile_factors()
        equivalent_identity = envs.compile_factor_identity_manifest()

    with patch.dict(
        os.environ,
        {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "false"},
        clear=False,
    ):
        disabled_factors = envs.compile_factors()

    assert factors["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] is True
    assert equivalent_factors["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] is True
    assert disabled_factors["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] is False
    assert (
        identity["combined_factor_digest"]
        == equivalent_identity["combined_factor_digest"]
    )
