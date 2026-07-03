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
    assert "VLLM_BUILD_PROFILE" in manifest["ignored_keys"]
    assert "VLLM_DISABLE_COMPILE_CACHE" in manifest["included_keys"]

    rendered = envs.render_compile_factor_manifest()
    reparsed = json.loads(rendered)
    assert reparsed["schema_version"] == manifest["schema_version"]
    assert reparsed["categories"] == manifest["categories"]
    assert reparsed["included_keys"] == manifest["included_keys"]
    assert reparsed["ignored_keys"] == manifest["ignored_keys"]
