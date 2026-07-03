# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import pickle
import sys
import types
from pathlib import Path


class _Logger:

    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _load_caching_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "vllm" / "compilation" / "caching.py"

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.9.0-light"
    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.SymInt = type("SymInt", (), {})

    torch_fx_mod = types.ModuleType("torch.fx")
    torch_fx_mod.GraphModule = type("GraphModule", (), {})
    torch_mod.fx = torch_fx_mod

    torch_graph_pickler_mod = types.ModuleType("torch.fx._graph_pickler")

    class GraphPickler:
        reducer_override = staticmethod(lambda self, obj: obj)

        @staticmethod
        def dumps(obj, options=None):
            return pickle.dumps(obj)

        @staticmethod
        def loads(data, fake_mode=None):
            return pickle.loads(data)

    class Options:

        def __init__(self, ops_filter=None):
            self.ops_filter = ops_filter

    torch_graph_pickler_mod.GraphPickler = GraphPickler
    torch_graph_pickler_mod.Options = Options

    torch_subclasses_mod = types.ModuleType("torch._subclasses")
    torch_subclasses_mod.FakeTensorMode = type("FakeTensorMode", (), {})

    torch_utils_mod = types.ModuleType("torch.utils")
    torch_pytree_mod = types.ModuleType("torch.utils._pytree")
    torch_pytree_mod.tree_map_only = lambda typ, fn, tree: tree
    torch_pytree_mod._private_register_pytree_node = lambda *args, **kwargs: None
    torch_pytree_mod._deregister_pytree_node = lambda *args, **kwargs: None
    torch_utils_mod._pytree = torch_pytree_mod

    torch_dynamo_mod = types.ModuleType("torch._dynamo")
    torch_aot_compile_mod = types.ModuleType("torch._dynamo.aot_compile")
    torch_aot_compile_mod.SerializableCallable = type(
        "SerializableCallable", (), {}
    )
    torch_dynamo_mod.aot_compile = torch_aot_compile_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.fx"] = torch_fx_mod
    sys.modules["torch.fx._graph_pickler"] = torch_graph_pickler_mod
    sys.modules["torch._subclasses"] = torch_subclasses_mod
    sys.modules["torch.utils"] = torch_utils_mod
    sys.modules["torch.utils._pytree"] = torch_pytree_mod
    sys.modules["torch._dynamo"] = torch_dynamo_mod
    sys.modules["torch._dynamo.aot_compile"] = torch_aot_compile_mod

    vllm_pkg = types.ModuleType("vllm")
    compilation_pkg = types.ModuleType("vllm.compilation")
    config_pkg = types.ModuleType("vllm.config")
    utils_pkg = types.ModuleType("vllm.utils")

    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_USE_MEGA_AOT_ARTIFACT = False
    envs_mod.compile_factors = lambda: []
    envs_mod.compile_factor_manifest = lambda: {"schema_version": 1}

    codegen_mod = types.ModuleType("vllm.compilation.codegen")
    codegen_mod.compile_execution_fn = lambda *args, **kwargs: None

    compiler_interface_mod = types.ModuleType("vllm.compilation.compiler_interface")
    compiler_interface_mod.get_inductor_factors = lambda: []

    counter_mod = types.ModuleType("vllm.compilation.counter")
    counter_mod.compilation_counter = types.SimpleNamespace(
        num_compiled_artifacts_saved=0,
        num_compiled_artifacts_loaded=0,
    )

    config_mod = types.ModuleType("vllm.config")
    config_mod.VllmConfig = type("VllmConfig", (), {"compute_hash": lambda self: "cfg"})
    config_mod.get_current_vllm_config = lambda: None

    config_utils_mod = types.ModuleType("vllm.config.utils")
    config_utils_mod.hash_factors = lambda factors: "factors"

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: _Logger()

    hashing_mod = types.ModuleType("vllm.utils.hashing")
    hashing_mod.safe_hash = lambda data, usedforsecurity=False: hashlib.sha256(data)

    sys.modules["vllm"] = vllm_pkg
    sys.modules["vllm.compilation"] = compilation_pkg
    sys.modules["vllm.compilation.codegen"] = codegen_mod
    sys.modules["vllm.compilation.compiler_interface"] = compiler_interface_mod
    sys.modules["vllm.compilation.counter"] = counter_mod
    sys.modules["vllm.config"] = config_mod
    sys.modules["vllm.config.utils"] = config_utils_mod
    sys.modules["vllm.envs"] = envs_mod
    sys.modules["vllm.logger"] = logger_mod
    sys.modules["vllm.utils"] = utils_pkg
    sys.modules["vllm.utils.hashing"] = hashing_mod

    spec = importlib.util.spec_from_file_location("vllm_caching_light", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["vllm_caching_light"] = module
    spec.loader.exec_module(module)
    return module, counter_mod.compilation_counter


def test_artifact_manifest_summary_and_identity() -> None:
    caching, counter = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()

    artifacts.insert("block0", "shape0", b"same-bytes")
    artifacts.insert("block1", "shape0", b"same-bytes")
    artifacts.insert("block2", "shape1", b"other-bytes")

    manifest = artifacts.manifest_summary()
    same_hash = caching.artifact_bytes_hash(b"same-bytes")
    other_hash = caching.artifact_bytes_hash(b"other-bytes")

    assert counter.num_compiled_artifacts_saved == 2
    assert manifest["schema_version"] == 1
    assert manifest["entry_count"] == 3
    assert manifest["unique_artifact_count"] == 2
    assert manifest["total_bytes"] == len(b"same-bytes") + len(b"other-bytes")
    assert manifest["entries"] == [
        {
            "submodule_name": "block0",
            "shape": "shape0",
            "artifact_hash": same_hash,
            "artifact_bytes": len(b"same-bytes"),
            "deduped": True,
            "reuse_reason": "content_addressed_dedup",
        },
        {
            "submodule_name": "block1",
            "shape": "shape0",
            "artifact_hash": same_hash,
            "artifact_bytes": len(b"same-bytes"),
            "deduped": True,
            "reuse_reason": "content_addressed_dedup",
        },
        {
            "submodule_name": "block2",
            "shape": "shape1",
            "artifact_hash": other_hash,
            "artifact_bytes": len(b"other-bytes"),
            "deduped": False,
            "reuse_reason": "unique_artifact",
        },
    ]
    stores_by_hash = {
        store["artifact_hash"]: {
            "artifact_bytes": store["artifact_bytes"],
            "entry_count": store["entry_count"],
        }
        for store in manifest["stores"]
    }
    assert stores_by_hash == {
        other_hash: {
            "artifact_bytes": len(b"other-bytes"),
            "entry_count": 1,
        },
        same_hash: {
            "artifact_bytes": len(b"same-bytes"),
            "entry_count": 2,
        },
    }

    rendered = json.loads(artifacts.render_manifest())
    assert rendered["store_identity"] == artifacts.store_identity()
    assert rendered["entries"] == manifest["entries"]
    assert rendered["stores"] == manifest["stores"]

    assert artifacts.reuse_summary() == {
        "schema_version": 1,
        "cache_hit_reason": "standalone_aot_artifact_manifest_match",
        "artifact_reuse_mode": "content_addressed_dedup",
        "entry_count": 3,
        "unique_artifact_count": 2,
        "deduped_entry_count": 2,
        "duplicate_entry_count": 1,
        "unique_bytes": len(b"same-bytes") + len(b"other-bytes"),
        "expanded_entry_bytes": len(b"same-bytes") * 2 + len(b"other-bytes"),
        "duplicate_bytes_elided": len(b"same-bytes"),
        "duplicate_artifact_loads_avoided": 1,
    }


def test_serialized_state_records_artifact_manifest_metadata() -> None:
    caching, _ = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    artifacts.insert("block0", "shape0", b"payload")
    artifacts.insert("block1", "shape0", b"payload")

    class _Graph:
        nodes = []

    class _GraphModule:
        graph = _Graph()

        def named_children(self):
            return []

    backend = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(compute_hash=lambda: "cfg-hash"),
        collect_standalone_compile_artifacts=lambda: (
            artifacts,
            {"block0": (0,)},
            {"block0": True},
        )
    )
    compiled_fn = types.SimpleNamespace(
        graph_module=_GraphModule(),
        example_inputs=[],
        prefix="unit-test",
        optimized_call=lambda *args, **kwargs: None,
        is_encoder=False,
        vllm_backend=backend,
        sym_tensor_indices=[],
        aot_autograd_config={},
        execution_code=None,
        submod_names=None,
        consts=None,
        shape_env=None,
        _fake_mode=None,
    )

    original_serialize_graph_module = (
        caching.VllmSerializableFunction.serialize_graph_module
    )
    caching.VllmSerializableFunction.serialize_graph_module = classmethod(
        lambda cls, graph_module: b"graph-module"
    )
    try:
        serialized = caching.VllmSerializableFunction.serialize_compile_artifacts(
            compiled_fn
        )
    finally:
        caching.VllmSerializableFunction.serialize_graph_module = (
            original_serialize_graph_module
        )
    state = pickle.loads(serialized)

    assert state["standalone_compile_artifact_manifest"] == artifacts.manifest_summary()
    assert (
        state["standalone_compile_artifact_store_identity"]
        == artifacts.store_identity()
    )
    assert state["standalone_compile_artifact_compatibility"] == {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": "2.9.0-light",
        "mega_aot_enabled": False,
        "env": {"schema_version": 1},
        "vllm_config_hash": "cfg-hash",
    }
    assert state["standalone_compile_artifact_reuse_summary"] == {
        "schema_version": 1,
        "cache_hit_reason": "standalone_aot_artifact_manifest_match",
        "artifact_reuse_mode": "content_addressed_dedup",
        "entry_count": 2,
        "unique_artifact_count": 1,
        "deduped_entry_count": 2,
        "duplicate_entry_count": 1,
        "unique_bytes": len(b"payload"),
        "expanded_entry_bytes": len(b"payload") * 2,
        "duplicate_bytes_elided": len(b"payload"),
        "duplicate_artifact_loads_avoided": 1,
    }
    assert state["sym_shape_indices_map"] == {"block0": (0,)}
    assert state["returns_tuple_map"] == {"block0": True}


def test_artifact_manifest_verification_detects_mismatch() -> None:
    caching, _ = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    artifacts.insert("block0", "shape0", b"payload-a")
    artifacts.insert("block1", "shape0", b"payload-b")

    manifest = artifacts.manifest_summary()
    manifest["store_identity"] = artifacts.store_identity()
    verified = artifacts.verify_manifest(manifest)

    assert verified["ok"] is True
    assert verified["expected_store_identity"] == artifacts.store_identity()
    assert verified["actual_store_identity"] == artifacts.store_identity()

    corrupted_manifest = json.loads(json.dumps(manifest))
    corrupted_manifest["total_bytes"] = 1
    corrupted = artifacts.verify_manifest(corrupted_manifest)

    assert corrupted["ok"] is False
    assert corrupted["expected_store_identity"] == artifacts.store_identity()
    assert corrupted["actual_store_identity"] == artifacts.store_identity()


def test_load_report_marks_already_loaded_fast_path() -> None:
    caching, _ = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    digest = caching.artifact_bytes_hash(b"payload")
    artifacts.submodule_bytes["block0_shape0"] = digest
    artifacts.submodule_bytes_store[digest] = b"payload"
    artifacts.loaded_submodule_store[digest] = object()

    artifacts.load_all()

    assert artifacts.last_load_report() == {
        "schema_version": 1,
        "load_path": "already_loaded",
        "loaded_artifact_count": 1,
        "deserialization_wall_time_ms": 0.0,
    }
