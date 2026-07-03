# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import pickle
import sys
import tempfile
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

    torch_inductor_mod = types.ModuleType("torch._inductor")
    torch_standalone_compile_mod = types.ModuleType(
        "torch._inductor.standalone_compile"
    )

    class AOTCompiledArtifact:

        deserialize = staticmethod(lambda entry: {"deserialized": entry})

    torch_standalone_compile_mod.AOTCompiledArtifact = AOTCompiledArtifact
    torch_inductor_mod.standalone_compile = torch_standalone_compile_mod

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
    sys.modules["torch._inductor"] = torch_inductor_mod
    sys.modules["torch._inductor.standalone_compile"] = (
        torch_standalone_compile_mod
    )
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

    env_override_mod = types.ModuleType("vllm.env_override")
    env_override_mod.patch_profile_manifest = lambda: {
        "schema_version": 1,
        "torch_version": "2.9.0-light",
        "fallback_namespace_coverage": {
            "schema_version": 1,
            "allow_list_proxy_active": True,
            "graph_binding_rebound": True,
            "namespaces": [
                {
                    "namespace": "vllm",
                    "prefix": "vllm::",
                    "registered_op_count": 2,
                    "registered_ops_digest": "vllm-digest",
                    "registered_ops_preview": [
                        "vllm::all_reduce",
                        "vllm::fused_add_rms_norm",
                    ],
                },
                {
                    "namespace": "vllm_aiter",
                    "prefix": "vllm_aiter::",
                    "registered_op_count": 1,
                    "registered_ops_digest": "aiter-digest",
                    "registered_ops_preview": [
                        "vllm_aiter::rocm_aiter_fused_moe",
                    ],
                },
            ],
        },
        "obsolete_patch_count": 0,
        "obsolete_patch_ids": [],
        "compile_surface_widening_count": 0,
        "compile_surface_widening_patch_ids": [],
        "patches": [
            {
                "patch_id": "lightweight-stub",
                "category": "correctness_patch",
                "eligible": True,
                "applied": True,
                "detail": "stubbed patch profile",
                "obsolete": False,
                "obsolete_reason": None,
                "compile_surface_effect": "neutral",
                "compile_surface_reason": None,
            }
        ],
    }
    env_override_mod.fallback_namespace_manifest = lambda: {
        "schema_version": 1,
        "allow_list_proxy_active": True,
        "graph_binding_rebound": True,
        "namespaces": [
            {
                "namespace": "vllm",
                "prefix": "vllm::",
                "registered_op_count": 2,
                "registered_ops_digest": "vllm-digest",
                "registered_ops_preview": [
                    "vllm::all_reduce",
                    "vllm::fused_add_rms_norm",
                ],
            },
            {
                "namespace": "vllm_aiter",
                "prefix": "vllm_aiter::",
                "registered_op_count": 1,
                "registered_ops_digest": "aiter-digest",
                "registered_ops_preview": [
                    "vllm_aiter::rocm_aiter_fused_moe",
                ],
            },
        ],
    }
    env_override_mod.fallback_creation_evidence_manifest = lambda: {
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
    sys.modules["vllm.env_override"] = env_override_mod
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

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(compute_hash=lambda: "cfg-hash"),
            compilation_config=types.SimpleNamespace(local_cache_dir=tmpdir),
            compiler_manager=types.SimpleNamespace(
                compiler=types.SimpleNamespace(name="inductor-light"),
                compute_hash=lambda cfg: "compiler-hash",
            ),
            collect_standalone_compile_artifacts=lambda: (
                artifacts,
                {"block0": (0,)},
                {"block0": True},
            ),
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
    sidecar, payload = caching.unpack_serialized_compile_artifact_bundle(serialized)
    state = caching.unpack_serialized_fn_state_bundle(payload)

    assert sidecar == {
        "schema_version": 1,
        "payload_kind": "vllm_standalone_compile_artifact_sidecar",
        "artifact_manifest": artifacts.manifest_summary(),
        "store_identity": artifacts.store_identity(),
        "compatibility": {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "torch_version": "2.9.0-light",
            "mega_aot_enabled": False,
            "env": {"schema_version": 1},
            "vllm_config_hash": "cfg-hash",
        },
        "reuse_summary": {
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
        },
        "proof_manifest": {
            "schema_version": 1,
            "compile_hashes": {
                "env_policy_hash": "factors",
                "config_hash": "cfg-hash",
                "code_hash": None,
                "compiler_hash": "compiler-hash",
            },
            "no_new_compile_expectation": {
                "schema_version": 1,
                "expected_new_compiled_artifacts": 0,
                "proof_mode": "standalone_aot_artifact_reuse",
            },
            "backend_identity": {
                "backend_class": "SimpleNamespace",
                "prefix": None,
                "is_encoder": False,
                "compiler_name": "inductor-light",
            },
            "toolchain_identity": {
                "python_version": ".".join(str(part) for part in sys.version_info[:3]),
                "torch_version": "2.9.0-light",
            },
            "patch_profile": {
                "schema_version": 1,
                "torch_version": "2.9.0-light",
                "fallback_namespace_coverage": {
                    "schema_version": 1,
                    "allow_list_proxy_active": True,
                    "graph_binding_rebound": True,
                    "namespaces": [
                        {
                            "namespace": "vllm",
                            "prefix": "vllm::",
                            "registered_op_count": 2,
                            "registered_ops_digest": "vllm-digest",
                            "registered_ops_preview": [
                                "vllm::all_reduce",
                                "vllm::fused_add_rms_norm",
                            ],
                        },
                        {
                            "namespace": "vllm_aiter",
                            "prefix": "vllm_aiter::",
                            "registered_op_count": 1,
                            "registered_ops_digest": "aiter-digest",
                            "registered_ops_preview": [
                                "vllm_aiter::rocm_aiter_fused_moe",
                            ],
                        },
                    ],
                },
                "obsolete_patch_count": 0,
                "obsolete_patch_ids": [],
                "compile_surface_widening_count": 0,
                "compile_surface_widening_patch_ids": [],
                "patches": [
                    {
                        "patch_id": "lightweight-stub",
                        "category": "correctness_patch",
                        "eligible": True,
                        "applied": True,
                        "detail": "stubbed patch profile",
                        "obsolete": False,
                        "obsolete_reason": None,
                        "compile_surface_effect": "neutral",
                        "compile_surface_reason": None,
                    }
                ],
            },
            "fallback_namespace_coverage": {
                "schema_version": 1,
                "allow_list_proxy_active": True,
                "graph_binding_rebound": True,
                "namespaces": [
                    {
                        "namespace": "vllm",
                        "prefix": "vllm::",
                        "registered_op_count": 2,
                        "registered_ops_digest": "vllm-digest",
                        "registered_ops_preview": [
                            "vllm::all_reduce",
                            "vllm::fused_add_rms_norm",
                        ],
                    },
                    {
                        "namespace": "vllm_aiter",
                        "prefix": "vllm_aiter::",
                        "registered_op_count": 1,
                        "registered_ops_digest": "aiter-digest",
                        "registered_ops_preview": [
                            "vllm_aiter::rocm_aiter_fused_moe",
                        ],
                    },
                ],
            },
            "fallback_creation_evidence": {
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
            },
            "shape_envelope": {
                "schema_version": 1,
                "submodule_count": 2,
                "total_shape_variants": 2,
                "submodules": [
                    {
                        "submodule_name": "block0",
                        "shape_variants": ["shape0"],
                        "shape_count": 1,
                        "symbolic_input_positions": [0],
                        "returns_tuple": True,
                    },
                    {
                        "submodule_name": "block1",
                        "shape_variants": ["shape0"],
                        "shape_count": 1,
                        "symbolic_input_positions": [],
                        "returns_tuple": False,
                    },
                ],
            },
        },
        "sym_shape_indices_map": {"block0": [0]},
        "returns_tuple_map": {"block0": True},
        "example_input_tensor_specs": {
            "schema_version": 1,
            "input_count": 0,
            "indexed_tensors": [],
        },
    }
    assert (
        caching.unpack_standalone_artifact_store_bundle(
            state["standalone_compile_artifact_store_bundle"]
        ).manifest_summary()
        == artifacts.manifest_summary()
    )
    assert "standalone_compile_artifacts" not in state
    assert "standalone_compile_artifact_manifest" not in state
    assert "standalone_compile_artifact_store_identity" not in state
    assert "standalone_compile_artifact_compatibility" not in state
    assert "standalone_compile_artifact_reuse_summary" not in state
    assert "standalone_compile_artifact_proof_manifest" not in state
    assert "sym_shape_indices_map" not in state
    assert "returns_tuple_map" not in state
    assert sidecar["compatibility"] == {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": "2.9.0-light",
        "mega_aot_enabled": False,
        "env": {"schema_version": 1},
        "vllm_config_hash": "cfg-hash",
    }
    assert sidecar["reuse_summary"] == {
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
    assert sidecar["proof_manifest"] == {
        "schema_version": 1,
        "compile_hashes": {
            "env_policy_hash": "factors",
            "config_hash": "cfg-hash",
            "code_hash": None,
            "compiler_hash": "compiler-hash",
        },
        "no_new_compile_expectation": {
            "schema_version": 1,
            "expected_new_compiled_artifacts": 0,
            "proof_mode": "standalone_aot_artifact_reuse",
        },
        "backend_identity": {
            "backend_class": "SimpleNamespace",
            "prefix": None,
            "is_encoder": False,
            "compiler_name": "inductor-light",
        },
        "toolchain_identity": {
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "torch_version": "2.9.0-light",
        },
        "patch_profile": {
            "schema_version": 1,
            "torch_version": "2.9.0-light",
            "fallback_namespace_coverage": {
                "schema_version": 1,
                "allow_list_proxy_active": True,
                "graph_binding_rebound": True,
                "namespaces": [
                    {
                        "namespace": "vllm",
                        "prefix": "vllm::",
                        "registered_op_count": 2,
                        "registered_ops_digest": "vllm-digest",
                        "registered_ops_preview": [
                            "vllm::all_reduce",
                            "vllm::fused_add_rms_norm",
                        ],
                    },
                    {
                        "namespace": "vllm_aiter",
                        "prefix": "vllm_aiter::",
                        "registered_op_count": 1,
                        "registered_ops_digest": "aiter-digest",
                        "registered_ops_preview": [
                            "vllm_aiter::rocm_aiter_fused_moe",
                        ],
                    },
                ],
            },
            "obsolete_patch_count": 0,
            "obsolete_patch_ids": [],
            "compile_surface_widening_count": 0,
            "compile_surface_widening_patch_ids": [],
            "patches": [
                {
                    "patch_id": "lightweight-stub",
                    "category": "correctness_patch",
                    "eligible": True,
                    "applied": True,
                    "detail": "stubbed patch profile",
                    "obsolete": False,
                    "obsolete_reason": None,
                    "compile_surface_effect": "neutral",
                    "compile_surface_reason": None,
                }
            ],
        },
        "fallback_namespace_coverage": {
            "schema_version": 1,
            "allow_list_proxy_active": True,
            "graph_binding_rebound": True,
            "namespaces": [
                {
                    "namespace": "vllm",
                    "prefix": "vllm::",
                    "registered_op_count": 2,
                    "registered_ops_digest": "vllm-digest",
                    "registered_ops_preview": [
                        "vllm::all_reduce",
                        "vllm::fused_add_rms_norm",
                    ],
                },
                {
                    "namespace": "vllm_aiter",
                    "prefix": "vllm_aiter::",
                    "registered_op_count": 1,
                    "registered_ops_digest": "aiter-digest",
                    "registered_ops_preview": [
                        "vllm_aiter::rocm_aiter_fused_moe",
                    ],
                },
            ],
        },
        "fallback_creation_evidence": {
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
        },
            "shape_envelope": {
                "schema_version": 1,
                "submodule_count": 2,
                "total_shape_variants": 2,
                "submodules": [
                    {
                        "submodule_name": "block0",
                        "shape_variants": ["shape0"],
                        "shape_count": 1,
                        "symbolic_input_positions": [0],
                        "returns_tuple": True,
                    },
                    {
                        "submodule_name": "block1",
                        "shape_variants": ["shape0"],
                        "shape_count": 1,
                        "symbolic_input_positions": [],
                        "returns_tuple": False,
                    },
                ],
            },
        }


def test_artifact_manifest_verification_detects_mismatch() -> None:
    caching, _ = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    artifacts.insert("block0", "shape0", b"payload-a")
    artifacts.insert("block1", "shape0", b"payload-b")

    manifest = artifacts.manifest_summary()
    manifest["store_identity"] = artifacts.store_identity()
    verified = artifacts.verify_manifest(manifest)

    assert verified["ok"] is True
    assert verified["reasons"] == []
    assert verified["expected_store_identity"] == artifacts.store_identity()
    assert verified["actual_store_identity"] == artifacts.store_identity()

    corrupted_manifest = json.loads(json.dumps(manifest))
    corrupted_manifest["total_bytes"] = 1
    corrupted = artifacts.verify_manifest(corrupted_manifest)

    assert corrupted["ok"] is False
    assert corrupted["reasons"] == ["total_bytes_mismatch"]
    assert corrupted["expected_store_identity"] == artifacts.store_identity()
    assert corrupted["actual_store_identity"] == artifacts.store_identity()


def test_compatibility_drift_explains_mismatches() -> None:
    caching, _ = _load_caching_module()

    expected = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "python_version": "3.13.0",
        "torch_version": "2.9.0-light",
        "mega_aot_enabled": False,
        "env": {"schema_version": 1},
        "vllm_config_hash": "cfg-a",
    }
    actual = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "python_version": "3.13.1",
        "torch_version": "2.9.0-light",
        "mega_aot_enabled": True,
        "env": {"schema_version": 2},
        "vllm_config_hash": "cfg-b",
    }

    drift = caching.explain_compatibility_drift(expected, actual)
    assert drift == {
        "ok": False,
        "reasons": [
            "compatibility_env_mismatch",
            "compatibility_mega_aot_enabled_mismatch",
            "compatibility_python_version_mismatch",
            "compatibility_vllm_config_hash_mismatch",
        ],
        "mismatches": [
            "env",
            "mega_aot_enabled",
            "python_version",
            "vllm_config_hash",
        ],
    }


def test_startup_closure_summary_classifies_outcomes() -> None:
    caching, _ = _load_caching_module()

    full = caching.summarize_startup_closure(
        manifest_verification={"ok": True, "reasons": []},
        compatibility_drift={"ok": True, "reasons": [], "mismatches": []},
        load_report={
            "schema_version": 1,
            "load_path": "fresh_deserialize",
            "loaded_artifact_count": 2,
            "deserialization_wall_time_ms": 1.25,
        },
        assumes_closure=False,
    )
    assert full == {
        "schema_version": 1,
        "status": "full_compile_closure",
        "reasons": [],
    }

    partial = caching.summarize_startup_closure(
        manifest_verification={"ok": True, "reasons": []},
        compatibility_drift={
            "ok": False,
            "reasons": ["compatibility_env_mismatch"],
            "mismatches": ["env"],
        },
        load_report={
            "schema_version": 1,
            "load_path": "already_loaded",
            "loaded_artifact_count": 2,
            "deserialization_wall_time_ms": 0.0,
        },
        assumes_closure=False,
    )
    assert partial == {
        "schema_version": 1,
        "status": "partial_compile_closure",
        "reasons": [
            "compatibility_env_mismatch",
            "artifact_store_preloaded",
        ],
    }

    assumed = caching.summarize_startup_closure(
        manifest_verification=None,
        compatibility_drift=None,
        load_report=None,
        assumes_closure=True,
    )
    assert assumed == {
        "schema_version": 1,
        "status": "closure_by_assumption",
        "reasons": ["closure_not_proven_by_manifest"],
    }


def test_proof_manifest_uses_cache_key_factors_when_available() -> None:
    caching, _ = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    artifacts.insert("block0", "shape0", b"payload")
    artifacts.insert("block0", "shape1", b"payload-2")

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "cache_key_factors.json"
        meta_path.write_text(
            json.dumps(
                {
                    "env": {"A": "B"},
                    "config_hash": "cfg-from-file",
                    "code_hash": "code-from-file",
                    "compiler_hash": "compiler-from-file",
                }
            )
        )
        backend = types.SimpleNamespace(
            prefix="unit-prefix",
            is_encoder=True,
            vllm_config=types.SimpleNamespace(compute_hash=lambda: "cfg-live"),
            compilation_config=types.SimpleNamespace(local_cache_dir=tmpdir),
            compiler_manager=types.SimpleNamespace(
                compiler=types.SimpleNamespace(name="inductor-light"),
                compute_hash=lambda cfg: "compiler-live",
            ),
        )

        proof = caching.build_standalone_artifact_proof_manifest(
            backend,
            artifacts,
            {"block0": [0, 2]},
            {"block0": True},
        )

    assert proof == {
        "schema_version": 1,
        "compile_hashes": {
            "env_policy_hash": "factors",
            "config_hash": "cfg-from-file",
            "code_hash": "code-from-file",
            "compiler_hash": "compiler-from-file",
        },
        "no_new_compile_expectation": {
            "schema_version": 1,
            "expected_new_compiled_artifacts": 0,
            "proof_mode": "standalone_aot_artifact_reuse",
        },
        "backend_identity": {
            "backend_class": "SimpleNamespace",
            "prefix": "unit-prefix",
            "is_encoder": True,
            "compiler_name": "inductor-light",
        },
        "toolchain_identity": {
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "torch_version": "2.9.0-light",
        },
        "patch_profile": {
            "schema_version": 1,
            "torch_version": "2.9.0-light",
            "fallback_namespace_coverage": {
                "schema_version": 1,
                "allow_list_proxy_active": True,
                "graph_binding_rebound": True,
                "namespaces": [
                    {
                        "namespace": "vllm",
                        "prefix": "vllm::",
                        "registered_op_count": 2,
                        "registered_ops_digest": "vllm-digest",
                        "registered_ops_preview": [
                            "vllm::all_reduce",
                            "vllm::fused_add_rms_norm",
                        ],
                    },
                    {
                        "namespace": "vllm_aiter",
                        "prefix": "vllm_aiter::",
                        "registered_op_count": 1,
                        "registered_ops_digest": "aiter-digest",
                        "registered_ops_preview": [
                            "vllm_aiter::rocm_aiter_fused_moe",
                        ],
                    },
                ],
            },
            "obsolete_patch_count": 0,
            "obsolete_patch_ids": [],
            "compile_surface_widening_count": 0,
            "compile_surface_widening_patch_ids": [],
            "patches": [
                {
                    "patch_id": "lightweight-stub",
                    "category": "correctness_patch",
                    "eligible": True,
                    "applied": True,
                    "detail": "stubbed patch profile",
                    "obsolete": False,
                    "obsolete_reason": None,
                    "compile_surface_effect": "neutral",
                    "compile_surface_reason": None,
                }
            ],
        },
        "fallback_namespace_coverage": {
            "schema_version": 1,
            "allow_list_proxy_active": True,
            "graph_binding_rebound": True,
            "namespaces": [
                {
                    "namespace": "vllm",
                    "prefix": "vllm::",
                    "registered_op_count": 2,
                    "registered_ops_digest": "vllm-digest",
                    "registered_ops_preview": [
                        "vllm::all_reduce",
                        "vllm::fused_add_rms_norm",
                    ],
                },
                {
                    "namespace": "vllm_aiter",
                    "prefix": "vllm_aiter::",
                    "registered_op_count": 1,
                    "registered_ops_digest": "aiter-digest",
                    "registered_ops_preview": [
                        "vllm_aiter::rocm_aiter_fused_moe",
                    ],
                },
            ],
        },
        "fallback_creation_evidence": {
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
        },
        "shape_envelope": {
            "schema_version": 1,
            "submodule_count": 1,
            "total_shape_variants": 2,
            "submodules": [
                {
                    "submodule_name": "block0",
                    "shape_variants": ("shape0", "shape1"),
                    "shape_count": 2,
                    "symbolic_input_positions": (0, 2),
                    "returns_tuple": True,
                }
            ],
        },
    }


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
        "target_artifact_count": 1,
        "fresh_deserialize_count": 0,
        "shared_reuse_count": 0,
        "already_loaded_count": 0,
        "deserialization_wall_time_ms": 0.0,
        "store_identity": artifacts.store_identity(),
    }


def test_get_loaded_materializes_only_requested_artifact() -> None:
    caching, counter = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    payload0 = pickle.dumps({"artifact": "payload0"})
    payload1 = pickle.dumps({"artifact": "payload1"})
    digest0 = caching.artifact_bytes_hash(payload0)
    digest1 = caching.artifact_bytes_hash(payload1)
    artifacts.submodule_bytes["block0_shape0"] = digest0
    artifacts.submodule_bytes["block1_shape0"] = digest1
    artifacts.submodule_bytes_store[digest0] = payload0
    artifacts.submodule_bytes_store[digest1] = payload1

    loaded = artifacts.get_loaded("block0", "shape0")
    report = artifacts.last_load_report()

    assert loaded == {"deserialized": {"artifact": "payload0"}}
    assert counter.num_compiled_artifacts_loaded == 1
    assert set(artifacts.loaded_submodule_store) == {digest0}
    assert report == {
        "schema_version": 1,
        "load_path": "fresh_deserialize",
        "loaded_artifact_count": 1,
        "target_artifact_count": 2,
        "fresh_deserialize_count": 1,
        "shared_reuse_count": 0,
        "already_loaded_count": 0,
        "deserialization_wall_time_ms": report["deserialization_wall_time_ms"],
        "store_identity": artifacts.store_identity(),
    }
    assert float(report["deserialization_wall_time_ms"]) >= 0.0


def test_lazy_loaded_artifact_defers_materialization_until_call() -> None:
    caching, counter = _load_caching_module()
    payload = pickle.dumps({"artifact": "payload"})
    digest = caching.artifact_bytes_hash(payload)

    artifacts = caching.StandaloneCompiledArtifacts()
    artifacts.submodule_bytes["block0_shape0"] = digest
    artifacts.submodule_bytes_store[digest] = payload
    artifacts.mark_deferred_materialization()

    class _CallableArtifact:
        def __call__(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    sys.modules[
        "torch._inductor.standalone_compile"
    ].AOTCompiledArtifact.deserialize = staticmethod(lambda entry: _CallableArtifact())

    lazy_artifact = artifacts.build_lazy_loaded_artifact(digest)
    report_before = artifacts.last_load_report()

    assert counter.num_compiled_artifacts_loaded == 0
    assert report_before["load_path"] == "deferred_materialization"
    assert lazy_artifact(1, token=2) == {"args": (1,), "kwargs": {"token": 2}}
    assert counter.num_compiled_artifacts_loaded == 1
    assert artifacts.last_load_report()["load_path"] == "fresh_deserialize"


def test_load_report_reuses_shared_loaded_store_without_deserializing() -> None:
    caching, counter = _load_caching_module()
    first = caching.StandaloneCompiledArtifacts()
    payload = pickle.dumps({"artifact": "payload"})
    digest = caching.artifact_bytes_hash(payload)
    first.submodule_bytes["block0_shape0"] = digest
    first.submodule_bytes_store[digest] = payload

    first.load_all()

    assert first.last_load_report()["load_path"] == "fresh_deserialize"
    assert counter.num_compiled_artifacts_loaded == 1

    deserialize_calls = 0

    def _fail_deserialize(entry):
        nonlocal deserialize_calls
        deserialize_calls += 1
        raise AssertionError("shared load path should not deserialize again")

    sys.modules[
        "torch._inductor.standalone_compile"
    ].AOTCompiledArtifact.deserialize = staticmethod(_fail_deserialize)

    second = caching.StandaloneCompiledArtifacts()
    second.submodule_bytes["block0_shape0"] = digest
    second.submodule_bytes_store[digest] = payload
    second.load_all()

    assert second.last_load_report() == {
        "schema_version": 1,
        "load_path": "shared_loaded_store",
        "loaded_artifact_count": 1,
        "target_artifact_count": 1,
        "fresh_deserialize_count": 0,
        "shared_reuse_count": 1,
        "already_loaded_count": 0,
        "deserialization_wall_time_ms": 0.0,
        "store_identity": second.store_identity(),
    }
    assert deserialize_calls == 0
    assert counter.num_compiled_artifacts_loaded == 1
    assert second.get_loaded("block0", "shape0") == first.get_loaded(
        "block0", "shape0"
    )


def test_compile_artifact_bundle_passthrough_without_sidecar() -> None:
    caching, _ = _load_caching_module()

    payload = b"legacy-pickle-payload"

    assert caching.pack_serialized_compile_artifact_bundle(payload, None) == payload
    assert caching.unpack_serialized_compile_artifact_bundle(payload) == (None, payload)


def test_serialized_fn_state_bundle_roundtrip() -> None:
    caching, _ = _load_caching_module()
    state = {
        "graph_module": b"graph-bytes",
        "example_inputs": b"example-input-bytes",
        "prefix": "unit-prefix",
        "is_encoder": True,
        "sym_tensor_indices": [0, 2],
        "aot_autograd_config": {"bundled_autograd_cache": True},
        "execution_code": "return x",
        "submod_names": ["block0", "block1"],
        "consts": ["const0"],
        "standalone_compile_artifact_store_bundle": b"artifact-store",
    }

    bundle = caching.pack_serialized_fn_state_bundle(state)
    restored = caching.unpack_serialized_fn_state_bundle(bundle)

    assert restored == state


def test_standalone_artifact_store_bundle_roundtrip() -> None:
    caching, _ = _load_caching_module()
    artifacts = caching.StandaloneCompiledArtifacts()
    artifacts.insert("block0", "shape0", b"payload0")
    artifacts.insert("block1", "shape1", b"payload1")

    bundle = caching.pack_standalone_artifact_store_bundle(artifacts)
    restored = caching.unpack_standalone_artifact_store_bundle(bundle)

    assert restored.manifest_summary() == artifacts.manifest_summary()
    assert restored.store_identity() == artifacts.store_identity()
    assert restored.get("block0", "shape0") == b"payload0"
    assert restored.get("block1", "shape1") == b"payload1"


def test_no_new_compile_verification_tracks_counter_deltas() -> None:
    caching, _ = _load_caching_module()

    verification = caching.verify_no_new_compile(
        {
            "schema_version": 1,
            "expected_new_compiled_artifacts": 0,
            "proof_mode": "standalone_aot_artifact_reuse",
        },
        compiled_artifacts_saved_before=10,
        compiled_artifacts_saved_after=10,
        compiled_artifacts_loaded_before=4,
        compiled_artifacts_loaded_after=7,
        load_report={
            "schema_version": 1,
            "load_path": "fresh_deserialize",
            "loaded_artifact_count": 3,
            "deserialization_wall_time_ms": 1.0,
        },
    )
    assert verification == {
        "schema_version": 1,
        "ok": True,
        "expected_new_compiled_artifacts": 0,
        "actual_new_compiled_artifacts": 0,
        "actual_loaded_artifacts": 3,
        "reasons": [],
    }

    drifted = caching.verify_no_new_compile(
        {
            "schema_version": 1,
            "expected_new_compiled_artifacts": 0,
            "proof_mode": "standalone_aot_artifact_reuse",
        },
        compiled_artifacts_saved_before=10,
        compiled_artifacts_saved_after=12,
        compiled_artifacts_loaded_before=4,
        compiled_artifacts_loaded_after=5,
        load_report=None,
    )
    assert drifted == {
        "schema_version": 1,
        "ok": False,
        "expected_new_compiled_artifacts": 0,
        "actual_new_compiled_artifacts": 2,
        "actual_loaded_artifacts": 1,
        "reasons": [
            "unexpected_new_compiled_artifacts",
            "load_report_missing",
        ],
    }


def test_example_input_tensor_specs_roundtrip_helpers() -> None:
    caching, _ = _load_caching_module()

    class _FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    class _FakeBuiltTensor:
        def __init__(self, shape, dtype, device):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    original_tensor = caching.torch.Tensor
    original_float32 = getattr(caching.torch, "float32", None)
    original_empty = getattr(caching.torch, "empty", None)
    caching.torch.Tensor = _FakeTensor
    caching.torch.float32 = "float32-dtype"
    caching.torch.empty = lambda shape, dtype, device: _FakeBuiltTensor(
        shape, dtype, device
    )
    try:
        specs = caching.build_example_input_tensor_specs(
            [_FakeTensor((4, 8), "torch.float32"), "ignored"],
            [0],
        )
        assert specs == {
            "schema_version": 1,
            "input_count": 2,
            "indexed_tensors": [
                {
                    "index": 0,
                    "shape": [4, 8],
                    "dtype": "float32",
                }
            ],
        }

        rebuilt = caching.build_sparse_example_inputs_from_tensor_specs(specs)
        assert len(rebuilt) == 2
        assert rebuilt[1] is None
        assert rebuilt[0].shape == (4, 8)
        assert rebuilt[0].dtype == "float32-dtype"
        assert rebuilt[0].device == "meta"
    finally:
        caching.torch.Tensor = original_tensor
        if original_float32 is None:
            delattr(caching.torch, "float32")
        else:
            caching.torch.float32 = original_float32
        if original_empty is None:
            delattr(caching.torch, "empty")
        else:
            caching.torch.empty = original_empty
