# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import hashlib
import inspect
import json
import os
import pickle
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import torch
from torch._subclasses import FakeTensorMode
from torch.fx._graph_pickler import GraphPickler, Options
from torch.utils import _pytree as pytree

import vllm.envs as envs
import vllm.env_override as env_override
from vllm.compilation.codegen import compile_execution_fn
from vllm.compilation.compiler_interface import get_inductor_factors
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.utils import hash_factors
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

try:
    from torch._dynamo.aot_compile import SerializableCallable
except ImportError:
    SerializableCallable = object

assert isinstance(SerializableCallable, type)

logger = init_logger(__name__)


class StandaloneCompiledArtifacts:
    """Storage for standalone compiled artifacts with content-based deduplication.

    Deduplication works via a two-level indirection:
    1. `submodule_bytes` maps "{submod_name}_{shape}" -> SHA256 hash
    2. `submodule_bytes_store` maps SHA256 hash -> actual bytes

    When inserting, we compute the SHA256 hash of the bytes. If the hash
    already exists in `submodule_bytes_store`, we reuse the existing entry
    rather than storing duplicate bytes. This is common because submodules
    often compile to identical artifacts (e.g., identical transformer layers
    split on attn)
    """

    def __init__(self) -> None:
        # dict from submodule name to byte hash
        self.submodule_bytes: dict[str, str] = {}
        # dict from byte hash to bytes
        self.submodule_bytes_store: dict[str, bytes] = {}
        # dict from byte hash to loaded module
        self.loaded_submodule_store: dict[str, Any] = {}
        self._last_load_report: dict[str, object] | None = None

    def insert(self, submod_name: str, shape: str, entry: bytes) -> None:
        hex_digest = artifact_bytes_hash(entry)
        self.submodule_bytes[artifact_entry_key(submod_name, shape)] = hex_digest
        if hex_digest not in self.submodule_bytes_store:
            self.submodule_bytes_store[hex_digest] = entry
            compilation_counter.num_compiled_artifacts_saved += 1
            logger.debug(
                "inserting new artifact for submod %s with shape %s "
                "(%s bytes) at hash %s",
                submod_name,
                shape,
                len(entry),
                hex_digest,
            )
        else:
            logger.debug(
                "reusing existing cache artifact for submod %s "
                "with shape %s (%s bytes) at hash %s",
                submod_name,
                shape,
                len(entry),
                hex_digest,
            )

    def get(self, submod_name: str, shape: str) -> bytes:
        logger.debug(
            "getting artifact for submod %s with shape %s",
            submod_name,
            shape,
        )
        return self.submodule_bytes_store[
            self.submodule_bytes[artifact_entry_key(submod_name, shape)]
        ]

    def get_loaded(self, submod_name: str, shape: str) -> Any:
        logger.debug(
            "getting artifact for submod %s with shape %s",
            submod_name,
            shape,
        )
        return self.loaded_submodule_store[
            self.submodule_bytes[artifact_entry_key(submod_name, shape)]
        ]

    def size_bytes(self) -> int:
        return sum(len(entry) for entry in self.submodule_bytes_store.values())

    def num_artifacts(self) -> int:
        return len(self.submodule_bytes_store)

    def num_entries(self) -> int:
        return len(self.submodule_bytes)

    def submodule_names(self) -> list[str]:
        # get unique "{submod_name}" from "{submod_name}_{shape}", preserving order
        names = [cache_key.rsplit("_", 1)[0] for cache_key in self.submodule_bytes]
        return list(dict.fromkeys(names))

    def manifest_summary(self) -> dict[str, object]:
        hash_usage_counts: dict[str, int] = {}
        for digest in self.submodule_bytes.values():
            hash_usage_counts[digest] = hash_usage_counts.get(digest, 0) + 1

        entries = []
        for entry_key, digest in sorted(self.submodule_bytes.items()):
            submod_name, shape = entry_key.rsplit("_", 1)
            entry_bytes = self.submodule_bytes_store[digest]
            entries.append(
                {
                    "submodule_name": submod_name,
                    "shape": shape,
                    "artifact_hash": digest,
                    "artifact_bytes": len(entry_bytes),
                    "deduped": hash_usage_counts[digest] > 1,
                    "reuse_reason": (
                        "content_addressed_dedup"
                        if hash_usage_counts[digest] > 1
                        else "unique_artifact"
                    ),
                }
            )

        stores = [
            {
                "artifact_hash": digest,
                "artifact_bytes": len(entry_bytes),
                "entry_count": hash_usage_counts[digest],
            }
            for digest, entry_bytes in sorted(self.submodule_bytes_store.items())
        ]

        return {
            "schema_version": 1,
            "entry_count": self.num_entries(),
            "unique_artifact_count": self.num_artifacts(),
            "total_bytes": self.size_bytes(),
            "entries": entries,
            "stores": stores,
        }

    def store_identity(self) -> str:
        manifest = self.manifest_summary()
        return safe_hash(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            usedforsecurity=False,
        ).hexdigest()

    def render_manifest(self) -> str:
        manifest = self.manifest_summary()
        manifest["store_identity"] = self.store_identity()
        return json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    def verify_manifest(
        self, manifest: dict[str, object] | None = None
    ) -> dict[str, object]:
        expected = manifest or self.manifest_summary()
        actual = self.manifest_summary()
        expected_entries = expected.get("entries")
        actual_entries = actual.get("entries")
        expected_stores = expected.get("stores")
        actual_stores = actual.get("stores")
        reasons = []
        if expected.get("schema_version") != actual.get("schema_version"):
            reasons.append("schema_version_mismatch")
        if expected.get("entry_count") != actual.get("entry_count"):
            reasons.append("entry_count_mismatch")
        if expected.get("unique_artifact_count") != actual.get("unique_artifact_count"):
            reasons.append("unique_artifact_count_mismatch")
        if expected.get("total_bytes") != actual.get("total_bytes"):
            reasons.append("total_bytes_mismatch")
        if expected_entries != actual_entries:
            reasons.append("entry_manifest_mismatch")
        if expected_stores != actual_stores:
            reasons.append("artifact_store_mismatch")
        result = {
            "ok": not reasons,
            "reasons": reasons,
            "expected_store_identity": expected.get("store_identity"),
            "actual_store_identity": self.store_identity(),
            "entry_count": actual.get("entry_count"),
            "unique_artifact_count": actual.get("unique_artifact_count"),
            "total_bytes": actual.get("total_bytes"),
        }
        return result

    def reuse_summary(self) -> dict[str, object]:
        unique_bytes = self.size_bytes()
        total_entry_bytes = 0
        deduped_entry_count = 0
        for digest in self.submodule_bytes.values():
            entry_size = len(self.submodule_bytes_store[digest])
            total_entry_bytes += entry_size
        hash_usage_counts: dict[str, int] = {}
        for digest in self.submodule_bytes.values():
            hash_usage_counts[digest] = hash_usage_counts.get(digest, 0) + 1
        for count in hash_usage_counts.values():
            if count > 1:
                deduped_entry_count += count

        duplicate_entry_count = self.num_entries() - self.num_artifacts()
        return {
            "schema_version": 1,
            "cache_hit_reason": "standalone_aot_artifact_manifest_match",
            "artifact_reuse_mode": "content_addressed_dedup",
            "entry_count": self.num_entries(),
            "unique_artifact_count": self.num_artifacts(),
            "deduped_entry_count": deduped_entry_count,
            "duplicate_entry_count": duplicate_entry_count,
            "unique_bytes": unique_bytes,
            "expanded_entry_bytes": total_entry_bytes,
            "duplicate_bytes_elided": total_entry_bytes - unique_bytes,
            "duplicate_artifact_loads_avoided": duplicate_entry_count,
        }

    def last_load_report(self) -> dict[str, object] | None:
        return self._last_load_report

    def load_all(self) -> None:
        import concurrent.futures

        # check already loaded
        if len(self.loaded_submodule_store) == len(self.submodule_bytes_store):
            self._last_load_report = {
                "schema_version": 1,
                "load_path": "already_loaded",
                "loaded_artifact_count": len(self.loaded_submodule_store),
                "deserialization_wall_time_ms": 0.0,
            }
            return

        from torch._inductor.standalone_compile import AOTCompiledArtifact

        def _load_entry(entry_bytes: bytes) -> AOTCompiledArtifact:
            entry = pickle.loads(entry_bytes)
            compilation_counter.num_compiled_artifacts_loaded += 1
            return AOTCompiledArtifact.deserialize(entry)

        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            entries = list(self.submodule_bytes_store.values())
            loaded_entries = list(executor.map(_load_entry, entries))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        for i, k in enumerate(self.submodule_bytes_store.keys()):
            self.loaded_submodule_store[k] = loaded_entries[i]

        self._last_load_report = {
            "schema_version": 1,
            "load_path": "fresh_deserialize",
            "loaded_artifact_count": len(loaded_entries),
            "deserialization_wall_time_ms": round(elapsed_ms, 6),
        }
        logger.debug("loaded all %s submodules", self.num_artifacts())

    def __getstate__(self) -> dict[str, dict[str, str] | dict[str, bytes]]:
        return {
            "submodule_bytes": self.submodule_bytes,
            "submodule_bytes_store": self.submodule_bytes_store,
        }

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        self.submodule_bytes = state["submodule_bytes"]
        self.submodule_bytes_store = state["submodule_bytes_store"]
        self.loaded_submodule_store = {}


@contextlib.contextmanager
def patch_pytree_map_over_slice():
    pytree._private_register_pytree_node(
        slice, lambda x: ([x.start, x.stop, x.step], None), lambda x, c: slice(*x)
    )

    try:
        yield
    finally:
        pytree._deregister_pytree_node(slice)


class VllmSerializableFunction(SerializableCallable):  # type: ignore[misc]
    """
    A wrapper around a compiled function by vllm. It will forward the tensor
    inputs to the compiled function and return the result.
    It also implements a serialization interface to support PyTorch's precompile
    with custom backend, so that we can save and load the compiled function on
    disk. There's no need to wrap around the compiled function if we don't want
    to serialize them in particular cases.
    Right now serialization for the custom backend is done via
    serializing the Dynamo fx graph plus example inputs.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule | bytes,
        example_inputs: Sequence[Any],
        prefix: str,
        optimized_call: Callable[..., Any],
        is_encoder: bool = False,
        vllm_backend: Any | None = None,
        sym_tensor_indices: list[int] | None = None,
        aot_autograd_config: dict[str, Any] | None = None,
        execution_code: str | None = None,
        submod_names: list[str] | None = None,
        consts: list[Any] | None = None,
    ) -> None:
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.prefix = prefix
        self.optimized_call = optimized_call
        self.is_encoder = is_encoder
        self.shape_env = None
        self.vllm_backend = vllm_backend
        self.sym_tensor_indices = sym_tensor_indices
        self.execution_code = execution_code
        self.submod_names = submod_names
        self.consts = consts
        self._fake_mode: Any | None = None

        import torch._functorch.config as functorch_config

        self.aot_autograd_config = (
            aot_autograd_config or functorch_config.save_config_portable()
        )
        sym_input = next(
            (i for i in self.example_inputs if isinstance(i, torch.SymInt)), None
        )
        if sym_input is not None:
            self.shape_env = sym_input.node.shape_env

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimized_call(*args, **kwargs)

    @classmethod
    def serialize_graph_module(cls, graph_module: torch.fx.GraphModule) -> bytes:
        import sympy

        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(
            self: GraphPickler, obj: Any
        ) -> tuple[Callable[..., Any], tuple[Any, ...]] | Any:
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        with (
            patch.object(GraphPickler, "reducer_override", _graph_reducer_override),
            patch_pytree_map_over_slice(),
        ):
            return GraphPickler.dumps(graph_module, Options(ops_filter=None))

    @classmethod
    def deserialize_graph_module(
        cls, data: bytes, fake_mode: FakeTensorMode
    ) -> torch.fx.GraphModule:
        with patch_pytree_map_over_slice():
            return GraphPickler.loads(data, fake_mode)

    @classmethod
    def serialize_compile_artifacts(
        cls, compiled_fn: "VllmSerializableFunction"
    ) -> bytes:
        state = compiled_fn.__dict__.copy()
        state.pop("optimized_call")
        state.pop("shape_env")
        state.pop("vllm_backend", None)
        state.pop("_fake_mode", None)
        for node in state["graph_module"].graph.nodes:
            node.meta.pop("source_fn_stack", None)
            node.meta.pop("nn_module_stack", None)
        for name, submod in state["graph_module"].named_children():
            if hasattr(submod, "graph"):
                for node in submod.graph.nodes:
                    node.meta.pop("source_fn_stack", None)
                    node.meta.pop("nn_module_stack", None)

        if state.get("sym_tensor_indices"):
            # put tensor inputs on meta device since their data
            # isn't needed, yet we need the meta for make_copy_and_call
            state["example_inputs"] = pytree.tree_map_only(
                torch.Tensor,
                lambda inp: torch.empty_like(inp, device="meta"),
                state["example_inputs"],
            )
        else:
            # mask off all tensor inputs since they are large and not needed.
            state["example_inputs"] = pytree.tree_map_only(
                torch.Tensor,
                lambda inp: torch.empty_like(inp, device="meta"),
                state["example_inputs"],
            )

        state["graph_module"] = cls.serialize_graph_module(state["graph_module"])
        state["example_inputs"] = GraphPickler.dumps(state["example_inputs"])

        if compiled_fn.vllm_backend:
            (
                standalone_compile_artifacts,
                sym_shape_indices_map,
                returns_tuple_map,
            ) = compiled_fn.vllm_backend.collect_standalone_compile_artifacts()
            vllm_config = getattr(compiled_fn.vllm_backend, "vllm_config", None)
            state["standalone_compile_artifacts"] = standalone_compile_artifacts
            state["standalone_compile_artifact_manifest"] = (
                standalone_compile_artifacts.manifest_summary()
            )
            state["standalone_compile_artifact_store_identity"] = (
                standalone_compile_artifacts.store_identity()
            )
            state["standalone_compile_artifact_compatibility"] = (
                build_standalone_artifact_compatibility_manifest(vllm_config)
            )
            state["standalone_compile_artifact_reuse_summary"] = (
                standalone_compile_artifacts.reuse_summary()
            )
            state["standalone_compile_artifact_proof_manifest"] = (
                build_standalone_artifact_proof_manifest(
                    compiled_fn.vllm_backend,
                    standalone_compile_artifacts,
                    sym_shape_indices_map,
                    returns_tuple_map,
                )
            )
            state["sym_shape_indices_map"] = sym_shape_indices_map
            state["returns_tuple_map"] = returns_tuple_map
        return pickle.dumps(state)

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> "VllmSerializableFunction":
        from torch._guards import TracingContext, tracing
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        state = pickle.loads(data)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

        state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)

        standalone_compile_artifacts = state.pop("standalone_compile_artifacts", None)
        standalone_compile_artifact_manifest = state.pop(
            "standalone_compile_artifact_manifest", None
        )
        standalone_compile_artifact_store_identity = state.pop(
            "standalone_compile_artifact_store_identity", None
        )
        standalone_compile_artifact_compatibility = state.pop(
            "standalone_compile_artifact_compatibility", None
        )
        standalone_compile_artifact_reuse_summary = state.pop(
            "standalone_compile_artifact_reuse_summary", None
        )
        standalone_compile_artifact_proof_manifest = state.pop(
            "standalone_compile_artifact_proof_manifest", None
        )
        sym_shape_indices_map = state.pop("sym_shape_indices_map", {})
        returns_tuple_map = state.pop("returns_tuple_map", {})

        saved_aot_autograd_config = state["aot_autograd_config"]
        if saved_aot_autograd_config is not None:
            functorch_ctx = torch._functorch.config.patch(saved_aot_autograd_config)
        else:
            functorch_ctx = contextlib.nullcontext()

        if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            assert standalone_compile_artifacts is not None
            current_vllm_config = get_current_vllm_config()
            submod_names = standalone_compile_artifacts.submodule_names()
            num_submods = len(submod_names)
            num_artifacts = standalone_compile_artifacts.num_artifacts()
            if standalone_compile_artifact_manifest is not None:
                manifest_with_identity = dict(standalone_compile_artifact_manifest)
                manifest_with_identity["store_identity"] = (
                    standalone_compile_artifact_store_identity
                )
                verification = standalone_compile_artifacts.verify_manifest(
                    manifest_with_identity
                )
                if not verification["ok"]:
                    raise ValueError(
                        "Standalone compile artifact manifest verification failed: "
                        f"reasons={verification['reasons']} "
                        f"expected_store_identity={verification['expected_store_identity']} "
                        f"actual_store_identity={verification['actual_store_identity']}"
                    )
                compatibility_drift = explain_compatibility_drift(
                    standalone_compile_artifact_compatibility,
                    build_standalone_artifact_compatibility_manifest(
                        current_vllm_config
                    ),
                )
                startup_closure = summarize_startup_closure(
                    manifest_verification=verification,
                    compatibility_drift=compatibility_drift,
                    load_report=None,
                    assumes_closure=False,
                )
                logger.info(
                    "loading standalone compile artifacts. entries=%d unique_artifacts=%d store_identity=%s reuse=%s compatibility=%s compatibility_drift=%s startup_closure=%s proof=%s",
                    standalone_compile_artifact_manifest.get("entry_count", 0),
                    standalone_compile_artifact_manifest.get(
                        "unique_artifact_count", 0
                    ),
                    standalone_compile_artifact_store_identity or "<unknown>",
                    (
                        json.dumps(
                            standalone_compile_artifact_reuse_summary,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if standalone_compile_artifact_reuse_summary is not None
                        else "<unknown>"
                    ),
                    (
                        json.dumps(
                            standalone_compile_artifact_compatibility,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if standalone_compile_artifact_compatibility is not None
                        else "<unknown>"
                    ),
                    json.dumps(
                        compatibility_drift,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        startup_closure,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    (
                        json.dumps(
                            standalone_compile_artifact_proof_manifest,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if standalone_compile_artifact_proof_manifest is not None
                        else "<unknown>"
                    ),
                )
                if not compatibility_drift["ok"]:
                    logger.warning(
                        "standalone compile artifact compatibility drift detected. reasons=%s mismatches=%s",
                        compatibility_drift["reasons"],
                        compatibility_drift["mismatches"],
                    )

            with functorch_ctx:
                fn = reconstruct_serializable_fn_from_mega_artifact(
                    state=state,
                    standalone_compile_artifacts=standalone_compile_artifacts,
                    vllm_config=current_vllm_config,
                    sym_shape_indices_map=sym_shape_indices_map,
                    returns_tuple_map=returns_tuple_map,
                    fake_mode=fake_mode,
                )

            logger.info(
                "reconstructed serializable fn from standalone compile "
                "artifacts. num_artifacts=%d num_submods=%d",
                num_artifacts,
                num_submods,
            )

            return fn

        state["graph_module"] = cls.deserialize_graph_module(
            state["graph_module"], fake_mode
        )
        state["graph_module"].recompile()

        # Fall back to standard VllmBackend.
        # Use a lazy closure: the backend needs traced_files for cache
        # dir computation, but those are only populated after
        # _verify_source_unchanged runs in decorators.py (which happens
        # after deserialization completes).
        from vllm.compilation.backends import VllmBackend

        is_encoder = state.get("is_encoder", False)
        vllm_config = get_current_vllm_config()
        compile_inputs = list(state["example_inputs"])

        def optimized_call(*example_inputs: Any) -> Any:
            vllm_backend: VllmBackend = VllmBackend(
                vllm_config, state["prefix"], is_encoder
            )
            with tracing(TracingContext(fake_mode)), functorch_ctx:
                fn.optimized_call = vllm_backend(
                    state["graph_module"], compile_inputs
                ).optimized_call
                fn.vllm_backend = vllm_backend
            return fn.optimized_call(*example_inputs)

        fn = cls(**state, optimized_call=optimized_call)
        fn._fake_mode = fake_mode
        return fn

    def finalize_loading(self, vllm_config: VllmConfig) -> None:
        """Eagerly initialize the compiled backend and perform all loading.

        Must be called after _verify_source_unchanged has populated
        compilation_config.traced_files, which is needed for cache dir
        computation.
        """
        if self._fake_mode is None:
            return  # Already finalized, or mega path (no _fake_mode set)

        from torch._guards import TracingContext, tracing

        from vllm.compilation.backends import VllmBackend

        saved_aot_autograd_config = self.aot_autograd_config
        if saved_aot_autograd_config is not None:
            functorch_ctx = torch._functorch.config.patch(saved_aot_autograd_config)
        else:
            functorch_ctx = contextlib.nullcontext()

        vllm_backend = VllmBackend(vllm_config, self.prefix, self.is_encoder)
        with tracing(TracingContext(self._fake_mode)), functorch_ctx:
            result = vllm_backend(self.graph_module, list(self.example_inputs))
            self.optimized_call = result.optimized_call
            self.vllm_backend = vllm_backend

        logger.info(
            "finalized non-mega AOT loading. startup_closure=%s",
            json.dumps(
                summarize_startup_closure(
                    manifest_verification=None,
                    compatibility_drift=None,
                    load_report=None,
                    assumes_closure=True,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        self._fake_mode = None

    @property
    def co_name(self) -> Literal["VllmSerializableFunction"]:
        """
        Used for depyf debugging.
        """
        return "VllmSerializableFunction"


def reconstruct_serializable_fn_from_mega_artifact(
    state: dict[str, Any],
    standalone_compile_artifacts: "StandaloneCompiledArtifacts",
    vllm_config: VllmConfig,
    sym_shape_indices_map: dict[str, list[int]],
    returns_tuple_map: dict[str, bool],
    fake_mode: FakeTensorMode,
) -> "VllmSerializableFunction":
    """Construct a VllmSerializableFunction from cached inductor artifacts.

    This function reconstructs a callable model from pre-compiled inductor
    artifacts without re-running the compilation. It:
    1. Loads all cached artifacts
    2. Builds compiled callables for each submodule/shape
    3. Creates PiecewiseBackend instances that dispatch to cached artifacts
    4. Wraps with cudagraph if needed
    5. Returns the final VllmSerializableFunction

    Note: This function shares similar logic with PiecewiseCompileInterpreter
    in backends.py. Both create PiecewiseBackend instances and wrap them with
    cudagraph. The key difference is:
    - this function: PiecewiseBackend receives pre-compiled runnables
      (compiled_runnables is set, graph is None)
    - PiecewiseCompileInterpreter: PiecewiseBackend receives the FX graph
      to compile (graph is set, compiled_runnables is None)

    If modifying the backend creation/wrapping logic, consider updating both.

    Args:
        state: Deserialized state dict containing graph_module, example_inputs,
            prefix, sym_tensor_indices, is_encoder, etc.
        standalone_compile_artifacts: The StandaloneCompiledArtifacts containing
            pre-compiled artifacts for each submodule/shape combination.
        vllm_config: The vLLM configuration.
        sym_shape_indices_map: Mapping from submod_name to sym_shape_indices.
        returns_tuple_map: Mapping from submod_name to returns_tuple.

    Returns:
        A VllmSerializableFunction that can be called directly.
    """
    from vllm.compilation.backends import (
        VllmBackend,
        make_copy_and_call,
        wrap_with_cudagraph_if_needed,
    )
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    prefix = state["prefix"]
    is_encoder = state.get("is_encoder", False)
    compilation_config = vllm_config.compilation_config

    standalone_compile_artifacts.load_all()
    load_report = standalone_compile_artifacts.last_load_report()
    startup_closure = summarize_startup_closure(
        manifest_verification={
            "ok": True,
            "reasons": [],
        },
        compatibility_drift={
            "ok": True,
            "reasons": [],
            "mismatches": [],
        },
        load_report=load_report,
        assumes_closure=False,
    )

    piecewise_submod_names = standalone_compile_artifacts.submodule_names()
    compiled_callables: dict[str, dict[str, Callable[..., Any]]] = {}

    for cache_key in standalone_compile_artifacts.submodule_bytes:
        submod_name, shape_str = cache_key.rsplit("_", 1)
        compiled_callables.setdefault(submod_name, {})[shape_str] = (
            standalone_compile_artifacts.get_loaded(submod_name, shape_str)
        )

    vllm_backend = VllmBackend(vllm_config, prefix, is_encoder)
    dummy_cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "dummy_cache")
    os.makedirs(dummy_cache_dir, exist_ok=True)
    vllm_backend.compiler_manager.initialize_cache(
        cache_dir=dummy_cache_dir,
        disable_cache=True,
        prefix=prefix,
    )

    # spot check that cached submodules exist in the graph structure
    # if an old cache is used, this will fail but that's fine because
    # we will just try this error and re-generate the new cache.
    graph_children = set(state["submod_names"])
    missing = set(piecewise_submod_names) - graph_children
    assert not missing, (
        f"artifacts reference submodules not in graph: {missing}. "
        f"graph has: {sorted(graph_children)}"
    )

    submod_callables = {}
    for i, submod_name in enumerate(piecewise_submod_names):
        assert submod_name in sym_shape_indices_map and submod_name in returns_tuple_map

        sym_shape_indices = sym_shape_indices_map[submod_name]
        returns_tuple = returns_tuple_map[submod_name]
        runnables = compiled_callables[submod_name]

        piecewise_backend = PiecewiseBackend(
            graph=None,  # not needed for cached artifacts
            vllm_config=vllm_config,
            piecewise_compile_index=i,
            total_piecewise_compiles=len(piecewise_submod_names),
            sym_shape_indices=sym_shape_indices,
            vllm_backend=vllm_backend,
            returns_tuple=returns_tuple,
            compiled_runnables=runnables,
        )

        is_first = i == 0
        is_last = i == len(piecewise_submod_names) - 1
        wrapped_backend = wrap_with_cudagraph_if_needed(
            piecewise_backend,
            vllm_config,
            compilation_config,
            is_first,
            is_last,
        )

        submod_callables[submod_name] = wrapped_backend
        logger.debug(
            "Replaced submodule %s with piecewise backend from cache",
            submod_name,
        )

    if load_report is not None:
        logger.info(
            "standalone compile artifact load complete. loaded_artifacts=%d deserialization_wall_time_ms=%.6f load_path=%s startup_closure=%s",
            load_report.get("loaded_artifact_count", 0),
            float(load_report.get("deserialization_wall_time_ms", 0.0)),
            load_report.get("load_path", "<unknown>"),
            json.dumps(startup_closure, sort_keys=True, separators=(",", ":")),
        )

    # Use codegen'd execution code if available, fall back to split_gm
    execution_code = state.get("execution_code")
    submod_names = state.get("submod_names")
    if execution_code is not None and submod_names is not None:
        consts = state.get("consts")
        runtime_callable = compile_execution_fn(
            execution_code, submod_callables, submod_names, consts
        )
    else:
        logger.warning(
            "No execution code found, falling back to graph module execution."
        )
        runtime_callable = GraphPickler.loads(
            state["graph_module"], fake_mode=fake_mode
        )

    if compilation_config.cudagraph_copy_inputs:
        sym_tensor_indices = state["sym_tensor_indices"]
        input_buffers = [
            torch.empty_like(
                state["example_inputs"][idx], device=vllm_config.device_config.device
            )
            for idx in sym_tensor_indices
        ]
        optimized_call = make_copy_and_call(
            sym_tensor_indices, input_buffers, runtime_callable
        )
    else:
        optimized_call = runtime_callable

    fn = VllmSerializableFunction(
        **state,
        optimized_call=optimized_call,
        vllm_backend=None,
    )
    return fn


def aot_compile_hash_factors(vllm_config: VllmConfig) -> list[str]:
    factors = []
    # 0. factors come from the env, for example, The values of
    # VLLM_PP_LAYER_PARTITION will affect the computation graph.
    env_hash = hash_factors(envs.compile_factors())
    factors.append(env_hash)

    # 1. factors come from the vllm_config (it mainly summarizes how the
    #    model is created)
    config_hash = vllm_config.compute_hash()
    factors.append(config_hash)

    # 2. inductor factors if applicable
    if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        factors.extend(get_inductor_factors())

    return factors


def artifact_entry_key(submod_name: str, shape: str) -> str:
    return f"{submod_name}_{shape}"


def artifact_bytes_hash(entry: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(entry)
    return hasher.hexdigest()


def build_standalone_artifact_compatibility_manifest(
    vllm_config: VllmConfig | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": getattr(torch, "__version__", "<unknown>"),
        "mega_aot_enabled": envs.VLLM_USE_MEGA_AOT_ARTIFACT,
        "env": envs.compile_factor_manifest(),
        "vllm_config_hash": (
            vllm_config.compute_hash() if vllm_config is not None else None
        ),
    }


def load_compile_cache_key_factors(
    local_cache_dir: str | None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    meta_path = Path(local_cache_dir) / "cache_key_factors.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning("could not read compile cache key factors from %s", meta_path)
        return None


def build_shape_envelope_summary(
    standalone_compile_artifacts: StandaloneCompiledArtifacts,
    sym_shape_indices_map: dict[str, list[int]] | None,
    returns_tuple_map: dict[str, bool] | None,
) -> dict[str, object]:
    submodule_shapes: dict[str, list[str]] = {}
    for cache_key in sorted(standalone_compile_artifacts.submodule_bytes):
        submod_name, shape_str = cache_key.rsplit("_", 1)
        submodule_shapes.setdefault(submod_name, []).append(shape_str)

    submodules = []
    for submod_name, shapes in submodule_shapes.items():
        submodules.append(
            {
                "submodule_name": submod_name,
                "shape_variants": tuple(shapes),
                "shape_count": len(shapes),
                "symbolic_input_positions": tuple(
                    (sym_shape_indices_map or {}).get(submod_name, [])
                ),
                "returns_tuple": bool((returns_tuple_map or {}).get(submod_name, False)),
            }
        )

    return {
        "schema_version": 1,
        "submodule_count": len(submodules),
        "total_shape_variants": sum(item["shape_count"] for item in submodules),
        "submodules": submodules,
    }


def build_standalone_artifact_proof_manifest(
    vllm_backend: Any,
    standalone_compile_artifacts: StandaloneCompiledArtifacts,
    sym_shape_indices_map: dict[str, list[int]] | None,
    returns_tuple_map: dict[str, bool] | None,
) -> dict[str, object]:
    vllm_config = getattr(vllm_backend, "vllm_config", None)
    compilation_config = getattr(vllm_backend, "compilation_config", None)
    compiler_manager = getattr(vllm_backend, "compiler_manager", None)
    cache_key_factors = load_compile_cache_key_factors(
        getattr(compilation_config, "local_cache_dir", None)
    )

    compile_hashes = {
        "env_policy_hash": (
            hash_factors(cache_key_factors["env"])
            if cache_key_factors is not None and "env" in cache_key_factors
            else hash_factors(envs.compile_factors())
        ),
        "config_hash": (
            cache_key_factors.get("config_hash")
            if cache_key_factors is not None
            else (vllm_config.compute_hash() if vllm_config is not None else None)
        ),
        "code_hash": (
            cache_key_factors.get("code_hash") if cache_key_factors is not None else None
        ),
        "compiler_hash": (
            cache_key_factors.get("compiler_hash")
            if cache_key_factors is not None
            else (
                compiler_manager.compute_hash(vllm_config)
                if compiler_manager is not None and vllm_config is not None
                else None
            )
        ),
    }

    return {
        "schema_version": 1,
        "compile_hashes": compile_hashes,
        "backend_identity": {
            "backend_class": type(vllm_backend).__name__,
            "prefix": getattr(vllm_backend, "prefix", None),
            "is_encoder": bool(getattr(vllm_backend, "is_encoder", False)),
            "compiler_name": (
                getattr(getattr(compiler_manager, "compiler", None), "name", None)
                if compiler_manager is not None
                else None
            ),
        },
        "toolchain_identity": {
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "torch_version": getattr(torch, "__version__", "<unknown>"),
        },
        "patch_profile": env_override.patch_profile_manifest(),
        "shape_envelope": build_shape_envelope_summary(
            standalone_compile_artifacts,
            sym_shape_indices_map,
            returns_tuple_map,
        ),
    }


def explain_compatibility_drift(
    expected: dict[str, object] | None,
    actual: dict[str, object] | None,
) -> dict[str, object]:
    if expected is None or actual is None:
        return {
            "ok": False,
            "reasons": ["compatibility_metadata_missing"],
            "mismatches": [],
        }

    mismatches = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            mismatches.append(key)

    reasons = [f"compatibility_{key}_mismatch" for key in mismatches]
    return {
        "ok": not mismatches,
        "reasons": reasons,
        "mismatches": mismatches,
    }


def summarize_startup_closure(
    manifest_verification: dict[str, object] | None,
    compatibility_drift: dict[str, object] | None,
    load_report: dict[str, object] | None,
    assumes_closure: bool,
) -> dict[str, object]:
    reasons = []
    if assumes_closure:
        reasons.append("closure_not_proven_by_manifest")
        return {
            "schema_version": 1,
            "status": "closure_by_assumption",
            "reasons": reasons,
        }

    if manifest_verification is None:
        reasons.append("manifest_verification_missing")
    elif not bool(manifest_verification.get("ok", False)):
        reasons.extend(manifest_verification.get("reasons", []))

    if compatibility_drift is None:
        reasons.append("compatibility_verification_missing")
    elif not bool(compatibility_drift.get("ok", False)):
        reasons.extend(compatibility_drift.get("reasons", []))

    if load_report is not None and load_report.get("load_path") == "already_loaded":
        reasons.append("artifact_store_preloaded")

    status = "full_compile_closure" if not reasons else "partial_compile_closure"
    return {
        "schema_version": 1,
        "status": status,
        "reasons": reasons,
    }


def render_aot_compile_factor_manifest(vllm_config: VllmConfig) -> str:
    manifest = {
        "schema_version": 1,
        "env": envs.compile_factor_manifest(),
        "vllm_config_hash": vllm_config.compute_hash(),
        "inductor_factors": get_inductor_factors() if envs.VLLM_USE_MEGA_AOT_ARTIFACT else [],
    }
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def _compute_code_hash_with_content(file_contents: dict[str, str]) -> str:
    items = list(sorted(file_contents.items(), key=lambda x: x[0]))
    hash_content = []
    for filepath, content in items:
        hash_content.append(filepath)
        if filepath == "<string>":
            # This means the function was dynamically generated, with
            # e.g. exec(). We can't actually check these.
            continue
        hash_content.append(content)
    result: str = safe_hash(
        "\n".join(hash_content).encode(), usedforsecurity=False
    ).hexdigest()
    return result


def _compute_code_hash(files: set[str]) -> str:
    logger.debug(
        "Traced files (to be considered for compilation cache):\n%s", "\n".join(files)
    )
    file_contents = {}
    for filepath in files:
        # Skip files that don't exist (e.g., <string>, <frozen modules>, etc.)
        if not os.path.isfile(filepath):
            file_contents[filepath] = ""
        else:
            with open(filepath) as f:
                file_contents[filepath] = f.read()
    return _compute_code_hash_with_content(file_contents)
