# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
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
from vllm.compilation.codegen import compile_execution_fn, compile_execution_plan_fn
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

_COMPILE_ARTIFACT_BUNDLE_MAGIC = b"VLLM_AOT_BUNDLE_V1\n"
_COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES = 8
_STANDALONE_ARTIFACT_STORE_BUNDLE_MAGIC = b"VLLM_STANDALONE_AOT_STORE_V1\n"
_SERIALIZED_FN_STATE_BUNDLE_MAGIC = b"VLLM_SERIALIZED_FN_STATE_V1\n"
_SHARED_LOADED_ARTIFACT_STORES: dict[str, dict[str, Any]] = {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def build_standalone_artifact_sidecar(
    *,
    standalone_compile_artifact_manifest: dict[str, object],
    standalone_compile_artifact_store_identity: str,
    standalone_compile_artifact_placement_identity: str,
    standalone_compile_artifact_compatibility: dict[str, object],
    standalone_compile_artifact_reuse_summary: dict[str, object],
    standalone_compile_artifact_proof_manifest: dict[str, object],
    sym_shape_indices_map: dict[str, list[int]] | dict[str, tuple[int, ...]],
    returns_tuple_map: dict[str, bool],
    example_input_tensor_specs: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "payload_kind": "vllm_standalone_compile_artifact_sidecar",
        "artifact_manifest": _json_ready(standalone_compile_artifact_manifest),
        "store_identity": standalone_compile_artifact_store_identity,
        "placement_identity": standalone_compile_artifact_placement_identity,
        "compatibility": _json_ready(standalone_compile_artifact_compatibility),
        "reuse_summary": _json_ready(standalone_compile_artifact_reuse_summary),
        "proof_manifest": _json_ready(standalone_compile_artifact_proof_manifest),
        "sym_shape_indices_map": _json_ready(sym_shape_indices_map),
        "returns_tuple_map": _json_ready(returns_tuple_map),
        "example_input_tensor_specs": _json_ready(example_input_tensor_specs),
    }


def build_no_new_compile_expectation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_new_compiled_artifacts": 0,
        "proof_mode": "standalone_aot_artifact_reuse",
    }


def build_example_input_tensor_specs(
    example_inputs: Sequence[Any],
    sym_tensor_indices: list[int] | None,
) -> dict[str, object] | None:
    if not sym_tensor_indices:
        return {
            "schema_version": 1,
            "input_count": len(example_inputs),
            "indexed_tensors": [],
        }

    indexed_tensors = []
    for index in sym_tensor_indices:
        if index >= len(example_inputs):
            return None
        example_input = example_inputs[index]
        if not isinstance(example_input, torch.Tensor):
            return None
        indexed_tensors.append(
            {
                "index": index,
                "shape": list(example_input.shape),
                "dtype": str(example_input.dtype).removeprefix("torch."),
            }
        )

    return {
        "schema_version": 1,
        "input_count": len(example_inputs),
        "indexed_tensors": indexed_tensors,
    }


def build_sparse_example_inputs_from_tensor_specs(
    specs: dict[str, object],
) -> list[Any]:
    input_count = int(specs.get("input_count", 0))
    example_inputs: list[Any] = [None] * input_count
    for item in specs.get("indexed_tensors", []):
        if not isinstance(item, dict):
            continue
        dtype_name = str(item["dtype"])
        dtype = getattr(torch, dtype_name)
        shape = tuple(int(dim) for dim in item["shape"])
        example_inputs[int(item["index"])] = torch.empty(
            shape,
            dtype=dtype,
            device="meta",
        )
    return example_inputs


def pack_serialized_compile_artifact_bundle(
    payload: bytes,
    sidecar: dict[str, object] | None,
) -> bytes:
    if sidecar is None:
        return payload

    sidecar_bytes = json.dumps(
        sidecar,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        (
            _COMPILE_ARTIFACT_BUNDLE_MAGIC,
            len(sidecar_bytes).to_bytes(
                _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES,
                byteorder="big",
                signed=False,
            ),
            sidecar_bytes,
            payload,
        )
    )


def unpack_serialized_compile_artifact_bundle(
    data: bytes,
) -> tuple[dict[str, object] | None, bytes]:
    if not data.startswith(_COMPILE_ARTIFACT_BUNDLE_MAGIC):
        return None, data

    offset = len(_COMPILE_ARTIFACT_BUNDLE_MAGIC)
    sidecar_size = int.from_bytes(
        data[offset : offset + _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES],
        byteorder="big",
        signed=False,
    )
    offset += _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES
    sidecar_bytes = data[offset : offset + sidecar_size]
    payload = data[offset + sidecar_size :]
    return json.loads(sidecar_bytes), payload


def pack_standalone_artifact_store_bundle(
    artifacts: "StandaloneCompiledArtifacts",
) -> bytes:
    offset = 0
    payload_parts = []
    artifacts_index = []
    for digest, entry_bytes in sorted(artifacts.submodule_bytes_store.items()):
        entry_size = len(entry_bytes)
        artifacts_index.append(
            {
                "artifact_hash": digest,
                "offset": offset,
                "size": entry_size,
            }
        )
        payload_parts.append(entry_bytes)
        offset += entry_size

    header = {
        "schema_version": 1,
        "payload_kind": "vllm_standalone_artifact_store_bundle",
        "store_identity": artifacts.store_identity(),
        "submodule_bytes": dict(sorted(artifacts.submodule_bytes.items())),
        "artifacts": artifacts_index,
    }
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        (
            _STANDALONE_ARTIFACT_STORE_BUNDLE_MAGIC,
            len(header_bytes).to_bytes(
                _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES,
                byteorder="big",
                signed=False,
            ),
            header_bytes,
            b"".join(payload_parts),
        )
    )


def unpack_standalone_artifact_store_bundle(
    data: bytes,
) -> "StandaloneCompiledArtifacts":
    if not data.startswith(_STANDALONE_ARTIFACT_STORE_BUNDLE_MAGIC):
        raise ValueError("invalid standalone artifact store bundle header")

    offset = len(_STANDALONE_ARTIFACT_STORE_BUNDLE_MAGIC)
    header_size = int.from_bytes(
        data[offset : offset + _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES],
        byteorder="big",
        signed=False,
    )
    offset += _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES
    header = json.loads(data[offset : offset + header_size])
    payload = data[offset + header_size :]

    artifacts = StandaloneCompiledArtifacts()
    artifacts.submodule_bytes = {
        str(key): str(digest)
        for key, digest in dict(header["submodule_bytes"]).items()
    }
    for item in header["artifacts"]:
        entry_offset = int(item["offset"])
        entry_size = int(item["size"])
        entry_bytes = payload[entry_offset : entry_offset + entry_size]
        digest = str(item["artifact_hash"])
        if artifact_bytes_hash(entry_bytes) != digest:
            raise ValueError(
                "standalone artifact store bundle hash verification failed "
                f"for digest={digest}"
            )
        artifacts.submodule_bytes_store[digest] = entry_bytes

    expected_store_identity = header.get("store_identity")
    if expected_store_identity is not None:
        actual_store_identity = artifacts.store_identity()
        if actual_store_identity != expected_store_identity:
            raise ValueError(
                "standalone artifact store identity verification failed "
                f"expected={expected_store_identity} actual={actual_store_identity}"
            )
    return artifacts


def pack_serialized_fn_state_bundle(state: dict[str, Any]) -> bytes:
    metadata = {
        "schema_version": 1,
        "payload_kind": "vllm_serialized_fn_state_bundle",
        "prefix": state["prefix"],
        "is_encoder": bool(state["is_encoder"]),
        "sym_tensor_indices": state.get("sym_tensor_indices"),
        "execution_plan": _json_ready(state.get("execution_plan")),
        "execution_code": state.get("execution_code"),
        "submod_names": state.get("submod_names"),
        "blobs": [],
    }
    blob_payloads: list[bytes] = []
    offset = 0
    blob_specs = []
    if state.get("graph_module") is not None:
        blob_specs.append(("graph_module", state["graph_module"], "raw"))
    if state.get("example_inputs") is not None:
        blob_specs.append(("example_inputs", state["example_inputs"], "raw"))
    if state.get("standalone_compile_artifact_store_bundle") is not None:
        blob_specs.append(
            (
                "standalone_compile_artifact_store_bundle",
                state["standalone_compile_artifact_store_bundle"],
                "raw",
            )
        )
    if state.get("aot_autograd_config") is not None:
        blob_specs.append(
            (
                "aot_autograd_config",
                pickle.dumps(state["aot_autograd_config"]),
                "pickle",
            )
        )
    if state.get("consts") is not None:
        blob_specs.append(("consts", pickle.dumps(state["consts"]), "pickle"))

    for name, blob, codec in blob_specs:
        blob_bytes = bytes(blob)
        metadata["blobs"].append(
            {
                "name": name,
                "offset": offset,
                "size": len(blob_bytes),
                "codec": codec,
            }
        )
        blob_payloads.append(blob_bytes)
        offset += len(blob_bytes)

    header_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        (
            _SERIALIZED_FN_STATE_BUNDLE_MAGIC,
            len(header_bytes).to_bytes(
                _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES,
                byteorder="big",
                signed=False,
            ),
            header_bytes,
            b"".join(blob_payloads),
        )
    )


def unpack_serialized_fn_state_bundle(
    data: bytes,
    *,
    eager_pickle: bool = True,
) -> dict[str, Any]:
    if not data.startswith(_SERIALIZED_FN_STATE_BUNDLE_MAGIC):
        raise ValueError("invalid serialized function state bundle header")

    offset = len(_SERIALIZED_FN_STATE_BUNDLE_MAGIC)
    header_size = int.from_bytes(
        data[offset : offset + _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES],
        byteorder="big",
        signed=False,
    )
    offset += _COMPILE_ARTIFACT_BUNDLE_SIZE_BYTES
    header = json.loads(data[offset : offset + header_size])
    payload = data[offset + header_size :]

    state: dict[str, Any] = {
        "graph_module": None,
        "example_inputs": None,
        "prefix": header["prefix"],
        "is_encoder": bool(header["is_encoder"]),
        "sym_tensor_indices": header.get("sym_tensor_indices"),
        "execution_plan": header.get("execution_plan"),
        "execution_code": header.get("execution_code"),
        "submod_names": header.get("submod_names"),
        "aot_autograd_config": None,
        "consts": None,
        "standalone_compile_artifact_store_bundle": None,
    }
    serialized_blob_codecs: dict[str, str] = {}
    for item in header["blobs"]:
        blob_offset = int(item["offset"])
        blob_size = int(item["size"])
        blob = payload[blob_offset : blob_offset + blob_size]
        name = str(item["name"])
        codec = str(item["codec"])
        serialized_blob_codecs[name] = codec
        if codec == "pickle" and eager_pickle:
            state[name] = pickle.loads(blob)
        else:
            state[name] = blob
    if not eager_pickle:
        state["_serialized_blob_codecs"] = serialized_blob_codecs
    return state


def materialize_serialized_fn_state_fields(
    state: dict[str, Any],
    *names: str,
) -> None:
    serialized_blob_codecs = state.get("_serialized_blob_codecs")
    if not isinstance(serialized_blob_codecs, dict):
        return

    for name in names:
        if serialized_blob_codecs.get(name) != "pickle":
            continue
        value = state.get(name)
        if isinstance(value, bytes):
            state[name] = pickle.loads(value)


def build_artifact_load_topology_summary(
    *,
    store_identity: str,
    unique_artifact_count: int,
    unique_bytes: int,
    expanded_entry_bytes: int,
) -> dict[str, object]:
    local_rank = int(getattr(envs, "LOCAL_RANK", 0))
    dp_rank = int(getattr(envs, "VLLM_DP_RANK", 0))
    dp_rank_local = int(getattr(envs, "VLLM_DP_RANK_LOCAL", dp_rank))
    dp_size = max(int(getattr(envs, "VLLM_DP_SIZE", 1)), 1)
    global_rank = int(os.environ.get("RANK", str(dp_rank)))
    local_processes = max(local_rank + 1, 1)

    duplicate_process_loads_estimate = max(dp_size - 1, 0)
    cluster_unique_bytes_estimate = unique_bytes * dp_size
    cluster_expanded_entry_bytes_estimate = expanded_entry_bytes * dp_size

    return {
        "schema_version": 1,
        "store_identity": store_identity,
        "process_id": os.getpid(),
        "global_rank": global_rank,
        "local_rank": local_rank,
        "data_parallel_rank": dp_rank,
        "data_parallel_rank_local": dp_rank_local,
        "data_parallel_size": dp_size,
        "local_process_count_estimate": local_processes,
        "unique_artifact_count": unique_artifact_count,
        "unique_bytes": unique_bytes,
        "expanded_entry_bytes": expanded_entry_bytes,
        "duplicate_process_loads_estimate": duplicate_process_loads_estimate,
        "duplicate_rank_loads_estimate": duplicate_process_loads_estimate,
        "cluster_unique_bytes_estimate": cluster_unique_bytes_estimate,
        "cluster_expanded_entry_bytes_estimate": cluster_expanded_entry_bytes_estimate,
        "cluster_duplicate_artifact_bytes_estimate": (
            cluster_expanded_entry_bytes_estimate - cluster_unique_bytes_estimate
        ),
    }


def build_rank_local_placement_manifest(
    *,
    submodule_bytes: dict[str, str],
) -> dict[str, object]:
    placement_entries = [
        {
            "entry_key": entry_key,
            "artifact_hash": digest,
        }
        for entry_key, digest in sorted(submodule_bytes.items())
    ]
    return {
        "schema_version": 1,
        "global_rank": int(os.environ.get("RANK", str(getattr(envs, "VLLM_DP_RANK", 0)))),
        "local_rank": int(getattr(envs, "LOCAL_RANK", 0)),
        "data_parallel_rank": int(getattr(envs, "VLLM_DP_RANK", 0)),
        "data_parallel_rank_local": int(
            getattr(
                envs,
                "VLLM_DP_RANK_LOCAL",
                int(getattr(envs, "VLLM_DP_RANK", 0)),
            )
        ),
        "data_parallel_size": max(int(getattr(envs, "VLLM_DP_SIZE", 1)), 1),
        "entries": placement_entries,
    }


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
        return self._load_digest(
            self.submodule_bytes[artifact_entry_key(submod_name, shape)]
        )

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

    def payload_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "unique_artifact_count": self.num_artifacts(),
            "total_bytes": self.size_bytes(),
            "stores": [
                {
                    "artifact_hash": digest,
                    "artifact_bytes": len(entry_bytes),
                }
                for digest, entry_bytes in sorted(self.submodule_bytes_store.items())
            ],
        }

    def placement_manifest(self) -> dict[str, object]:
        return build_rank_local_placement_manifest(
            submodule_bytes=self.submodule_bytes,
        )

    def store_identity(self) -> str:
        manifest = self.payload_manifest()
        return safe_hash(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            usedforsecurity=False,
        ).hexdigest()

    def placement_identity(self) -> str:
        manifest = self.placement_manifest()
        return safe_hash(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            usedforsecurity=False,
        ).hexdigest()

    def render_manifest(self) -> str:
        manifest = self.manifest_summary()
        manifest["store_identity"] = self.store_identity()
        manifest["placement_identity"] = self.placement_identity()
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
            "expected_placement_identity": expected.get("placement_identity"),
            "actual_placement_identity": self.placement_identity(),
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
        store_identity = self.store_identity()
        return {
            "schema_version": 1,
            "cache_hit_reason": "standalone_aot_artifact_manifest_match",
            "artifact_reuse_mode": "content_addressed_dedup",
            "store_identity": store_identity,
            "placement_identity": self.placement_identity(),
            "entry_count": self.num_entries(),
            "unique_artifact_count": self.num_artifacts(),
            "deduped_entry_count": deduped_entry_count,
            "duplicate_entry_count": duplicate_entry_count,
            "unique_bytes": unique_bytes,
            "expanded_entry_bytes": total_entry_bytes,
            "duplicate_bytes_elided": total_entry_bytes - unique_bytes,
            "duplicate_artifact_loads_avoided": duplicate_entry_count,
            "load_topology": build_artifact_load_topology_summary(
                store_identity=store_identity,
                unique_artifact_count=self.num_artifacts(),
                unique_bytes=unique_bytes,
                expanded_entry_bytes=total_entry_bytes,
            ),
        }

    def last_load_report(self) -> dict[str, object] | None:
        return self._last_load_report

    def _store_cache_identity(self) -> tuple[str, tuple[str, ...]]:
        return self.store_identity(), tuple(sorted(self.submodule_bytes_store))

    def _init_load_report(self, store_identity: str, target_artifact_count: int) -> None:
        self._last_load_report = {
            "schema_version": 1,
            "load_path": "already_loaded",
            "loaded_artifact_count": len(self.loaded_submodule_store),
            "target_artifact_count": target_artifact_count,
            "fresh_deserialize_count": 0,
            "shared_reuse_count": 0,
            "already_loaded_count": 0,
            "deserialization_wall_time_ms": 0.0,
            "store_identity": store_identity,
            "load_topology": build_artifact_load_topology_summary(
                store_identity=store_identity,
                unique_artifact_count=self.num_artifacts(),
                unique_bytes=self.size_bytes(),
                expanded_entry_bytes=sum(
                    len(self.submodule_bytes_store[digest])
                    for digest in self.submodule_bytes.values()
                ),
            ),
        }

    def _update_load_path(self) -> None:
        assert self._last_load_report is not None
        fresh_count = int(self._last_load_report["fresh_deserialize_count"])
        shared_count = int(self._last_load_report["shared_reuse_count"])
        already_loaded_count = int(self._last_load_report["already_loaded_count"])
        active_paths = sum(
            1
            for count in (fresh_count, shared_count, already_loaded_count)
            if count > 0
        )
        if active_paths > 1:
            self._last_load_report["load_path"] = "mixed_materialization"
        elif fresh_count > 0:
            self._last_load_report["load_path"] = "fresh_deserialize"
        elif shared_count > 0:
            self._last_load_report["load_path"] = "shared_loaded_store"
        else:
            self._last_load_report["load_path"] = "already_loaded"

    def _record_load_result(
        self,
        *,
        store_identity: str,
        source: Literal["already_loaded", "shared_loaded_store", "fresh_deserialize"],
        target_artifact_count: int,
        deserialization_elapsed_ms: float = 0.0,
    ) -> None:
        if self._last_load_report is None:
            self._init_load_report(store_identity, target_artifact_count)
        assert self._last_load_report is not None
        report = self._last_load_report
        report["loaded_artifact_count"] = len(self.loaded_submodule_store)
        report["target_artifact_count"] = target_artifact_count
        report["store_identity"] = store_identity
        if source == "already_loaded":
            report["already_loaded_count"] = int(report["already_loaded_count"]) + 1
        elif source == "shared_loaded_store":
            report["shared_reuse_count"] = int(report["shared_reuse_count"]) + 1
        else:
            report["fresh_deserialize_count"] = (
                int(report["fresh_deserialize_count"]) + 1
            )
            report["deserialization_wall_time_ms"] = round(
                float(report["deserialization_wall_time_ms"]) + deserialization_elapsed_ms,
                6,
            )
        self._update_load_path()

    def _load_digest(self, digest: str) -> Any:
        store_identity, _ = self._store_cache_identity()
        target_artifact_count = len(self.submodule_bytes_store)
        if digest in self.loaded_submodule_store:
            self._record_load_result(
                store_identity=store_identity,
                source="already_loaded",
                target_artifact_count=target_artifact_count,
            )
            return self.loaded_submodule_store[digest]

        shared_loaded_store = _SHARED_LOADED_ARTIFACT_STORES.get(store_identity)
        if shared_loaded_store is not None and digest in shared_loaded_store:
            self.loaded_submodule_store[digest] = shared_loaded_store[digest]
            self._record_load_result(
                store_identity=store_identity,
                source="shared_loaded_store",
                target_artifact_count=target_artifact_count,
            )
            return self.loaded_submodule_store[digest]

        from torch._inductor.standalone_compile import AOTCompiledArtifact

        start_time = time.perf_counter()
        entry = pickle.loads(self.submodule_bytes_store[digest])
        compilation_counter.num_compiled_artifacts_loaded += 1
        loaded_artifact = AOTCompiledArtifact.deserialize(entry)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.loaded_submodule_store[digest] = loaded_artifact
        _SHARED_LOADED_ARTIFACT_STORES.setdefault(store_identity, {})[
            digest
        ] = loaded_artifact
        self._record_load_result(
            store_identity=store_identity,
            source="fresh_deserialize",
            target_artifact_count=target_artifact_count,
            deserialization_elapsed_ms=elapsed_ms,
        )
        return loaded_artifact

    def build_lazy_loaded_artifact(self, digest: str) -> "LazyLoadedArtifact":
        return LazyLoadedArtifact(self, digest)

    def mark_deferred_materialization(self) -> None:
        store_identity, _ = self._store_cache_identity()
        self._init_load_report(
            store_identity, target_artifact_count=len(self.submodule_bytes_store)
        )
        assert self._last_load_report is not None
        self._last_load_report["load_path"] = "deferred_materialization"

    def load_all(self) -> None:
        store_identity, store_keys = self._store_cache_identity()

        # check already loaded
        if len(self.loaded_submodule_store) == len(self.submodule_bytes_store):
            self._init_load_report(
                store_identity, target_artifact_count=len(self.submodule_bytes_store)
            )
            return

        shared_loaded_store = _SHARED_LOADED_ARTIFACT_STORES.get(store_identity)
        if shared_loaded_store is not None:
            shared_store_keys = tuple(sorted(shared_loaded_store))
            if shared_store_keys == store_keys:
                self.loaded_submodule_store = {
                    digest: shared_loaded_store[digest] for digest in store_keys
                }
                self._last_load_report = {
                    "schema_version": 1,
                    "load_path": "shared_loaded_store",
                    "loaded_artifact_count": len(self.loaded_submodule_store),
                    "target_artifact_count": len(self.submodule_bytes_store),
                    "fresh_deserialize_count": 0,
                    "shared_reuse_count": len(self.loaded_submodule_store),
                    "already_loaded_count": 0,
                    "deserialization_wall_time_ms": 0.0,
                    "store_identity": store_identity,
                }
                return

        self._init_load_report(
            store_identity, target_artifact_count=len(self.submodule_bytes_store)
        )
        for digest in self.submodule_bytes_store:
            self._load_digest(digest)
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
        execution_plan: dict[str, object] | None = None,
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
        self.execution_plan = execution_plan
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
        sidecar = None
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

        example_input_tensor_specs = build_example_input_tensor_specs(
            state["example_inputs"],
            state.get("sym_tensor_indices"),
        )

        if compiled_fn.vllm_backend:
            (
                standalone_compile_artifacts,
                sym_shape_indices_map,
                returns_tuple_map,
            ) = compiled_fn.vllm_backend.collect_standalone_compile_artifacts()
            vllm_config = getattr(compiled_fn.vllm_backend, "vllm_config", None)
            state["standalone_compile_artifact_store_bundle"] = (
                pack_standalone_artifact_store_bundle(standalone_compile_artifacts)
            )
            standalone_compile_artifact_manifest = (
                standalone_compile_artifacts.manifest_summary()
            )
            standalone_compile_artifact_store_identity = (
                standalone_compile_artifacts.store_identity()
            )
            standalone_compile_artifact_placement_identity = (
                standalone_compile_artifacts.placement_identity()
            )
            standalone_compile_artifact_compatibility = (
                build_standalone_artifact_compatibility_manifest(vllm_config)
            )
            standalone_compile_artifact_reuse_summary = (
                standalone_compile_artifacts.reuse_summary()
            )
            standalone_compile_artifact_proof_manifest = (
                build_standalone_artifact_proof_manifest(
                    compiled_fn.vllm_backend,
                    standalone_compile_artifacts,
                    sym_shape_indices_map,
                    returns_tuple_map,
                )
            )
            sidecar = build_standalone_artifact_sidecar(
                standalone_compile_artifact_manifest=(
                    standalone_compile_artifact_manifest
                ),
                standalone_compile_artifact_store_identity=(
                    standalone_compile_artifact_store_identity
                ),
                standalone_compile_artifact_placement_identity=(
                    standalone_compile_artifact_placement_identity
                ),
                standalone_compile_artifact_compatibility=(
                    standalone_compile_artifact_compatibility
                ),
                standalone_compile_artifact_reuse_summary=(
                    standalone_compile_artifact_reuse_summary
                ),
                standalone_compile_artifact_proof_manifest=(
                    standalone_compile_artifact_proof_manifest
                ),
                sym_shape_indices_map=sym_shape_indices_map,
                returns_tuple_map=returns_tuple_map,
                example_input_tensor_specs=example_input_tensor_specs,
            )

        include_graph_module_blob = True
        include_example_inputs_blob = True
        if (
            envs.VLLM_USE_MEGA_AOT_ARTIFACT
            and state.get("execution_plan") is not None
            and state.get("submod_names") is not None
        ):
            include_graph_module_blob = False
        if envs.VLLM_USE_MEGA_AOT_ARTIFACT and example_input_tensor_specs is not None:
            include_example_inputs_blob = False

        if include_graph_module_blob:
            state["graph_module"] = cls.serialize_graph_module(state["graph_module"])
        else:
            state["graph_module"] = None

        if include_example_inputs_blob:
            state["example_inputs"] = GraphPickler.dumps(state["example_inputs"])
        else:
            state["example_inputs"] = None
        return pack_serialized_compile_artifact_bundle(
            pack_serialized_fn_state_bundle(state), sidecar
        )

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> "VllmSerializableFunction":
        from torch._guards import TracingContext, tracing
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        sidecar, payload = unpack_serialized_compile_artifact_bundle(data)
        state = unpack_serialized_fn_state_bundle(payload, eager_pickle=False)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

        standalone_compile_artifact_store_bundle = state.pop(
            "standalone_compile_artifact_store_bundle", None
        )
        standalone_compile_artifacts = (
            unpack_standalone_artifact_store_bundle(
                standalone_compile_artifact_store_bundle
            )
            if standalone_compile_artifact_store_bundle is not None
            else None
        )
        sidecar_metadata = sidecar or {}
        standalone_compile_artifact_manifest = sidecar_metadata.get(
            "artifact_manifest"
        ) or state.pop("standalone_compile_artifact_manifest", None)
        standalone_compile_artifact_store_identity = sidecar_metadata.get(
            "store_identity"
        ) or state.pop("standalone_compile_artifact_store_identity", None)
        standalone_compile_artifact_placement_identity = sidecar_metadata.get(
            "placement_identity"
        ) or state.pop("standalone_compile_artifact_placement_identity", None)
        standalone_compile_artifact_compatibility = sidecar_metadata.get(
            "compatibility"
        ) or state.pop("standalone_compile_artifact_compatibility", None)
        standalone_compile_artifact_reuse_summary = sidecar_metadata.get(
            "reuse_summary"
        ) or state.pop("standalone_compile_artifact_reuse_summary", None)
        standalone_compile_artifact_proof_manifest = sidecar_metadata.get(
            "proof_manifest"
        ) or state.pop("standalone_compile_artifact_proof_manifest", None)
        sym_shape_indices_map = {
            name: tuple(indices)
            for name, indices in (
                sidecar_metadata.get("sym_shape_indices_map")
                or state.pop("sym_shape_indices_map", {})
            ).items()
        }
        returns_tuple_map = {
            name: bool(returns_tuple)
            for name, returns_tuple in (
                sidecar_metadata.get("returns_tuple_map")
                or state.pop("returns_tuple_map", {})
            ).items()
        }
        example_input_tensor_specs = sidecar_metadata.get("example_input_tensor_specs")

        materialize_serialized_fn_state_fields(state, "aot_autograd_config")
        saved_aot_autograd_config = state["aot_autograd_config"]
        if saved_aot_autograd_config is not None:
            functorch_ctx = torch._functorch.config.patch(saved_aot_autograd_config)
        else:
            functorch_ctx = contextlib.nullcontext()

        if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            assert standalone_compile_artifacts is not None
            current_vllm_config = get_current_vllm_config()
            if current_vllm_config.compilation_config.cudagraph_copy_inputs:
                if example_input_tensor_specs is not None:
                    state["example_inputs"] = build_sparse_example_inputs_from_tensor_specs(
                        example_input_tensor_specs
                    )
                elif state["example_inputs"] is not None:
                    state["example_inputs"] = GraphPickler.loads(
                        state["example_inputs"], fake_mode
                    )
                else:
                    state["example_inputs"] = []
            else:
                state["example_inputs"] = []
            submod_names = standalone_compile_artifacts.submodule_names()
            num_submods = len(submod_names)
            num_artifacts = standalone_compile_artifacts.num_artifacts()
            if standalone_compile_artifact_manifest is not None:
                manifest_with_identity = dict(standalone_compile_artifact_manifest)
                manifest_with_identity["store_identity"] = (
                    standalone_compile_artifact_store_identity
                )
                manifest_with_identity["placement_identity"] = (
                    standalone_compile_artifact_placement_identity
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
                compiled_artifacts_saved_before = (
                    compilation_counter.num_compiled_artifacts_saved
                )
                compiled_artifacts_loaded_before = (
                    compilation_counter.num_compiled_artifacts_loaded
                )
                fn = reconstruct_serializable_fn_from_mega_artifact(
                    state=state,
                    standalone_compile_artifacts=standalone_compile_artifacts,
                    vllm_config=current_vllm_config,
                    sym_shape_indices_map=sym_shape_indices_map,
                    returns_tuple_map=returns_tuple_map,
                    fake_mode=fake_mode,
                )
                load_report = standalone_compile_artifacts.last_load_report()
                no_new_compile_verification = verify_no_new_compile(
                    (
                        standalone_compile_artifact_proof_manifest.get(
                            "no_new_compile_expectation"
                        )
                        if standalone_compile_artifact_proof_manifest is not None
                        else None
                    ),
                    compiled_artifacts_saved_before=compiled_artifacts_saved_before,
                    compiled_artifacts_saved_after=(
                        compilation_counter.num_compiled_artifacts_saved
                    ),
                    compiled_artifacts_loaded_before=compiled_artifacts_loaded_before,
                    compiled_artifacts_loaded_after=(
                        compilation_counter.num_compiled_artifacts_loaded
                    ),
                    load_report=load_report,
                )

            logger.info(
                "reconstructed serializable fn from standalone compile "
                "artifacts. num_artifacts=%d num_submods=%d no_new_compile=%s",
                num_artifacts,
                num_submods,
                json.dumps(
                    no_new_compile_verification,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

            state.pop("_serialized_blob_codecs", None)
            return fn

        if state["example_inputs"] is None or state["graph_module"] is None:
            raise ValueError(
                "serialized function state missing graph fallback payloads"
            )
        state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)
        state["graph_module"] = cls.deserialize_graph_module(
            state["graph_module"], fake_mode
        )
        state["graph_module"].recompile()
        state.pop("_serialized_blob_codecs", None)

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

    piecewise_submod_names = standalone_compile_artifacts.submodule_names()
    compiled_callables: dict[str, dict[str, Callable[..., Any]]] = {}

    for cache_key in standalone_compile_artifacts.submodule_bytes:
        submod_name, shape_str = cache_key.rsplit("_", 1)
        compiled_callables.setdefault(submod_name, {})[shape_str] = (
            standalone_compile_artifacts.build_lazy_loaded_artifact(
                standalone_compile_artifacts.submodule_bytes[cache_key]
            )
        )

    standalone_compile_artifacts.mark_deferred_materialization()
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
            "standalone compile artifact load complete. loaded_artifacts=%d deserialization_wall_time_ms=%.6f load_path=%s topology=%s startup_closure=%s",
            load_report.get("loaded_artifact_count", 0),
            float(load_report.get("deserialization_wall_time_ms", 0.0)),
            load_report.get("load_path", "<unknown>"),
            json.dumps(
                load_report.get("load_topology", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            json.dumps(startup_closure, sort_keys=True, separators=(",", ":")),
        )

    # Use codegen'd execution code if available, fall back to split_gm
    execution_plan = state.get("execution_plan")
    execution_code = state.get("execution_code")
    submod_names = state.get("submod_names")
    if execution_plan is not None and submod_names is not None:
        materialize_serialized_fn_state_fields(state, "consts")
        consts = state.get("consts")
        runtime_callable = compile_execution_plan_fn(
            execution_plan, submod_callables, submod_names, consts
        )
    elif execution_code is not None and submod_names is not None:
        materialize_serialized_fn_state_fields(state, "consts")
        consts = state.get("consts")
        runtime_callable = compile_execution_fn(
            execution_code, submod_callables, submod_names, consts
        )
    else:
        logger.warning(
            "No execution code found, falling back to graph module execution."
        )
        if state.get("graph_module") is None:
            raise ValueError(
                "missing graph module bytes for mega-artifact graph fallback"
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

    state.pop("_serialized_blob_codecs", None)
    fn = VllmSerializableFunction(
        **state,
        optimized_call=optimized_call,
        vllm_backend=None,
    )
    return fn


def aot_compile_hash_factors(vllm_config: VllmConfig) -> dict[str, object]:
    """Return the canonical factors used to key AOT compile-adjacent caches."""

    env_identity = envs.compile_factor_identity_manifest()
    env_factors = envs.compile_factors()
    return {
        "env_identity": env_identity,
        "env_factors": env_factors,
        "env_policy_hash": hash_factors(env_factors),
        "vllm_config_hash": vllm_config.compute_hash(),
        "inductor_factors": (
            list(get_inductor_factors()) if envs.VLLM_USE_MEGA_AOT_ARTIFACT else []
        ),
        "mega_aot_enabled": envs.VLLM_USE_MEGA_AOT_ARTIFACT,
    }


def build_aot_compile_plan(
    *,
    vllm_config: VllmConfig,
    model_key: str,
    cache_enabled: bool,
    rank: int,
    data_parallel_rank: int,
) -> dict[str, object]:
    env_identity = envs.compile_factor_identity_manifest()
    env_factors = envs.compile_factors()
    inductor_factors = (
        list(get_inductor_factors()) if envs.VLLM_USE_MEGA_AOT_ARTIFACT else []
    )
    requested_policy = {
        "env_identity": env_identity,
        "model_key": model_key,
        "mega_aot_enabled": envs.VLLM_USE_MEGA_AOT_ARTIFACT,
    }
    normalized_policy = {
        "env_policy_hash": hash_factors(env_factors),
        "env_factor_digest": env_identity["combined_factor_digest"],
        "vllm_config_hash": vllm_config.compute_hash(),
        "model_key": model_key,
        "inductor_factors": inductor_factors,
        "mega_aot_enabled": envs.VLLM_USE_MEGA_AOT_ARTIFACT,
    }
    resolved_aot_plan = {
        "normalized_policy_hash": _stable_digest(normalized_policy),
        "artifact_kind": "torch_aot_compile",
    }
    materialization_plan = {
        "cache_enabled": cache_enabled,
        "rank": rank,
        "data_parallel_rank": data_parallel_rank,
    }

    plan = {
        "schema_version": 1,
        "requested_policy": requested_policy,
        "requested_policy_id": _stable_digest(requested_policy),
        "normalized_policy": normalized_policy,
        "normalized_policy_id": _stable_digest(normalized_policy),
        "resolved_aot_plan": resolved_aot_plan,
        "resolved_aot_plan_id": _stable_digest(resolved_aot_plan),
        "materialization_plan": materialization_plan,
        "materialization_plan_id": _stable_digest(materialization_plan),
    }
    plan["canonical_aot_plan_id"] = plan["resolved_aot_plan_id"]
    return plan


def render_aot_compile_plan(plan: dict[str, object]) -> str:
    return json.dumps(_json_ready(plan), sort_keys=True, separators=(",", ":"))


def artifact_entry_key(submod_name: str, shape: str) -> str:
    return f"{submod_name}_{shape}"


def artifact_bytes_hash(entry: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(entry)
    return hasher.hexdigest()


class LazyLoadedArtifact:
    def __init__(
        self,
        standalone_compile_artifacts: "StandaloneCompiledArtifacts",
        artifact_digest: str,
    ) -> None:
        self.standalone_compile_artifacts = standalone_compile_artifacts
        self.artifact_digest = artifact_digest

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        loaded_artifact = self.standalone_compile_artifacts._load_digest(
            self.artifact_digest
        )
        return loaded_artifact(*args, **kwargs)


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


def build_compile_root_identity(
    *,
    local_cache_dir: str | None,
    cache_key_factors: dict[str, object] | None,
    backend_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    canonical_compile_plan = (
        cache_key_factors.get("canonical_compile_plan")
        if cache_key_factors is not None
        else None
    )
    root = Path(local_cache_dir) if local_cache_dir else None
    return {
        "schema_version": 1,
        "cache_kind": "torch_compile_cache",
        "local_cache_dir": str(root) if root is not None else None,
        "root_plan_kind": (
            "canonical_compile_plan" if canonical_compile_plan is not None else None
        ),
        "root_plan_id": (
            canonical_compile_plan.get("canonical_compile_plan_id")
            if canonical_compile_plan is not None
            else None
        ),
        "requested_policy_id": (
            canonical_compile_plan.get("requested_policy_id")
            if canonical_compile_plan is not None
            else None
        ),
        "normalized_policy_id": (
            canonical_compile_plan.get("normalized_policy_id")
            if canonical_compile_plan is not None
            else None
        ),
        "resolved_plan_id": (
            canonical_compile_plan.get("resolved_compile_plan_id")
            if canonical_compile_plan is not None
            else None
        ),
        "materialization_plan_id": (
            canonical_compile_plan.get("materialization_plan_id")
            if canonical_compile_plan is not None
            else None
        ),
        "verification_plan_id": (
            canonical_compile_plan.get("verification_plan_id")
            if canonical_compile_plan is not None
            else None
        ),
        "backend_identity": _json_ready(backend_identity),
    }


def build_compile_replay_plan(
    cache_key_factors: dict[str, object] | None,
) -> dict[str, object] | None:
    canonical_compile_plan = (
        cache_key_factors.get("canonical_compile_plan")
        if cache_key_factors is not None
        else None
    )
    if canonical_compile_plan is None:
        return None
    return {
        "schema_version": 1,
        "replay_plan_kind": "canonical_compile_verification_plan",
        "root_plan_id": canonical_compile_plan.get("canonical_compile_plan_id"),
        "replay_plan_id": canonical_compile_plan.get("verification_plan_id"),
        "replay_plan": _json_ready(canonical_compile_plan.get("verification_plan")),
    }


def build_graph_artifact_store_manifest(
    *,
    local_cache_dir: str | None,
    cache_key_factors: dict[str, object] | None,
    artifact_files: dict[str, str] | None = None,
    backend_identity: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None

    root = Path(local_cache_dir)
    artifact_layout = artifact_files or {
        "cache_key_factors": "cache_key_factors.json",
        "compiler_cache": "vllm_compile_cache.py",
        "computation_graph": "computation_graph.py",
    }

    artifacts = []
    for artifact_kind, relative_path in sorted(artifact_layout.items()):
        artifact_path = Path(relative_path)
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        present = artifact_path.exists()
        artifact_record: dict[str, object] = {
            "artifact_kind": artifact_kind,
            "relative_path": str(
                artifact_path.relative_to(root) if present else Path(relative_path)
            ),
            "present": present,
            "size_bytes": artifact_path.stat().st_size if present else None,
            "sha256": None,
        }
        if present:
            artifact_record["sha256"] = hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest()
        artifacts.append(artifact_record)

    compile_hashes = {
        "env_policy_hash": (
            hash_factors(cache_key_factors["env"])
            if cache_key_factors is not None and "env" in cache_key_factors
            else hash_factors(envs.compile_factors())
        ),
        "config_hash": (
            cache_key_factors.get("config_hash") if cache_key_factors is not None else None
        ),
        "code_hash": (
            cache_key_factors.get("code_hash") if cache_key_factors is not None else None
        ),
        "compiler_hash": (
            cache_key_factors.get("compiler_hash")
            if cache_key_factors is not None
            else None
        ),
    }

    return {
        "schema_version": 1,
        "payload_kind": "vllm_graph_artifact_store",
        "store_kind": "torch_compile_cache",
        "local_cache_dir": str(root),
        "root_identity": build_compile_root_identity(
            local_cache_dir=local_cache_dir,
            cache_key_factors=cache_key_factors,
            backend_identity=backend_identity,
        ),
        "replay_plan": build_compile_replay_plan(cache_key_factors),
        "compile_hashes": compile_hashes,
        "cache_key_factors_source": (
            "cache_key_factors_file" if cache_key_factors is not None else "live_fallback"
        ),
        "source_fingerprint": (
            _json_ready(cache_key_factors.get("source_fingerprint"))
            if cache_key_factors is not None
            else None
        ),
        "env_identity": (
            _json_ready(cache_key_factors.get("env_identity"))
            if cache_key_factors is not None
            else None
        ),
        "compile_surface_fingerprint": (
            _json_ready(cache_key_factors.get("compile_surface_fingerprint"))
            if cache_key_factors is not None
            else None
        ),
        "canonical_compile_plan": (
            _json_ready(cache_key_factors.get("canonical_compile_plan"))
            if cache_key_factors is not None
            else None
        ),
        "canonical_compile_plan_id": (
            cache_key_factors.get("canonical_compile_plan", {}).get(
                "canonical_compile_plan_id"
            )
            if cache_key_factors is not None
            else None
        ),
        "backend_identity": _json_ready(backend_identity),
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts if item["present"]),
        "artifacts": artifacts,
    }


def write_graph_artifact_store_manifest(
    *,
    local_cache_dir: str | None,
    cache_key_factors: dict[str, object] | None,
    artifact_files: dict[str, str] | None = None,
    backend_identity: dict[str, object] | None = None,
) -> dict[str, object] | None:
    manifest = build_graph_artifact_store_manifest(
        local_cache_dir=local_cache_dir,
        cache_key_factors=cache_key_factors,
        artifact_files=artifact_files,
        backend_identity=backend_identity,
    )
    if manifest is None:
        return None

    meta_path = Path(local_cache_dir) / "graph_artifact_store.json"
    meta_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def load_graph_artifact_store_manifest(
    local_cache_dir: str | None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    meta_path = Path(local_cache_dir) / "graph_artifact_store.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning(
            "could not read graph artifact store manifest from %s", meta_path
        )
        return None


def build_compile_replay_manifest(
    *,
    local_cache_dir: str | None,
    cache_key_factors: dict[str, object] | None,
    graph_artifact_store: dict[str, object] | None,
    backend_identity: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    return {
        "schema_version": 1,
        "payload_kind": "vllm_compile_replay_manifest",
        "root_identity": build_compile_root_identity(
            local_cache_dir=local_cache_dir,
            cache_key_factors=cache_key_factors,
            backend_identity=backend_identity,
        ),
        "replay_plan": build_compile_replay_plan(cache_key_factors),
        "env_identity": (
            _json_ready(cache_key_factors.get("env_identity"))
            if cache_key_factors is not None
            else None
        ),
        "canonical_compile_plan": (
            _json_ready(cache_key_factors.get("canonical_compile_plan"))
            if cache_key_factors is not None
            else None
        ),
        "canonical_compile_plan_id": (
            cache_key_factors.get("canonical_compile_plan", {}).get(
                "canonical_compile_plan_id"
            )
            if cache_key_factors is not None
            else None
        ),
        "graph_artifact_store": _json_ready(graph_artifact_store),
        "backend_identity": _json_ready(backend_identity),
    }


def build_derived_compile_artifact_provenance(
    compile_replay_manifest: dict[str, object] | None,
) -> dict[str, object]:
    canonical_compile_plan = (
        _json_ready(compile_replay_manifest.get("canonical_compile_plan"))
        if compile_replay_manifest is not None
        else None
    )
    canonical_compile_plan_id = (
        canonical_compile_plan.get("canonical_compile_plan_id")
        if isinstance(canonical_compile_plan, dict)
        else None
    )
    return {
        "root_identity": (
            _json_ready(compile_replay_manifest.get("root_identity"))
            if compile_replay_manifest is not None
            else None
        ),
        "replay_plan": (
            _json_ready(compile_replay_manifest.get("replay_plan"))
            if compile_replay_manifest is not None
            else None
        ),
        "env_identity": (
            _json_ready(compile_replay_manifest.get("env_identity"))
            if compile_replay_manifest is not None
            else None
        ),
        "canonical_compile_plan": canonical_compile_plan,
        "canonical_compile_plan_id": canonical_compile_plan_id,
    }


def write_compile_replay_manifest(
    *,
    local_cache_dir: str | None,
    cache_key_factors: dict[str, object] | None,
    graph_artifact_store: dict[str, object] | None,
    backend_identity: dict[str, object] | None = None,
) -> dict[str, object] | None:
    manifest = build_compile_replay_manifest(
        local_cache_dir=local_cache_dir,
        cache_key_factors=cache_key_factors,
        graph_artifact_store=graph_artifact_store,
        backend_identity=backend_identity,
    )
    if manifest is None:
        return None

    meta_path = Path(local_cache_dir) / "compile_replay_manifest.json"
    meta_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def load_compile_replay_manifest(
    local_cache_dir: str | None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    meta_path = Path(local_cache_dir) / "compile_replay_manifest.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning("could not read compile replay manifest from %s", meta_path)
        return None


def build_cudagraph_capture_manifest(
    *,
    local_cache_dir: str | None,
    compile_replay_manifest: dict[str, object] | None,
    runtime_mode: str,
    cudagraph_capture_sizes: list[int] | None,
    captured_entries: list[dict[str, object]],
    cudagraph_options: dict[str, object] | None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    provenance = build_derived_compile_artifact_provenance(compile_replay_manifest)
    return {
        "schema_version": 1,
        "payload_kind": "vllm_cudagraph_capture_manifest",
        "root_identity": provenance["root_identity"],
        "replay_plan": provenance["replay_plan"],
        "env_identity": provenance["env_identity"],
        "canonical_compile_plan": provenance["canonical_compile_plan"],
        "canonical_compile_plan_id": provenance["canonical_compile_plan_id"],
        "runtime_mode": runtime_mode,
        "capture_size_policy": list(cudagraph_capture_sizes or []),
        "capture_count": len(captured_entries),
        "captured_entries": _json_ready(captured_entries),
        "cudagraph_options": _json_ready(cudagraph_options),
    }


def write_cudagraph_capture_manifest(
    *,
    local_cache_dir: str | None,
    compile_replay_manifest: dict[str, object] | None,
    runtime_mode: str,
    cudagraph_capture_sizes: list[int] | None,
    captured_entries: list[dict[str, object]],
    cudagraph_options: dict[str, object] | None,
) -> dict[str, object] | None:
    manifest = build_cudagraph_capture_manifest(
        local_cache_dir=local_cache_dir,
        compile_replay_manifest=compile_replay_manifest,
        runtime_mode=runtime_mode,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        captured_entries=captured_entries,
        cudagraph_options=cudagraph_options,
    )
    if manifest is None:
        return None
    meta_path = Path(local_cache_dir) / "cudagraph_capture_manifest.json"
    meta_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def load_cudagraph_capture_manifest(
    local_cache_dir: str | None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    meta_path = Path(local_cache_dir) / "cudagraph_capture_manifest.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning("could not read cudagraph capture manifest from %s", meta_path)
        return None


def build_autotune_cache_manifest(
    *,
    local_cache_dir: str | None,
    compile_replay_manifest: dict[str, object] | None,
    backend_name: str,
    base_cache_dir: str,
    cache_directories: list[dict[str, object]],
    environment_overrides: dict[str, object],
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    provenance = build_derived_compile_artifact_provenance(compile_replay_manifest)
    return {
        "schema_version": 1,
        "payload_kind": "vllm_autotune_cache_manifest",
        "backend_name": backend_name,
        "owning_local_cache_dir": local_cache_dir,
        "base_cache_dir": base_cache_dir,
        "root_identity": provenance["root_identity"],
        "replay_plan": provenance["replay_plan"],
        "env_identity": provenance["env_identity"],
        "canonical_compile_plan": provenance["canonical_compile_plan"],
        "canonical_compile_plan_id": provenance["canonical_compile_plan_id"],
        "cache_directories": _json_ready(cache_directories),
        "environment_overrides": _json_ready(environment_overrides),
    }


def build_warmup_materialization_manifest(
    *,
    local_cache_dir: str | None,
    compile_replay_manifest: dict[str, object] | None,
    worker_execution_mode: str,
    warmup_sizes: list[int] | None,
    cudagraph_capture_sizes: list[int] | None,
    cuda_graph_memory_bytes: int | None,
    stages: list[dict[str, object]],
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    provenance = build_derived_compile_artifact_provenance(compile_replay_manifest)
    normalized_stages = _json_ready(stages)
    executed_stage_count = sum(
        1
        for item in normalized_stages
        if isinstance(item, dict) and item.get("status") == "executed"
    )
    return {
        "schema_version": 1,
        "payload_kind": "vllm_warmup_materialization_manifest",
        "owning_local_cache_dir": local_cache_dir,
        "worker_execution_mode": worker_execution_mode,
        "root_identity": provenance["root_identity"],
        "replay_plan": provenance["replay_plan"],
        "env_identity": provenance["env_identity"],
        "canonical_compile_plan": provenance["canonical_compile_plan"],
        "canonical_compile_plan_id": provenance["canonical_compile_plan_id"],
        "warmup_sizes": list(warmup_sizes or []),
        "cudagraph_capture_sizes": list(cudagraph_capture_sizes or []),
        "cuda_graph_memory_bytes": cuda_graph_memory_bytes,
        "stage_count": len(normalized_stages),
        "executed_stage_count": executed_stage_count,
        "stages": normalized_stages,
    }


def write_autotune_cache_manifest(
    *,
    local_cache_dir: str | None,
    compile_replay_manifest: dict[str, object] | None,
    backend_name: str,
    base_cache_dir: str,
    cache_directories: list[dict[str, object]],
    environment_overrides: dict[str, object],
) -> dict[str, object] | None:
    manifest = build_autotune_cache_manifest(
        local_cache_dir=local_cache_dir,
        compile_replay_manifest=compile_replay_manifest,
        backend_name=backend_name,
        base_cache_dir=base_cache_dir,
        cache_directories=cache_directories,
        environment_overrides=environment_overrides,
    )
    if manifest is None:
        return None
    Path(base_cache_dir).mkdir(parents=True, exist_ok=True)
    meta_path = Path(base_cache_dir) / "autotune_cache_manifest.json"
    meta_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def write_warmup_materialization_manifest(
    *,
    local_cache_dir: str | None,
    compile_replay_manifest: dict[str, object] | None,
    worker_execution_mode: str,
    warmup_sizes: list[int] | None,
    cudagraph_capture_sizes: list[int] | None,
    cuda_graph_memory_bytes: int | None,
    stages: list[dict[str, object]],
) -> dict[str, object] | None:
    manifest = build_warmup_materialization_manifest(
        local_cache_dir=local_cache_dir,
        compile_replay_manifest=compile_replay_manifest,
        worker_execution_mode=worker_execution_mode,
        warmup_sizes=warmup_sizes,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        cuda_graph_memory_bytes=cuda_graph_memory_bytes,
        stages=stages,
    )
    if manifest is None:
        return None
    meta_path = Path(local_cache_dir) / "warmup_materialization_manifest.json"
    meta_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def load_autotune_cache_manifest(
    base_cache_dir: str | None,
) -> dict[str, object] | None:
    if not base_cache_dir:
        return None
    meta_path = Path(base_cache_dir) / "autotune_cache_manifest.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning("could not read autotune cache manifest from %s", meta_path)
        return None


def load_warmup_materialization_manifest(
    local_cache_dir: str | None,
) -> dict[str, object] | None:
    if not local_cache_dir:
        return None
    meta_path = Path(local_cache_dir) / "warmup_materialization_manifest.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning(
            "could not read warmup materialization manifest from %s", meta_path
        )
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
    local_cache_dir = getattr(compilation_config, "local_cache_dir", None)
    cache_key_factors = load_compile_cache_key_factors(
        local_cache_dir
    )
    graph_artifact_store = load_graph_artifact_store_manifest(local_cache_dir)
    compile_replay_manifest = load_compile_replay_manifest(local_cache_dir)
    if graph_artifact_store is None:
        graph_artifact_store = build_graph_artifact_store_manifest(
            local_cache_dir=local_cache_dir,
            cache_key_factors=cache_key_factors,
            backend_identity={
                "backend_class": type(vllm_backend).__name__,
                "prefix": getattr(vllm_backend, "prefix", None),
                "is_encoder": bool(getattr(vllm_backend, "is_encoder", False)),
                "compiler_name": (
                    getattr(getattr(compiler_manager, "compiler", None), "name", None)
                    if compiler_manager is not None
                    else None
                ),
            },
        )
    if compile_replay_manifest is None:
        compile_replay_manifest = build_compile_replay_manifest(
            local_cache_dir=local_cache_dir,
            cache_key_factors=cache_key_factors,
            graph_artifact_store=graph_artifact_store,
            backend_identity={
                "backend_class": type(vllm_backend).__name__,
                "prefix": getattr(vllm_backend, "prefix", None),
                "is_encoder": bool(getattr(vllm_backend, "is_encoder", False)),
                "compiler_name": (
                    getattr(getattr(compiler_manager, "compiler", None), "name", None)
                    if compiler_manager is not None
                    else None
                ),
            },
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
        "no_new_compile_expectation": build_no_new_compile_expectation(),
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
        "source_fingerprint": (
            _json_ready(cache_key_factors.get("source_fingerprint"))
            if cache_key_factors is not None
            else None
        ),
        "env_identity": (
            _json_ready(cache_key_factors.get("env_identity"))
            if cache_key_factors is not None
            else None
        ),
        "compile_surface_fingerprint": (
            _json_ready(cache_key_factors.get("compile_surface_fingerprint"))
            if cache_key_factors is not None
            else None
        ),
        "canonical_compile_plan": (
            _json_ready(cache_key_factors.get("canonical_compile_plan"))
            if cache_key_factors is not None
            else None
        ),
        "root_identity": (
            _json_ready(compile_replay_manifest.get("root_identity"))
            if compile_replay_manifest is not None
            else None
        ),
        "replay_plan": (
            _json_ready(compile_replay_manifest.get("replay_plan"))
            if compile_replay_manifest is not None
            else None
        ),
        "compile_replay_manifest": compile_replay_manifest,
        "graph_artifact_store": graph_artifact_store,
        "patch_profile": env_override.patch_profile_manifest(),
        "fallback_namespace_coverage": env_override.fallback_namespace_manifest(),
        "fallback_creation_evidence": env_override.fallback_creation_evidence_manifest(),
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


def verify_no_new_compile(
    expectation: dict[str, object] | None,
    *,
    compiled_artifacts_saved_before: int,
    compiled_artifacts_saved_after: int,
    compiled_artifacts_loaded_before: int,
    compiled_artifacts_loaded_after: int,
    load_report: dict[str, object] | None,
) -> dict[str, object]:
    if expectation is None:
        return {
            "schema_version": 1,
            "ok": False,
            "expected_new_compiled_artifacts": None,
            "actual_new_compiled_artifacts": None,
            "actual_loaded_artifacts": None,
            "reasons": ["no_new_compile_expectation_missing"],
        }

    expected_new_compiled_artifacts = int(
        expectation.get("expected_new_compiled_artifacts", 0)
    )
    actual_new_compiled_artifacts = (
        compiled_artifacts_saved_after - compiled_artifacts_saved_before
    )
    actual_loaded_artifacts = (
        compiled_artifacts_loaded_after - compiled_artifacts_loaded_before
    )
    reasons = []
    if actual_new_compiled_artifacts != expected_new_compiled_artifacts:
        reasons.append("unexpected_new_compiled_artifacts")
    if load_report is None:
        reasons.append("load_report_missing")

    return {
        "schema_version": 1,
        "ok": not reasons,
        "expected_new_compiled_artifacts": expected_new_compiled_artifacts,
        "actual_new_compiled_artifacts": actual_new_compiled_artifacts,
        "actual_loaded_artifacts": actual_loaded_artifacts,
        "reasons": reasons,
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
    plan = build_aot_compile_plan(
        vllm_config=vllm_config,
        model_key="<manifest_only>",
        cache_enabled=not envs.VLLM_DISABLE_COMPILE_CACHE,
        rank=0,
        data_parallel_rank=0,
    )
    manifest = {
        "schema_version": 1,
        "env": envs.compile_factor_manifest(),
        "vllm_config_hash": vllm_config.compute_hash(),
        "resolved_compilation_policy": (
            vllm_config.resolved_compilation_policy_manifest()
            if hasattr(vllm_config, "resolved_compilation_policy_manifest")
            else {
                "schema_version": 1,
                "status": "unavailable",
                "reason": (
                    "config_does_not_expose_resolved_compilation_policy_manifest"
                ),
            }
        ),
        "inductor_factors": (
            get_inductor_factors() if envs.VLLM_USE_MEGA_AOT_ARTIFACT else []
        ),
        "aot_compile_plan": plan,
        "aot_compile_plan_id": plan["canonical_aot_plan_id"],
    }
    return json.dumps(_json_ready(manifest), sort_keys=True, separators=(",", ":"))


def _fingerprint_python_source(
    source: str,
    reachable_symbols: Sequence[str] | None = None,
) -> tuple[str, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "raw_text_fallback", source
    if not reachable_symbols:
        return "python_ast", ast.dump(tree, include_attributes=False)

    reachable_set = {
        symbol for symbol in reachable_symbols if symbol and "<locals>" not in symbol
    }
    if not reachable_set:
        return "python_ast", ast.dump(tree, include_attributes=False)

    selected_nodes = _select_reachable_module_nodes(tree, reachable_set)
    selected_dump = [
        ast.dump(node, include_attributes=False) for node in selected_nodes
    ]
    return (
        "python_ast_reachable",
        json.dumps(selected_dump, sort_keys=True, separators=(",", ":")),
    )


def build_compile_source_fingerprint_from_content(
    file_contents: dict[str, str],
    reachable_symbols_by_path: dict[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    items = list(sorted(file_contents.items(), key=lambda x: x[0]))
    files = []
    aggregate_material = []
    reachable_symbols_by_path = reachable_symbols_by_path or {}

    for filepath, content in items:
        normalized_path = _normalize_compile_source_path(filepath)
        raw_sha256 = safe_hash(
            content.encode(), usedforsecurity=False
        ).hexdigest()
        fingerprint_mode = "path_only"
        fingerprint_material = ""
        reachable_symbols = _reachable_symbols_for_path(
            filepath, reachable_symbols_by_path
        )
        if filepath == "<string>" or filepath.startswith("<"):
            fingerprint_mode = "path_only"
        elif filepath.endswith(".py"):
            fingerprint_mode, fingerprint_material = _fingerprint_python_source(
                content, reachable_symbols=reachable_symbols
            )
        else:
            fingerprint_mode = "raw_text"
            fingerprint_material = content

        semantic_sha256 = safe_hash(
            fingerprint_material.encode(), usedforsecurity=False
        ).hexdigest()
        files.append(
            {
                "path": filepath,
                "normalized_path": normalized_path,
                "fingerprint_mode": fingerprint_mode,
                "raw_sha256": raw_sha256,
                "semantic_sha256": semantic_sha256,
                "size_bytes": len(content.encode()),
                "reachable_symbols": list(reachable_symbols),
            }
        )
        aggregate_material.extend((normalized_path, fingerprint_mode, semantic_sha256))

    return {
        "schema_version": 1,
        "file_count": len(files),
        "files": files,
        "aggregate_hash": safe_hash(
            "\n".join(aggregate_material).encode(), usedforsecurity=False
        ).hexdigest(),
    }


def build_compile_surface_fingerprint(
    *,
    source_fingerprint: dict[str, object] | None,
    graph_text: str,
    placeholder_names: Sequence[str],
    node_targets: Sequence[str],
    splitting_ops: Sequence[str] | None,
    custom_ops: Sequence[str] | None,
    enabled_passes: Sequence[str],
    inductor_passes: Sequence[str],
    dynamic_shapes_type: str,
    dynamic_shapes_evaluate_guards: bool,
    use_inductor_graph_partition: bool,
    enabled_custom_ops: dict[str, int] | None = None,
    disabled_custom_ops: dict[str, int] | None = None,
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "source_fingerprint_hash": (
            source_fingerprint.get("aggregate_hash")
            if source_fingerprint is not None
            else None
        ),
        "graph_sha256": safe_hash(
            graph_text.encode(), usedforsecurity=False
        ).hexdigest(),
        "placeholder_names": normalize_placeholder_names(placeholder_names),
        "node_targets": normalize_node_targets(node_targets),
        "splitting_ops": list(splitting_ops or []),
        "custom_ops": list(custom_ops or []),
        "enabled_passes": list(enabled_passes),
        "inductor_passes": list(inductor_passes),
        "dynamic_shapes_type": dynamic_shapes_type,
        "dynamic_shapes_evaluate_guards": dynamic_shapes_evaluate_guards,
        "use_inductor_graph_partition": use_inductor_graph_partition,
        "enabled_custom_ops": dict(sorted((enabled_custom_ops or {}).items())),
        "disabled_custom_ops": dict(sorted((disabled_custom_ops or {}).items())),
    }
    manifest["aggregate_hash"] = safe_hash(
        json.dumps(
            _json_ready(manifest),
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        usedforsecurity=False,
    ).hexdigest()
    return manifest


def _normalize_compile_source_path(filepath: str) -> str:
    normalized = filepath.replace("\\", "/")
    for marker in ("/torch/_inductor/", "/torch/_dynamo/"):
        if marker in normalized:
            return normalized.rsplit("/", 1)[-1]
    return normalized


def normalize_placeholder_names(placeholder_names: Sequence[str]) -> list[str]:
    return [f"arg{index}" for index, _ in enumerate(placeholder_names)]


def normalize_node_targets(node_targets: Sequence[str]) -> list[str]:
    placeholder_map = {}
    normalized = []
    next_index = 0
    for target in node_targets:
        if target.startswith("placeholder:"):
            _, name = target.split(":", 1)
            if name not in placeholder_map:
                placeholder_map[name] = f"arg{next_index}"
                next_index += 1
            normalized.append(f"placeholder:{placeholder_map[name]}")
        else:
            normalized.append(target)
    return normalized


def _reachable_symbols_for_path(
    filepath: str,
    reachable_symbols_by_path: dict[str, Sequence[str]],
) -> list[str]:
    symbols = reachable_symbols_by_path.get(filepath, ())
    return sorted(
        {
            symbol.split(".", 1)[0]
            for symbol in symbols
            if symbol and "<locals>" not in symbol
        }
    )


def _select_reachable_module_nodes(
    tree: ast.Module,
    reachable_symbols: set[str],
) -> list[ast.stmt]:
    entries = [_module_node_entry(node) for node in tree.body]
    required_symbols = set(reachable_symbols)
    selected_indices: set[int] = set()

    changed = True
    while changed:
        changed = False
        for index, entry in enumerate(entries):
            defined_symbols = entry["defined_symbols"]
            if not defined_symbols or selected_indices.__contains__(index):
                continue
            if defined_symbols.isdisjoint(required_symbols):
                continue
            selected_indices.add(index)
            required_symbols.update(entry["referenced_symbols"])
            changed = True

    selected_nodes = []
    for index, entry in enumerate(entries):
        if (
            index in selected_indices
            or entry["always_include"]
            or not entry["defined_symbols"]
        ):
            selected_nodes.append(entry["node"])
    return selected_nodes


def _module_node_entry(node: ast.stmt) -> dict[str, object]:
    defined_symbols = _defined_module_symbols(node)
    referenced_symbols = _referenced_module_symbols(node)
    always_include = isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.TryStar,
            ast.With,
            ast.AsyncWith,
            ast.Match,
            ast.Expr,
        ),
    )
    return {
        "node": node,
        "defined_symbols": defined_symbols,
        "referenced_symbols": referenced_symbols - defined_symbols,
        "always_include": always_include,
    }


def _defined_module_symbols(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = set()
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", 1)[0]
            names.add(bound_name)
        return names
    if isinstance(node, ast.Assign):
        return _defined_assignment_targets(node.targets)
    if isinstance(node, ast.AnnAssign):
        return _defined_assignment_targets([node.target])
    if isinstance(node, ast.AugAssign):
        return _defined_assignment_targets([node.target])
    return set()


def _defined_assignment_targets(targets: Sequence[ast.expr]) -> set[str]:
    names = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
    return names


def _referenced_module_symbols(node: ast.stmt) -> set[str]:
    referenced = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
            referenced.add(inner.id)
    return referenced


def _stable_digest(payload: object) -> str:
    return safe_hash(
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        usedforsecurity=False,
    ).hexdigest()


def build_canonical_compile_plan(
    *,
    env_factors: dict[str, object],
    env_identity: dict[str, object],
    config_hash: str,
    compiler_hash: str,
    source_fingerprint: dict[str, object],
    compile_surface_fingerprint: dict[str, object],
    backend_identity: dict[str, object],
    cache_enabled: bool,
    cache_namespace_prefix: str,
    rank: int,
    data_parallel_rank: int,
) -> dict[str, object]:
    requested_policy = {
        "env_identity": env_identity,
        "backend_identity": backend_identity,
        "cache_namespace_prefix": cache_namespace_prefix,
    }
    normalized_policy = {
        "env_policy_hash": hash_factors(env_factors),
        "env_factor_digest": env_identity["combined_factor_digest"],
        "config_hash": config_hash,
        "compiler_hash": compiler_hash,
        "backend_identity": backend_identity,
    }
    resolved_compile_plan = {
        "normalized_policy_hash": _stable_digest(normalized_policy),
        "source_fingerprint_hash": source_fingerprint["aggregate_hash"],
        "compile_surface_hash": compile_surface_fingerprint["aggregate_hash"],
        "backend_identity": backend_identity,
    }
    materialization_plan = {
        "cache_enabled": cache_enabled,
        "cache_namespace_prefix": cache_namespace_prefix,
        "rank": rank,
        "data_parallel_rank": data_parallel_rank,
    }
    verification_plan = {
        "proof_mode": "compile_plan_manifest_v1",
        "compile_surface_hash": compile_surface_fingerprint["aggregate_hash"],
        "source_fingerprint_hash": source_fingerprint["aggregate_hash"],
    }

    plan = {
        "schema_version": 1,
        "requested_policy": requested_policy,
        "requested_policy_id": _stable_digest(requested_policy),
        "normalized_policy": normalized_policy,
        "normalized_policy_id": _stable_digest(normalized_policy),
        "resolved_compile_plan": resolved_compile_plan,
        "resolved_compile_plan_id": _stable_digest(resolved_compile_plan),
        "materialization_plan": materialization_plan,
        "materialization_plan_id": _stable_digest(materialization_plan),
        "verification_plan": verification_plan,
        "verification_plan_id": _stable_digest(verification_plan),
    }
    plan["canonical_compile_plan_id"] = plan["resolved_compile_plan_id"]
    return plan


def render_canonical_compile_plan(plan: dict[str, object]) -> str:
    return json.dumps(_json_ready(plan), sort_keys=True, separators=(",", ":"))


def _compute_code_hash_with_content(file_contents: dict[str, str]) -> str:
    return build_compile_source_fingerprint_from_content(file_contents)["aggregate_hash"]


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
