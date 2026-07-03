# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path


def _load_external_build():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "vllm" / "external_build.py"
    spec = importlib.util.spec_from_file_location("vllm_external_build", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


external_build = _load_external_build()


def test_resolve_external_dependency_sources_prefers_cache(tmp_path) -> None:
    triton_cache = (
        tmp_path / "triton_kernels-src" / "python" / "triton_kernels" / "triton_kernels"
    )
    deepgemm_cache = tmp_path / "deepgemm-src"
    triton_cache.mkdir(parents=True)
    deepgemm_cache.mkdir(parents=True)

    resolutions = external_build.resolve_external_dependency_sources(tmp_path)
    by_name = {resolution.name: resolution for resolution in resolutions}

    assert by_name["triton_kernels"].origin == "cache"
    assert by_name["triton_kernels"].source_dir == str(triton_cache.resolve())
    assert by_name["deepgemm"].origin == "cache"
    assert by_name["deepgemm"].source_dir == str(deepgemm_cache.resolve())
    assert by_name["flashmla"].origin == "unresolved"


def test_write_external_dependency_manifest_records_resolutions(tmp_path) -> None:
    resolutions = external_build.resolve_external_dependency_sources(tmp_path)
    manifest_path = tmp_path / "deps.json"

    external_build.write_external_dependency_manifest(
        manifest_path,
        tmp_path,
        resolutions,
    )

    manifest = manifest_path.read_text()
    assert "fetchcontent_base_dir" in manifest
    assert "triton_kernels" in manifest


def test_sync_editable_install_roots_writes_manifest_and_copies_deltas(
    tmp_path,
) -> None:
    source_root = tmp_path / "build" / "vllm" / "third_party" / "triton_kernels"
    destination_root = tmp_path / "src" / "vllm" / "third_party" / "triton_kernels"
    (source_root / "nested").mkdir(parents=True)
    (source_root / "nested" / "kernel.py").write_text("print('hi')\n")
    manifest_path = tmp_path / "editable.json"

    external_build.sync_editable_install_roots(
        manifest_path,
        (
            external_build.EditableSyncSpec(
                source_root=str(source_root),
                destination_root=str(destination_root),
                patterns=("**/*.py",),
            ),
        ),
    )

    synced_file = destination_root / "nested" / "kernel.py"
    assert synced_file.read_text() == "print('hi')\n"
    assert "kernel.py" in manifest_path.read_text()
