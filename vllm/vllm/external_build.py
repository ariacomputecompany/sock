# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalDependencySpec:
    name: str
    cmake_var: str
    env_var: str
    cache_relative_source_dirs: tuple[str, ...]


@dataclass(frozen=True)
class ExternalDependencyResolution:
    name: str
    cmake_var: str
    env_var: str
    source_dir: str | None
    origin: str


_DEPENDENCY_SPECS = (
    ExternalDependencySpec(
        name="triton_kernels",
        cmake_var="TRITON_KERNELS_SRC_DIR",
        env_var="TRITON_KERNELS_SRC_DIR",
        cache_relative_source_dirs=(
            "triton_kernels-src/python/triton_kernels/triton_kernels",
        ),
    ),
    ExternalDependencySpec(
        name="vllm_flash_attn",
        cmake_var="VLLM_FLASH_ATTN_SRC_DIR",
        env_var="VLLM_FLASH_ATTN_SRC_DIR",
        cache_relative_source_dirs=("vllm-flash-attn-src",),
    ),
    ExternalDependencySpec(
        name="deepgemm",
        cmake_var="DEEPGEMM_SRC_DIR",
        env_var="DEEPGEMM_SRC_DIR",
        cache_relative_source_dirs=("deepgemm-src",),
    ),
    ExternalDependencySpec(
        name="qutlass",
        cmake_var="QUTLASS_SRC_DIR",
        env_var="QUTLASS_SRC_DIR",
        cache_relative_source_dirs=("qutlass-src",),
    ),
    ExternalDependencySpec(
        name="flashmla",
        cmake_var="FLASH_MLA_SRC_DIR",
        env_var="FLASH_MLA_SRC_DIR",
        cache_relative_source_dirs=("flashmla-src",),
    ),
    ExternalDependencySpec(
        name="fmha_sm100",
        cmake_var="FMHA_SM100_SRC_DIR",
        env_var="FMHA_SM100_SRC_DIR",
        cache_relative_source_dirs=("fmha_sm100-src",),
    ),
    ExternalDependencySpec(
        name="cutlass",
        cmake_var="VLLM_CUTLASS_SRC_DIR",
        env_var="VLLM_CUTLASS_SRC_DIR",
        cache_relative_source_dirs=("cutlass-src",),
    ),
)


def _normalize_source_dir(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def resolve_external_dependency_sources(
    fetchcontent_base_dir: str | Path,
) -> tuple[ExternalDependencyResolution, ...]:
    fetchcontent_base_dir = Path(fetchcontent_base_dir)
    resolutions: list[ExternalDependencyResolution] = []
    for spec in _DEPENDENCY_SPECS:
        explicit = os.getenv(spec.env_var, "").strip()
        if explicit:
            resolutions.append(
                ExternalDependencyResolution(
                    name=spec.name,
                    cmake_var=spec.cmake_var,
                    env_var=spec.env_var,
                    source_dir=_normalize_source_dir(explicit),
                    origin="env",
                )
            )
            continue

        cached_dir = None
        for relative_dir in spec.cache_relative_source_dirs:
            candidate = fetchcontent_base_dir / relative_dir
            if candidate.exists():
                cached_dir = str(candidate.resolve())
                break
        resolutions.append(
            ExternalDependencyResolution(
                name=spec.name,
                cmake_var=spec.cmake_var,
                env_var=spec.env_var,
                source_dir=cached_dir,
                origin="cache" if cached_dir else "unresolved",
            )
        )
    return tuple(resolutions)


def build_external_dependency_manifest(
    fetchcontent_base_dir: str | Path,
    resolutions: tuple[ExternalDependencyResolution, ...],
) -> dict:
    return {
        "fetchcontent_base_dir": str(Path(fetchcontent_base_dir).resolve()),
        "dependencies": [asdict(resolution) for resolution in resolutions],
    }


def write_external_dependency_manifest(
    manifest_path: str | Path,
    fetchcontent_base_dir: str | Path,
    resolutions: tuple[ExternalDependencyResolution, ...],
) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            build_external_dependency_manifest(fetchcontent_base_dir, resolutions),
            indent=2,
            sort_keys=True,
        )
    )


def resolved_cmake_args(
    resolutions: tuple[ExternalDependencyResolution, ...],
) -> tuple[str, ...]:
    return tuple(
        f"-D{resolution.cmake_var}={resolution.source_dir}"
        for resolution in resolutions
        if resolution.source_dir
    )


@dataclass(frozen=True)
class EditableSyncSpec:
    source_root: str
    destination_root: str
    patterns: tuple[str, ...] = ("**/*",)


def _should_sync_file(source_path: Path, destination_path: Path) -> bool:
    if not destination_path.exists():
        return True
    if source_path.stat().st_size != destination_path.stat().st_size:
        return True
    return source_path.read_bytes() != destination_path.read_bytes()


def sync_editable_install_roots(
    manifest_path: str | Path,
    specs: tuple[EditableSyncSpec, ...],
) -> None:
    manifest_entries: list[dict[str, object]] = []
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        source_root = Path(spec.source_root)
        destination_root = Path(spec.destination_root)
        if not source_root.exists():
            continue

        copied_files: list[str] = []
        seen_destinations: set[Path] = set()
        for pattern in spec.patterns:
            for source_path in sorted(source_root.glob(pattern)):
                if source_path.is_dir():
                    continue
                relative_path = source_path.relative_to(source_root)
                destination_path = destination_root / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                seen_destinations.add(destination_path)
                if _should_sync_file(source_path, destination_path):
                    shutil.copy2(source_path, destination_path)
                copied_files.append(str(relative_path))

        manifest_entries.append(
            {
                "source_root": str(source_root.resolve()),
                "destination_root": str(destination_root.resolve()),
                "patterns": list(spec.patterns),
                "files": copied_files,
            }
        )

    manifest_path.write_text(
        json.dumps({"editable_sync_roots": manifest_entries}, indent=2, sort_keys=True)
    )
