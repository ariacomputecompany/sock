from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ROCm-on-WSL prerequisites for vendored vLLM."
    )
    parser.add_argument(
        "--build-dlpack",
        action="store_true",
        help="Build the tvm_ffi ROCm torch-c-dlpack addon if it is missing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Reserved for future machine-readable output.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ensure_dlpack_addon() -> tuple[bool, str]:
    import torch

    cache_dir = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    major, minor = torch.__version__.split(".")[:2]
    libname = f"libtorch_c_dlpack_addon_torch{major}{minor}-rocm.so"
    libpath = cache_dir / libname
    if libpath.exists():
        return True, str(libpath)

    build_script = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "tvm_ffi"
        / "utils"
        / "_build_optional_torch_c_dlpack.py"
    )
    result = run(
        [
            sys.executable,
            str(build_script),
            "--output-dir",
            str(cache_dir),
            "--libname",
            libname,
            "--build-with-rocm",
        ]
    )
    if result.returncode != 0:
        summary = result.stderr.strip() or result.stdout.strip() or "unknown build error"
        return False, summary
    if not libpath.exists():
        return False, f"build completed but {libpath} was not created"
    return True, str(libpath)


def main() -> None:
    args = parse_args()

    import torch

    print(f"python={sys.executable}")
    print(f"torch_version={torch.__version__}")
    print(f"hip_version={torch.version.hip!r}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("ROCm device is not visible to torch.")

    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"rocm_path={os.environ.get('ROCM_PATH') or '/opt/rocm'}")
    print(f"ld_library_path={os.environ.get('LD_LIBRARY_PATH', '')}")

    hip_header = None
    for candidate in (
        Path("/usr/include/hip/hip_runtime_api.h"),
        Path("/opt/rocm/include/hip/hip_runtime_api.h"),
    ):
        if candidate.exists():
            hip_header = candidate
            break
    if hip_header is None:
        print(
            "missing_hip_header=1 install_hint='sudo apt-get install -y libamdhip64-dev libhsa-runtime-dev'"
        )
    else:
        print(f"hip_header={hip_header}")

    import tvm_ffi._optional_torch_c_dlpack as dlpack_mod

    print(f"dlpack_loaded={dlpack_mod._LIB is not None}")
    if dlpack_mod._LIB is not None:
        print(f"dlpack_library={dlpack_mod._LIB._name}")
    elif args.build_dlpack:
        ok, detail = ensure_dlpack_addon()
        print(f"dlpack_rebuild_ok={ok}")
        print(f"dlpack_rebuild_detail={detail}")
        if ok:
            import importlib

            dlpack_mod = importlib.reload(dlpack_mod)
            print(f"dlpack_loaded_after_rebuild={dlpack_mod._LIB is not None}")
            if dlpack_mod._LIB is not None:
                print(f"dlpack_library={dlpack_mod._LIB._name}")

    if os.environ.get("HF_TOKEN"):
        print("hf_token=present")
    else:
        print("hf_token=missing")

    if "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower():
        print("wsl_detected=1")
        print("wsl_pin_memory=vllm will force pin_memory=False on WSL")


if __name__ == "__main__":
    main()
