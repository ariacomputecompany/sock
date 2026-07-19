from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_SOURCE_ROOT = REPO_ROOT / "vllm"
BUILD_PLAN_PATH = REPO_ROOT / "runtime.buildplan.json"

if str(VLLM_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_SOURCE_ROOT))

_BUILD_PROFILES_PATH = VLLM_SOURCE_ROOT / "vllm" / "build_profiles.py"
_BUILD_PROFILES_SPEC = importlib.util.spec_from_file_location(
    "sock_vllm_build_profiles", _BUILD_PROFILES_PATH
)
if _BUILD_PROFILES_SPEC is None or _BUILD_PROFILES_SPEC.loader is None:
    raise SystemExit(f"Unable to load build profiles from {_BUILD_PROFILES_PATH}")
_BUILD_PROFILES = importlib.util.module_from_spec(_BUILD_PROFILES_SPEC)
_BUILD_PROFILES_SPEC.loader.exec_module(_BUILD_PROFILES)

resolve_build_profile = _BUILD_PROFILES.resolve_build_profile
supported_build_profile_csv = _BUILD_PROFILES.supported_build_profile_csv


@dataclass(frozen=True)
class CommandStep:
    name: str
    argv: list[str]
    cwd: Path
    env: dict[str, str]

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": self.argv,
            "cwd": str(self.cwd),
            "env": self.env,
        }


def load_build_plan() -> dict[str, Any]:
    with BUILD_PLAN_PATH.open() as f:
        plan = json.load(f)
    if plan.get("schema_version") != 1:
        raise SystemExit("Unsupported runtime.buildplan.json schema_version")
    return plan


def command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            argv,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        return None


def command_ok(argv: list[str]) -> bool:
    try:
        subprocess.run(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
        return True
    except Exception:
        return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_cuda_arches() -> list[str]:
    query = command_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    )
    if query:
        arches = [line.strip() for line in query.splitlines() if line.strip()]
        if arches:
            return arches

    try:
        import torch

        if torch.cuda.is_available():
            return [
                ".".join(str(part) for part in torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return []


def detect_rocm_arches() -> list[str]:
    rocm_agent = command_output(["rocm_agent_enumerator"])
    if rocm_agent:
        return [
            line.strip()
            for line in rocm_agent.splitlines()
            if line.strip().startswith("gfx")
        ]

    rocminfo = command_output(["rocminfo"])
    if rocminfo:
        arches = sorted(
            {
                part.strip()
                for line in rocminfo.splitlines()
                for part in line.replace(":", " ").split()
                if part.startswith("gfx")
            }
        )
        if arches:
            return arches
    return []


def python_header_path(python: str) -> str | None:
    return command_output(
        [
            python,
            "-c",
            (
                "import pathlib, sysconfig; "
                "print(pathlib.Path(sysconfig.get_paths()['include']) / 'Python.h')"
            ),
        ]
    )


def build_preflight(profile_name: str, profile_plan: dict[str, Any], python: str) -> dict[str, Any]:
    required_tools = ["git"]
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    c_compiler = next((tool for tool in ("cc", "gcc", "clang") if shutil.which(tool)), None)
    cxx_compiler = next(
        (tool for tool in ("c++", "g++", "clang++") if shutil.which(tool)), None
    )
    if c_compiler is None:
        missing_tools.append("cc|gcc|clang")
    if cxx_compiler is None:
        missing_tools.append("c++|g++|clang++")

    header = python_header_path(python)
    python_headers_ok = bool(header and Path(header).exists())
    python_venv_ok = command_ok([python, "-m", "venv", "--help"])
    accelerator_probe_ok = True
    accelerator_probe = "none"
    if profile_name == "cuda":
        accelerator_probe = "nvidia-smi"
        accelerator_probe_ok = shutil.which("nvidia-smi") is not None and command_ok(
            ["nvidia-smi", "-L"]
        )
    elif profile_name == "rocm":
        accelerator_probe = "rocminfo|rocm_agent_enumerator"
        accelerator_probe_ok = (
            shutil.which("rocm_agent_enumerator") is not None
            and command_ok(["rocm_agent_enumerator"])
        ) or (shutil.which("rocminfo") is not None and command_ok(["rocminfo"]))

    issues = []
    if missing_tools:
        issues.append("missing_build_tools")
    if not python_headers_ok:
        issues.append("missing_python_headers")
    if not python_venv_ok:
        issues.append("missing_python_venv")
    if not accelerator_probe_ok:
        issues.append("accelerator_probe_failed")

    return {
        "ok": not issues,
        "issues": issues,
        "required_system_packages": profile_plan["system_packages"],
        "missing_tools": missing_tools,
        "c_compiler": c_compiler,
        "cxx_compiler": cxx_compiler,
        "python_header": header,
        "python_headers_ok": python_headers_ok,
        "python_venv_ok": python_venv_ok,
        "accelerator_probe": accelerator_probe,
        "accelerator_probe_ok": accelerator_probe_ok,
    }


def require_preflight_ok(preflight: dict[str, Any]) -> None:
    if preflight["ok"]:
        return
    raise SystemExit(
        "Runtime preflight failed: "
        + ", ".join(preflight["issues"])
        + ". Install system packages first: "
        + ", ".join(preflight["required_system_packages"])
    )


def requirement_digests(requirements: list[str]) -> list[dict[str, str]]:
    return [
        {"path": requirement, "sha256": file_sha256(REPO_ROOT / requirement)}
        for requirement in requirements
    ]


def detect_profile(requested: str) -> tuple[str, list[str]]:
    normalized = requested.strip().lower()
    if normalized in {"cuda", "nvidia"}:
        return "cuda", detect_cuda_arches()
    if normalized in {"rocm", "amd"}:
        return "rocm", detect_rocm_arches()
    if normalized != "auto":
        raise SystemExit(f"Unsupported runtime profile '{requested}'. Use auto, cuda, or rocm.")

    cuda_arches = detect_cuda_arches()
    if cuda_arches:
        return "cuda", cuda_arches

    rocm_arches = detect_rocm_arches()
    if rocm_arches:
        return "rocm", rocm_arches

    raise SystemExit(
        "Unable to auto-detect a CUDA or ROCm accelerator. Pass --profile explicitly "
        "after installing the vendor driver/runtime."
    )


def select_build_profile(
    profile_plan: dict[str, Any], requested: str, accelerator_arches: list[str]
) -> str:
    if requested != "auto":
        resolve_build_profile(requested)
        return requested

    for arch in accelerator_arches:
        for rule in profile_plan.get("build_profile_by_compute_capability", []):
            if arch.startswith(rule["prefix"]):
                selected = rule["build_profile"]
                resolve_build_profile(selected)
                return selected

    selected = profile_plan["default_build_profile"]
    resolve_build_profile(selected)
    return selected


def resolve_env(
    profile_plan: dict[str, Any],
    build_profile: str,
    accelerator_arches: list[str],
) -> dict[str, str]:
    env = dict(profile_plan["env"])
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_BUILD_PROFILE"] = build_profile
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = load_build_plan()["pretend_version"]

    if profile_plan["target_device"] == "cuda" and accelerator_arches:
        env["TORCH_CUDA_ARCH_LIST"] = ";".join(accelerator_arches)

    cpu_count = os.cpu_count() or 4
    env["MAX_JOBS"] = os.environ.get("MAX_JOBS", str(max(1, min(8, cpu_count))))
    if profile_plan["target_device"] == "cuda":
        env["NVCC_THREADS"] = os.environ.get("NVCC_THREADS", "4")
    return env


def python_for_venv(venv_root: Path) -> Path:
    return venv_root / "bin" / "python"


def step_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra)
    return env


def prepend_path(env: dict[str, str], path: Path) -> dict[str, str]:
    next_env = dict(env)
    parts = [str(path)]
    existing = next_env.get("PATH") or os.environ.get("PATH")
    if existing:
        parts.append(existing)
    next_env["PATH"] = os.pathsep.join(parts)
    return next_env


def planned_steps(
    plan: dict[str, Any],
    profile_plan: dict[str, Any],
    build_profile: str,
    accelerator_arches: list[str],
    python: str,
    recreate_venv: bool,
) -> list[CommandStep]:
    venv_root = REPO_ROOT / plan["venv_path"]
    venv_python = python_for_venv(venv_root)
    env = prepend_path(resolve_env(profile_plan, build_profile, accelerator_arches), venv_root / "bin")
    pip = [str(venv_python), "-m", "pip"]
    steps = []
    if recreate_venv:
        steps.append(
            CommandStep(
                name="remove_existing_venv",
                argv=[
                    python,
                    "-c",
                    (
                        "import shutil; "
                        f"shutil.rmtree({str(venv_root)!r}, ignore_errors=True)"
                    ),
                ],
                cwd=REPO_ROOT,
                env={},
            )
        )

    steps.extend(
        [
            CommandStep(
                name="create_venv",
                argv=[python, "-m", "venv", str(venv_root)],
                cwd=REPO_ROOT,
                env={},
            ),
            CommandStep(
                name="upgrade_bootstrap",
                argv=[
                    *pip,
                    "install",
                    "--disable-pip-version-check",
                    "--upgrade",
                    *plan["pip_bootstrap"],
                ],
                cwd=REPO_ROOT,
                env=env,
            ),
        ]
    )

    for requirement in profile_plan["requirements"]:
        steps.append(
            CommandStep(
                name=f"install_requirements:{requirement}",
                argv=[
                    *pip,
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    requirement,
                ],
                cwd=REPO_ROOT,
                env=env,
            )
        )

    steps.extend(
        [
            CommandStep(
                name="install_vendored_vllm_editable",
                argv=[
                    *pip,
                    "install",
                    "--disable-pip-version-check",
                    "-e",
                    plan["editable_path"],
                    "--no-build-isolation",
                    "--no-deps",
                ],
                cwd=REPO_ROOT,
                env=env,
            ),
            CommandStep(
                name="verify_runtime_import",
                argv=[
                    str(venv_python),
                    "-c",
                    (
                        "import importlib.metadata, torch, vllm; "
                        "print('vllm', importlib.metadata.version('vllm')); "
                        "print('torch', torch.__version__, getattr(torch.version, 'cuda', None), "
                        "torch.cuda.is_available())"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
            ),
        ]
    )
    return steps


def resolved_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_build_plan()
    profile_name, accelerator_arches = detect_profile(args.profile)
    profile_plan = plan["profiles"][profile_name]
    build_profile = select_build_profile(
        profile_plan, args.build_profile, accelerator_arches
    )
    resolution = resolve_build_profile(build_profile)
    env = resolve_env(profile_plan, build_profile, accelerator_arches)
    steps = planned_steps(
        plan,
        profile_plan,
        build_profile,
        accelerator_arches,
        args.python,
        args.recreate_venv,
    )
    preflight = build_preflight(profile_name, profile_plan, args.python)
    return {
        "schema_version": 1,
        "repo_root": str(REPO_ROOT),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": args.python,
        },
        "runtime_profile": profile_name,
        "target_device": profile_plan["target_device"],
        "accelerator_arches": accelerator_arches,
        "build_profile": build_profile,
        "build_profile_resolution": {
            "profile_family": resolution.profile_family,
            "developer_friendly": resolution.developer_friendly,
            "enabled_components": resolution.enabled_components,
            "disabled_components": resolution.disabled_components,
            "enabled_native_families": resolution.enabled_native_families,
            "disabled_native_families": resolution.disabled_native_families,
            "cmake_defines": resolution.cmake_defines,
        },
        "system_packages": profile_plan["system_packages"],
        "requirements": profile_plan["requirements"],
        "requirement_digests": requirement_digests(profile_plan["requirements"]),
        "environment": env,
        "preflight": preflight,
        "recreate_venv": args.recreate_venv,
        "preflight_only": args.preflight_only,
        "steps": [step.as_json() for step in steps],
        "dry_run": args.dry_run,
        "supported_build_profiles": supported_build_profile_csv(),
    }


def run_step(step: CommandStep) -> None:
    env = step_env(step.env)
    subprocess.run(step.argv, cwd=step.cwd, env=env, check=True)


def emit_summary(plan: dict[str, Any]) -> None:
    print(
        "runtime_install "
        f"profile={plan['runtime_profile']} "
        f"target={plan['target_device']} "
        f"arches={','.join(plan['accelerator_arches']) or 'unknown'} "
        f"build_profile={plan['build_profile']} "
        f"requirements={','.join(plan['requirements'])} "
        f"steps={len(plan['steps'])} "
        f"preflight_ok={str(plan['preflight']['ok']).lower()} "
        f"dry_run={str(plan['dry_run']).lower()}"
    )
    print("system packages: " + ", ".join(plan["system_packages"]))
    if plan["preflight"]["issues"]:
        print("preflight issues: " + ", ".join(plan["preflight"]["issues"]))
    for step in plan["steps"]:
        print(f"step {step['name']}: {' '.join(step['argv'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and install the deterministic sock accelerator runtime."
    )
    parser.add_argument("--profile", default="auto", help="auto, cuda, or rocm")
    parser.add_argument(
        "--build-profile",
        default="auto",
        help=f"auto or one of: {supported_build_profile_csv()}",
    )
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Remove the vendored vLLM virtualenv before installing.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Resolve and validate the install plan without running install steps.",
    )
    parser.add_argument("--format", choices=("summary", "json"), default="summary")
    args = parser.parse_args()

    plan = resolved_plan(args)
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        emit_summary(plan)

    if args.dry_run:
        return

    if args.preflight_only:
        require_preflight_ok(plan["preflight"])
        return

    require_preflight_ok(plan["preflight"])

    for raw_step in plan["steps"]:
        run_step(
            CommandStep(
                name=raw_step["name"],
                argv=raw_step["argv"],
                cwd=Path(raw_step["cwd"]),
                env=raw_step["env"],
            )
        )


if __name__ == "__main__":
    main()
