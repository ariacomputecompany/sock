from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rocm_wsl_smoke_matrix import extract_log_metrics
from scripts.sock_runtime_env import subprocess_env, tool_subprocess_env


DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generated sock vLLM entrypoint wrappers across one or more models."
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--intent", default="prefill-path")
    parser.add_argument("--scope-name", default="prefill_attention")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--heartbeat-s", type=int, default=30)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("tmp/sock-entrypoint-smoke"),
        help="Directory for generated bundles and logs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/sock-entrypoint-smoke.jsonl"),
        help="JSONL results path.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def run_checked(cmd: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=tool_subprocess_env(),
    )
    if log_path is not None:
        log_path.write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(cmd)}\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    return completed


def load_entrypoint(bundle_dir: Path, scope_name: str) -> dict[str, Any]:
    document = json.loads((bundle_dir / "vllm_entrypoints.json").read_text())
    for entrypoint in document.get("entrypoints", []):
        if entrypoint.get("scope_name") == scope_name:
            return entrypoint
    raise RuntimeError(f"entrypoint scope not found: {scope_name}")


def run_wrapper(
    args: argparse.Namespace,
    model: str,
    bundle_dir: Path,
    wrapper_path: Path,
) -> dict[str, Any]:
    log_path = bundle_dir / f"{args.scope_name}.log"
    env = subprocess_env(rocm_wsl=True)
    env["SOCK_VLLM_MODEL"] = model
    env["SOCK_VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    env["SOCK_VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(wrapper_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        next_heartbeat = start + args.heartbeat_s
        while True:
            returncode = process.poll()
            now = time.perf_counter()
            if returncode is not None:
                break
            elapsed_s = now - start
            if elapsed_s >= args.timeout_s:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log_handle.flush()
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                return {
                    "ok": False,
                    "returncode": process.returncode,
                    "elapsed_s": round(elapsed_s, 4),
                    "error": f"timed out after {args.timeout_s}s",
                    "log_path": str(log_path),
                    "log_tail": log_text[-4000:],
                }
            if now >= next_heartbeat:
                print(
                    f"entrypoint_model_heartbeat model={model} elapsed_s={elapsed_s:.1f} log={log_path}",
                    flush=True,
                )
                next_heartbeat = now + args.heartbeat_s
            time.sleep(1)

    elapsed_s = time.perf_counter() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "ok": returncode == 0,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 4),
        "log_path": str(log_path),
    }
    log_metrics = extract_log_metrics(log_text)
    if log_metrics:
        result["log_metrics"] = log_metrics
    if returncode != 0:
        result["log_tail"] = log_text[-4000:]
    return result


def run_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    model_slug = slug(model)
    bundle_dir = args.work_dir / model_slug
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        subprocess.run(["rm", "-rf", str(bundle_dir)], check=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    prepare_log_path = bundle_dir / "prepare.log"
    run_checked(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "sock",
            "--",
            "prepare",
            args.intent,
            "--out",
            str(bundle_dir),
            "--format",
            "json",
        ],
        log_path=prepare_log_path,
    )

    entrypoint = load_entrypoint(bundle_dir, args.scope_name)
    wrapper_path = bundle_dir / entrypoint["wrapper_path"]
    dry_run = run_checked([str(wrapper_path), "--dry-run"])
    dry_run_summary = json.loads(dry_run.stdout)
    wrapper_result = run_wrapper(args, model, bundle_dir, wrapper_path)

    return {
        "model": model,
        "intent": args.intent,
        "scope_name": args.scope_name,
        "bundle_dir": str(bundle_dir),
        "wrapper_path": str(wrapper_path),
        "dry_run": dry_run_summary,
        **wrapper_result,
    }


def main() -> int:
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    with args.out.open("w", encoding="utf-8") as handle:
        for model in models:
            print(f"entrypoint_model_start model={model}", flush=True)
            try:
                result = run_model(args, model)
            except Exception as exc:
                result = {
                    "model": model,
                    "intent": args.intent,
                    "scope_name": args.scope_name,
                    "ok": False,
                    "error": str(exc),
                }
            failed = failed or not result["ok"]
            print(json.dumps(result, sort_keys=True), flush=True)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if failed and not args.continue_on_error:
                break
    print(f"entrypoint_results={args.out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
