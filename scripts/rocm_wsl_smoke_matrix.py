from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_VLLM_ROOT = REPO_ROOT / "vllm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROCm/WSL vendored-vLLM smoke tests across one or more models."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model id to test. May be passed multiple times.",
    )
    parser.add_argument(
        "--prompt",
        default="Write one short sentence saying the ROCm smoke test passed.",
        help="Prompt to use for each model smoke.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=900,
        help="Per-model timeout in seconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/rocm-wsl-smoke-matrix.jsonl"),
        help="JSONL results path.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("tmp/rocm-wsl-smoke-logs"),
        help="Directory for per-model smoke logs.",
    )
    parser.add_argument(
        "--heartbeat-s",
        type=int,
        default=30,
        help="Seconds between in-progress status lines.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run all requested models even if one fails.",
    )
    return parser.parse_args()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def smoke_command(args: argparse.Namespace, model: str) -> list[str]:
    return [
        sys.executable,
        "scripts/rocm_wsl_qwen_smoke.py",
        "--model",
        model,
        "--prompt",
        args.prompt,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-iters",
        str(args.warmup_iters),
        "--json",
    ]


def extract_summary(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("VLLM_TARGET_DEVICE", "rocm")
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    env.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(VENDORED_VLLM_ROOT),
            *(part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part),
        ]
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{slug(model)}.log"
    start = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                smoke_command(args, model),
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
                        "model": model,
                        "ok": False,
                        "elapsed_s": round(elapsed_s, 4),
                        "error": f"timed out after {args.timeout_s}s",
                        "log_path": str(log_path),
                        "log_tail": log_text[-4000:],
                    }
                if now >= next_heartbeat:
                    print(
                        f"matrix_model_heartbeat model={model} elapsed_s={elapsed_s:.1f} log={log_path}",
                        flush=True,
                    )
                    next_heartbeat = now + args.heartbeat_s
                time.sleep(1)
    except subprocess.TimeoutExpired as exc:
        return {
            "model": model,
            "ok": False,
            "elapsed_s": args.timeout_s,
            "error": f"timed out after {args.timeout_s}s",
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }

    elapsed_s = time.perf_counter() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = extract_summary(log_text)
    result: dict[str, Any] = {
        "model": model,
        "ok": returncode == 0 and summary is not None,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 4),
        "log_path": str(log_path),
    }
    if summary is not None:
        result["summary"] = summary
    else:
        result["error"] = "smoke command did not emit a JSON summary"
    if returncode != 0:
        result["log_tail"] = log_text[-4000:]
    return result


def main() -> int:
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    with args.out.open("w", encoding="utf-8") as handle:
        for model in models:
            print(f"matrix_model_start model={model}", flush=True)
            result = run_model(args, model)
            failed = failed or not result["ok"]
            print(json.dumps(result, sort_keys=True), flush=True)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if failed and not args.continue_on_error:
                break

    print(f"matrix_results={args.out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
