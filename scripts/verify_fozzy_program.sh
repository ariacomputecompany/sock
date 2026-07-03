#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MANIFEST="$ROOT_DIR/fozzy/verification_program.json"
TRACE_DIR="$ROOT_DIR/.fozzy-traces"

mkdir -p "$TRACE_DIR"

python3 - "$MANIFEST" "$ROOT_DIR" "$TRACE_DIR" <<'PY'
import json
import pathlib
import subprocess
import sys

manifest = json.load(open(sys.argv[1]))
root = pathlib.Path(sys.argv[2])
trace_dir = pathlib.Path(sys.argv[3])

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)

for scenario in manifest["deterministic_scenarios"]:
    run(["fozzy", "validate", scenario, "--json"])

for scenario in manifest["contradiction_scenarios"]:
    run(["fozzy", "doctor", "--deep", "--scenario", scenario, "--runs", "5", "--seed", "424242", "--json"])

run(["fozzy", "test", "--det", "--strict-verify", *manifest["deterministic_scenarios"], "--json"])

for index, target in enumerate(manifest["host_trace_targets"], start=1):
    trace_path = root / target["trace_path"]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    seed = str(52000 + index)
    run([
        "fozzy",
        "run",
        target["scenario"],
        "--det",
        "--record",
        str(trace_path),
        "--seed",
        seed,
        "--proc-backend",
        "host",
        "--fs-backend",
        "host",
        "--http-backend",
        "host",
        "--json",
    ])
    run(["fozzy", "trace", "verify", str(trace_path), "--strict", "--json"])
    run(["fozzy", "replay", str(trace_path), "--json"])
    run(["fozzy", "ci", str(trace_path), "--json"])
PY
