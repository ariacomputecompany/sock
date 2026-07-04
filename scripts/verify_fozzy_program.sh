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

run(["fozzy", "doctor", "project", ".", "--strict", "--json"])

all_scenarios = []
seen = set()
for key in (
    "deterministic_scenarios",
    "contradiction_scenarios",
    "fuzz_targets",
    "explore_targets",
):
    for scenario in manifest.get(key, []):
        if scenario not in seen:
            seen.add(scenario)
            all_scenarios.append(scenario)

for target in manifest.get("host_trace_targets", []):
    scenario = target["scenario"]
    if scenario not in seen:
        seen.add(scenario)
        all_scenarios.append(scenario)

for target in manifest.get("memory_trace_targets", []):
    scenario = target["scenario"]
    if scenario not in seen:
        seen.add(scenario)
        all_scenarios.append(scenario)

for scenario in all_scenarios:
    run(["fozzy", "validate", scenario, "--json"])

for scenario in manifest["contradiction_scenarios"]:
    run([
        "fozzy",
        "doctor",
        "--deep",
        "--scenario",
        scenario,
        "--runs",
        "5",
        "--seed",
        "424242",
        "--strict",
        "--json",
    ])

run(["fozzy", "test", "--det", "--strict-verify", *manifest["deterministic_scenarios"], "--json"])

for scenario in manifest.get("fuzz_targets", []):
    run(["fozzy", "fuzz", scenario, "--json"])

for scenario in manifest.get("explore_targets", []):
    run(["fozzy", "explore", scenario, "--json"])

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
    run(["fozzy", "ci", str(trace_path), "--strict", "--json"])

for index, target in enumerate(manifest.get("memory_trace_targets", []), start=1):
    trace_path = root / target["trace_path"]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    seed = str(62000 + index)
    run([
        "fozzy",
        "run",
        target["scenario"],
        "--det",
        "--record",
        str(trace_path),
        "--seed",
        seed,
        "--json",
    ])
    run(["fozzy", "trace", "verify", str(trace_path), "--strict", "--json"])
    run(["fozzy", "replay", str(trace_path), "--json"])
    run(["fozzy", "ci", str(trace_path), "--strict", "--json"])

run(["fozzy", "map", "suites", "--root", ".", "--scenario-root", "tests", "--json"])
PY
