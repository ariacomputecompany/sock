#!/usr/bin/env python3
"""Run the canonical CUDA shim scenario matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sock_cuda_shim.diagnostics import evaluate_readiness
from sock_cuda_shim.inference import run_inference_contract
from sock_cuda_shim.kv_cache import TMHPhysicalPolicy
from sock_cuda_shim.scenarios import CANONICAL_SCENARIOS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--inference-contract",
        action="store_true",
        help="also validate the CUDA-shaped inference contract for each scenario",
    )
    args = parser.parse_args()

    rows = []
    failures = 0
    for scenario in CANONICAL_SCENARIOS:
        report = evaluate_readiness(
            devices=scenario.devices,
            env=scenario.env,
            build=scenario.build,
            kv_spec=scenario.kv_spec,
            request=scenario.request,
            attention_shape=scenario.attention,
            graph_plan=scenario.graph,
            distributed_plan=scenario.distributed,
            quantization_plan=scenario.quantization,
        )
        matched = report.ok is scenario.should_pass
        failures += 0 if matched else 1
        rows.append(
            {
                "name": scenario.name,
                "expected_ok": scenario.should_pass,
                "actual_ok": report.ok,
                "matched": matched,
                "backend": report.selected_attention_backend,
                "checks": list(report.checks),
                "failures": list(report.failures),
                "why": scenario.why,
                "inference_contract": (
                    _inference_contract_row(scenario)
                    if args.inference_contract
                    else None
                ),
            }
        )

    if args.json:
        print(json.dumps({"ok": failures == 0, "scenarios": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            status = "PASS" if row["matched"] else "FAIL"
            print(f"{status} {row['name']} backend={row['backend']} expected={row['expected_ok']}")
            for failure in row["failures"]:
                print(f"  {failure}")
    return 1 if failures else 0


def _inference_contract_row(scenario):
    report = run_inference_contract(
        scenario,
        tmh_policy=TMHPhysicalPolicy() if "blackwell" in scenario.name else None,
    )
    return {
        "ready": report.ready,
        "backend": report.selected_attention_backend,
        "kv_layout": report.kv_layout,
        "total_tokens": report.total_tokens,
        "graph_capture_required": report.graph_capture_required,
        "tmh_pressure": report.tmh_pressure,
    }


if __name__ == "__main__":
    raise SystemExit(main())
