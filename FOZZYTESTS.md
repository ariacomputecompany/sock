# Fozzy Verification Program

## Purpose

This document defines the full Fozzy scenario program for `sock`.

Every scenario named here exists in `tests/`, is wired into the repo-level
verification manifest, and is intended to be production-maintained rather than
treated as speculative backlog.

## Program Entry Point

Run the full verification program with:

```bash
scripts/verify_fozzy_program.sh
```

That script performs:

1. `fozzy doctor project . --strict --json`
2. `fozzy validate` for every scenario in the manifest
3. deep determinism checks for contradiction-sensitive scenarios
4. `fozzy test --det --strict-verify` across the deterministic manifest set
5. `fozzy run --det --json` across the explicit scenario-run target set
6. `fozzy fuzz` across fuzz targets
7. `fozzy explore` across explore targets
8. host-backed trace record, verify, replay, and CI checks
9. memory-oriented trace record, verify, replay, and CI checks
10. `fozzy map suites --root . --scenario-root tests --json`

## Current Scope

The current implemented scope is:

- `88` scenario files in `tests/`
- direct `sock` coverage for `plan`, `explain`, `build`, `prepare`, `measure`,
  `verify`, `replay`, `doctor`, and `benchmark`
- direct vendored `vllm` coverage for setup, docker, compile cache, serving,
  OpenAI entrypoints, structured output, disaggregated serving, Rust transport,
  and native FP8 kernel surfaces
- explicit `fuzz`, `explore`, trace, shrink, and memory-oriented scenario
  families

## Scenario Families

### Baseline CLI and bundle scenarios

- `tests/plan.pass.fozzy.json`
- `tests/plan.host.fozzy.json`
- `tests/explain.pass.fozzy.json`
- `tests/explain.scoped_prefill.fozzy.json`
- `tests/explain.replay_alignment.fozzy.json`
- `tests/build.prefill_scope.fozzy.json`
- `tests/build.prefill_warmup_scope.fozzy.json`
- `tests/build.decode_backend_scope.fozzy.json`
- `tests/build.flashinfer_cache_scope.fozzy.json`
- `tests/build.early_serve_readiness.fozzy.json`
- `tests/build.cache_invalidation.fozzy.json`
- `tests/build.reuse_decision.fozzy.json`
- `tests/build.shared_cache_root.fozzy.json`
- `tests/build.no_new_compile_closure.fozzy.json`
- `tests/build.invalid_subset_scope.fozzy.json`
- `tests/build.named_compile_regions.fozzy.json`
- `tests/prepare.prefill_path.fozzy.json`
- `tests/prepare.decode_path.fozzy.json`
- `tests/prepare.distributed_flashinfer_startup.fozzy.json`
- `tests/prepare.replay_safe_closure.fozzy.json`
- `tests/measure.prefill_path.fozzy.json`
- `tests/measure.decode_path.fozzy.json`
- `tests/measure.distributed_flashinfer_startup.fozzy.json`
- `tests/measure.replay_safe_closure.fozzy.json`
- `tests/verify.operator_gates.fozzy.json`
- `tests/verify.fail_closed.fozzy.json`
- `tests/replay.operator_gates.fozzy.json`
- `tests/replay.pass.fozzy.json`
- `tests/replay.proof.fozzy.json`
- `tests/backend.decision.fozzy.json`
- `tests/benchmark.matrix.fozzy.json`
- `tests/canonical.identity.fozzy.json`
- `tests/optimization.levels.fozzy.json`

### Explain and doctor expansion scenarios

- `tests/explain.json_shape.fozzy.json`
- `tests/explain.fail_closed_invalid_scope.fozzy.json`
- `tests/doctor.pass.fozzy.json`
- `tests/doctor.json.fozzy.json`
- `tests/doctor.host_snapshot.fozzy.json`
- `tests/doctor.degraded_host.fozzy.json`

### Fuzz scenarios

- `tests/fuzz.cli_surface_matrix.fozzy.json`
- `tests/fuzz.cli_scope_matrix.fozzy.json`
- `tests/fuzz.cli_output_modes.fozzy.json`
- `tests/fuzz.cache_namespace_inputs.fozzy.json`

### Explore scenarios

- `tests/explore.materialization_surface_matrix.fozzy.json`
- `tests/explore.materialization_wave_order.fozzy.json`
- `tests/explore.cache_reuse_vs_rebuild.fozzy.json`
- `tests/explore.readiness_frontier.fozzy.json`
- `tests/explore.vllm.serving_entrypoints.fozzy.json`
- `tests/explore.vllm.disagg_connectors.fozzy.json`
- `tests/explore.vllm.rust_transport.fozzy.json`
- `tests/explore.vllm.config_runtime.fozzy.json`

### Trace scenarios

- `tests/trace.build_prefill_scope.fozzy.json`
- `tests/trace.measure_prefill_path.fozzy.json`
- `tests/trace.verify_fail_closed.fozzy.json`
- `tests/trace.cache_reuse.fozzy.json`

These scenarios explicitly call `fozzy run`, `fozzy trace verify`, `fozzy replay`,
and `fozzy ci` against target scenarios.

### Shrink scenarios

- `tests/shrink.invalid_subset_scope.fozzy.json`
- `tests/shrink.bundle_tamper_verify.fozzy.json`
- `tests/shrink.cache_invalidation_regression.fozzy.json`
- `tests/shrink.vllm.serving_entrypoints.fozzy.json`
- `tests/shrink.vllm.disagg_connectors.fozzy.json`
- `tests/shrink.vllm.rust_transport.fozzy.json`
- `tests/shrink.vllm.config_runtime.fozzy.json`
- `tests/shrink.vllm.native_runtime.fozzy.json`

These scenarios explicitly call `fozzy shrink` on recorded traces so shrink is
part of the executable program rather than only a future idea.

### Memory-oriented scenarios

- `tests/memory.bundle_materialization.fozzy.json`
- `tests/memory.measure_report_growth.fozzy.json`
- `tests/memory.cache_root_churn.fozzy.json`

These scenarios focus on:

- artifact and report byte growth
- materialization node and wave accounting
- scoped cold versus scoped warm cache behavior

### Existing vendored `vllm` integration scenarios

- `tests/vllm.build_safe_native.fozzy.json`
- `tests/vllm.compile_input_cleanup.fozzy.json`

### Vendored `vllm` setup and build surface scenarios

- `tests/vllm.setup.profile_matrix.fozzy.json`
- `tests/vllm.setup.invalid_envs.fozzy.json`
- `tests/vllm.docker.build_args.fozzy.json`
- `tests/vllm.compile_cache_identity.fozzy.json`

### Vendored `vllm` serving and API surface scenarios

- `tests/vllm.serve.startup_smoke.fozzy.json`
- `tests/vllm.serve.invalid_config.fozzy.json`
- `tests/vllm.openai.api_basic.fozzy.json`
- `tests/vllm.realtime.connection_faults.fozzy.json`
- `tests/vllm.structured_output.fozzy.json`

### Vendored `vllm` disaggregated and supervisor scenarios

- `tests/vllm.disagg.proxy_boot.fozzy.json`
- `tests/vllm.disagg.retry_faults.fozzy.json`
- `tests/vllm.dp_supervisor.control_plane.fozzy.json`
- `tests/vllm.transport.partition_events.fozzy.json`

### Vendored `vllm` Rust frontend scenarios

- `tests/vllm.rust.client_smoke.fozzy.json`
- `tests/vllm.rust.client_retry_faults.fozzy.json`
- `tests/vllm.rust.transport_frame_fuzz.fozzy.json`
- `tests/vllm.rust.route_contracts.fozzy.json`

### Vendored `vllm` native kernel scenarios

- `tests/vllm.native.gemm_fp8_smoke.fozzy.json`
- `tests/vllm.native.gemm_fp8_shape_fuzz.fozzy.json`
- `tests/vllm.native.gemm_fp8_memory.fozzy.json`

## Manifest Wiring

`fozzy/verification_program.json` now contains:

- deterministic scenarios
- contradiction scenarios
- fuzz targets
- explore targets
- explicit scenario-run targets
- host trace targets
- memory trace targets

That means the repo does not merely list these scenarios; it executes them
through the shared verification program.

## Current Observations

From the latest local state:

- scenario count is `88`
- `fuzz_inputs`, `explore_schedule_faults`, and `memory_graph_diff_top` are now
  represented in suite-map coverage evidence
- `shrink_exercised` is now directly exercised against vendored `vllm` serving,
  disaggregated connector, Rust transport, config/runtime, and native runtime
  hotspot groups
- vendored `vllm` hotspot coverage moved from `21/663` to `659/663` covered
  required hotspots after adding the grouped explore and shrink suites
- the remaining suite-map work is a short tail, not absence of core scenario
  families
