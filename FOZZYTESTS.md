# Fozzy Verification Program

## Purpose

This document defines the production Fozzy program for `sock`.

It is not a speculative backlog. It describes the scenarios, command families,
recorded traces, and governance checks that are now expected to hold for this
repository.

## Current Status

- `fozzy` is installed and usable in this workspace.
- The repo has direct scenario coverage for `plan`, `explain`, `build`,
  `prepare`, `measure`, `verify`, `replay`, `doctor`, `benchmark`, and the
  native `vllm` build-safe surface.
- The verification program now includes all major Fozzy families we can apply
  safely to the current product surface:
  - deterministic scenario validation
  - contradiction and determinism checks via `doctor --deep`
  - host-backed trace recording and replay verification
  - `fuzz` coverage over CLI and scope surfaces
  - `explore` coverage over multi-intent materialization paths
  - memory-oriented report and artifact-growth checks

## Verification Entry Point

Run the full program with:

```bash
scripts/verify_fozzy_program.sh
```

That script performs:

1. `fozzy doctor project . --strict --json`
2. `fozzy validate` for every scenario in the manifest
3. `fozzy doctor --deep --scenario ... --runs 5 --seed 424242 --strict --json`
   for contradiction-sensitive scenarios
4. `fozzy test --det --strict-verify ... --json` across the deterministic suite
5. `fozzy fuzz ... --json` across fuzz targets
6. `fozzy explore ... --json` across explore targets
7. host-backed `fozzy run --det --record ... --proc-backend host --fs-backend host --http-backend host --json`
   for trace targets
8. `fozzy trace verify --strict --json`, `fozzy replay --json`, and
   `fozzy ci --strict --json` for every recorded trace
9. `fozzy run --det --record ... --json` for memory-oriented trace targets
10. `fozzy map suites --root . --scenario-root tests --json` to refresh the
    suite-map snapshot

## Scenario Inventory

Current repo scenarios:

- `tests/backend.decision.fozzy.json`
- `tests/benchmark.matrix.fozzy.json`
- `tests/build.cache_invalidation.fozzy.json`
- `tests/build.decode_backend_scope.fozzy.json`
- `tests/build.early_serve_readiness.fozzy.json`
- `tests/build.flashinfer_cache_scope.fozzy.json`
- `tests/build.invalid_subset_scope.fozzy.json`
- `tests/build.named_compile_regions.fozzy.json`
- `tests/build.no_new_compile_closure.fozzy.json`
- `tests/build.prefill_scope.fozzy.json`
- `tests/build.prefill_warmup_scope.fozzy.json`
- `tests/build.reuse_decision.fozzy.json`
- `tests/build.shared_cache_root.fozzy.json`
- `tests/canonical.identity.fozzy.json`
- `tests/doctor.json.fozzy.json`
- `tests/doctor.pass.fozzy.json`
- `tests/explain.pass.fozzy.json`
- `tests/explain.replay_alignment.fozzy.json`
- `tests/explain.scoped_prefill.fozzy.json`
- `tests/explore.materialization_surface_matrix.fozzy.json`
- `tests/fuzz.cli_surface_matrix.fozzy.json`
- `tests/memory.bundle_materialization.fozzy.json`
- `tests/measure.decode_path.fozzy.json`
- `tests/measure.distributed_flashinfer_startup.fozzy.json`
- `tests/measure.prefill_path.fozzy.json`
- `tests/measure.replay_safe_closure.fozzy.json`
- `tests/optimization.levels.fozzy.json`
- `tests/plan.host.fozzy.json`
- `tests/plan.pass.fozzy.json`
- `tests/prepare.decode_path.fozzy.json`
- `tests/prepare.distributed_flashinfer_startup.fozzy.json`
- `tests/prepare.prefill_path.fozzy.json`
- `tests/prepare.replay_safe_closure.fozzy.json`
- `tests/replay.operator_gates.fozzy.json`
- `tests/replay.pass.fozzy.json`
- `tests/replay.proof.fozzy.json`
- `tests/verify.fail_closed.fozzy.json`
- `tests/verify.operator_gates.fozzy.json`
- `tests/vllm.build_safe_native.fozzy.json`
- `tests/vllm.compile_input_cleanup.fozzy.json`

## Surface Coverage

### Deterministic CLI coverage

Covered directly:

- `plan`
- `explain`
- `build`
- `prepare`
- `measure`
- `verify`
- `replay`
- `doctor`
- `benchmark`

### Native and integration coverage

Covered directly:

- `vllm` build-safe native contract generation
- compile-input cleanup and normalization
- backend-decision alignment across `explain`, `build`, and `replay`

### Fuzz coverage

Focused on:

- CLI scope and output-mode matrixes
- fail-closed invalid scope requests
- plan-time backend and cache namespace scoping

### Explore coverage

Focused on:

- multi-intent prepare and measure surfaces
- materialization identity and closure invariants across supported intents
- distributed and replay-safe closure paths

### Memory-oriented coverage

Focused on:

- bundle and measurement artifact growth signals
- materialization node and wave accounting
- benchmark report byte deltas and trace references

## Known Caveat

`fozzy map suites --root . --scenario-root tests --json` is still noisy because
it scans vendored files under `vllm/`, including `vllm/.venv-codex`.

That means raw hotspot totals are not yet a clean governance number for product
work. The suite map is still useful, but it should be interpreted in this order:

1. first inspect non-venv product hotspots
2. then inspect vendored `vllm` hotspots
3. treat `.venv-codex` entries as mapper noise until we have an exclusion
   mechanism or remove that environment from the repo tree

## Most Recent Observations

From the latest local suite-map run:

- scenario count is `40`
- `fuzz_inputs`, `explore_schedule_faults`, and `memory_graph_diff_top` now
  appear in coverage evidence for the updated scenario set
- broad `test_det`, `run_record_replay_ci`, and `host_backends_run` signals are
  also present
- the remaining suite pressure is now mostly:
  - `shrink_exercised`
  - residual `explore_schedule_faults` and `memory_graph_diff_top` demand on
    noisy vendored hotspots
- the top returned uncovered rows still include many `.venv-codex` files, so the
  raw uncovered count should not be read as product-only debt

## Production Standard

The Fozzy program is considered healthy when all of the following remain true:

- the verification script passes end to end
- deterministic scenarios stay stable under `doctor --deep`
- recorded traces continue to verify and replay strictly
- scope-narrow builds remain measurably smaller than broad builds where the
  product claims they should
- fail-closed seams continue to reject unsupported standalone closures
- new `vllm` integration work extends this program instead of bypassing it

## Next Hardening Work

The next meaningful expansions are:

- add shrink-specific failure-trace workflows once we have a stable,
  intentionally failing minimization target that is useful in CI
- reduce suite-map noise by excluding or relocating `vllm/.venv-codex`
- add deeper runtime-near `vllm` entrypoint scenarios only when they can be
  exercised without compromising determinism or requiring heavyweight ambient
  runtime services
