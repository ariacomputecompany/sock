# vLLM Optimization Checklist

Purpose: capture the full optimization program for the vendored `vllm` in this repo as an execution checklist, ranked by production safety and expected leverage.

This document is intentionally biased toward:

- build-time reduction first
- closure proof first
- cache determinism first
- runtime preservation first

We are not shying away from deeper runtime refactors.
We are deferring them to the end so that:

- we earn the low-risk wins first
- we lock in strong measurement and replay infrastructure first
- we only touch hot runtime seams after the build/materialization world is legible
- every risky runtime refactor lands behind heavier `fozzy` coverage than the build-safe work

North stars:

- Lower local build times.
- Lower cold-start and warm-start latency.
- Higher steady-state throughput.
- Fewer runtime specialization surprises.
- Cleaner, smaller, more deterministic compile and warmup surfaces.
- Better developer experience when iterating on the fork.

Scope:

- Vendored `vllm` source in `/Users/deepsaint/Desktop/sock/vllm`.
- SOC integration and planning surface in `/Users/deepsaint/Desktop/sock/engine`.
- tinygrad-inspired compiler/runtime simplification ideas from `/Users/deepsaint/Desktop/sock/TINYGRADREF.md`.

Important grounding signals from the current tree:

- `vllm/vllm/envs.py` exposes a very large env-driven policy surface.
- `vllm/vllm/v1/worker/gpu_model_runner.py` is a major orchestration hotspot.
- `vllm/vllm/v1/core/sched/scheduler.py` is a large Python hot path.
- `vllm/vllm/config/compilation.py` centralizes many compile/cudagraph modes but still leaves policy spread across multiple files.
- `vllm/vllm/model_executor/warmup/` contains many separate warmup entry points.
- `vllm/setup.py`, `vllm/CMakeLists.txt`, `vllm/cmake/`, and `vllm/csrc/` represent a broad native build surface.
- `engine/src/vllm_adapter.rs` already identifies meaningful compile regions, cache ownership surfaces, and residual JIT surfaces that should be promoted into the actual optimization plan.

## Ranked execution model

The checklist below remains detailed and exhaustive, but work should be executed in ranked lanes.

### Lane A: highest-confidence build-safe work

Goal:

- reduce local build time
- reduce warm-start overhead
- improve cache reuse and explainability
- avoid changing live runtime semantics

Sections in order:

1. Canonical Compile Identity
2. Compile-Affecting Input Cleanup
3. Compilation Policy Simplification
4. Named Compile Regions
13. AOT Compile Artifact Improvements
14. Build System Simplification
15. Native Extension Surface Reduction
16. External Dependency and FetchContent Optimization
17. Toolchain and Build Parallelism Tuning
18. Backend Selection Rationalization
21. Replay, Explain, and Determinism
22. Validation as Executable Spec
23. Observability and Measurement
24. Fozzy-Backed Verification Program
25. Benchmark Matrix
26. SOC Integration Work
27. Tinygrad-Inspired Compiler Discipline

Notes:

- This lane is the default place to work.
- Unless proven otherwise, we assume these items can be landed without changing steady-state inference semantics.
- If a supposed Lane A item starts mutating request scheduling, execution ordering, or hot-path data layout, it should be escalated into Lane C or D.

### Lane B: medium-risk materialization and startup refactors

Goal:

- make warmup, autotune, and graph capture explicit artifacts
- reduce startup waste without destabilizing the serving loop

Sections in order:

5. Warmup Refactor
6. FlashInfer Autotune Artifactization
7. CUDAGraph Strategy Overhaul
11. Custom Op Boundary Cleanup
12. Inductor Patch and Fallback Governance
28. Codebase Hygiene and Deletion Work
29. Deployment Profiles

Notes:

- These items are still on the path to production V1.
- They are not “runtime free” even when the intent is build/startup improvement.
- Any change that alters warmup ordering, backend dispatch ordering, or graph-capture legality needs stronger replay and contradiction evidence than Lane A.

### Lane C: high-risk runtime-adjacent structural refactors

Goal:

- simplify orchestration
- reduce Python overhead
- improve long-term maintainability and profiling fidelity

Sections in order:

8. GPU Model Runner Decomposition
20. Multimodal and Encoder Separation

Notes:

- These are valuable, but they touch some of the most fragile execution seams in `vllm`.
- No Lane C work should start until Lane A is substantially complete and Lane B has given us materially better observability and replay coverage.

### Lane D: highest-risk hot-path algorithmic/runtime work

Goal:

- improve hot-path scheduling, cache behavior, and specialty execution paths
- only after we can prove correctness and non-regression with strong evidence

Sections in order:

9. Scheduler Hot Path Optimization
10. KV Cache and Prefix Cache Optimization
19. Quantization and MoE Path Rationalization

Notes:

- These sections are explicitly deferred to the end, not removed.
- They are the most likely to create subtle correctness, fairness, latency, or tail-behavior regressions.
- They must be treated as runtime engineering, not just build optimization.

## Fozzy gate policy

`fozzy` is the primary regression and production-proof tool for this entire program, not only the risky work.

Required baseline for every section:

- run deterministic strict tests first
- record at least one real trace for the active goal
- verify the trace with strict verify, replay, and CI
- keep before/after measurements for the exact phase being changed

Minimum lane-specific gate strength:

- Lane A:
  - deterministic strict scenarios for the changed build/materialization path
  - at least one host-backed recorded trace when the change affects delivered runtime artifacts
  - closure checks for no new compile where applicable
- Lane B:
  - all Lane A gates
  - explicit scenarios for warmup/autotune/cudagraph closure
  - contradiction checks when startup claims “no new tune” or “no new capture”
- Lane C:
  - all Lane B gates
  - cold startup, warm startup, first request, and steady-state decode scenarios
  - measured before/after latency and throughput comparison
  - rollback plan prepared before landing
- Lane D:
  - all Lane C gates
  - fairness/tail-latency/non-regression scenarios
  - mixed prefill/decode, MoE-heavy, quantized, and multimodal coverage where relevant
  - no merge without explicit proof that runtime behavior is preserved or intentionally improved

## Working rule

If there is any ambiguity about whether an item is “just build-time” or “runtime-adjacent,” treat it as runtime-adjacent and demand stronger `fozzy` evidence.

## Detailed checklist

The detailed sections below are preserved as the implementation checklist.
They should be executed according to the ranked lane order above, not by raw section number alone.

## 1. Canonical Compile Identity

- [ ] Introduce a single canonical compile-plan identity above vLLM config/env handling.
- [ ] Ensure every compile artifact, warmup artifact, cudagraph artifact, and autotune artifact derives from this canonical plan identity.
- [ ] Stop treating raw CLI flags, raw env var bags, and raw runtime state as primary cache-key inputs.
- [ ] Normalize all compile-affecting inputs before vLLM materialization begins.
- [ ] Use a structural hash for the resolved plan instead of relying on dispersed hash inputs.
- [ ] Keep separate identities for:
  - raw request
  - normalized request
  - resolved compile plan
  - materialization plan
  - replay/verification plan
- [ ] Make the canonical plan renderable into a stable textual form for diffing and debugging.
- [ ] Use the canonical plan as the root key for SOC explain/replay flows.
- [ ] Align this work with the tinygrad-inspired structural interning guidance in `/Users/deepsaint/Desktop/sock/TINYGRADREF.md`.

## 2. Compile-Affecting Input Cleanup

- [ ] Audit all compile-affecting env vars in `vllm/vllm/envs.py`.
- [ ] Classify env vars into:
  - compile-affecting
  - runtime-affecting but non-compile-affecting
  - debug-only
  - host-only
  - cache-location-only
- [ ] Remove non-essential env vars from compile identity calculation.
- [ ] Explicitly document why each remaining compile-affecting env var participates in the cache key.
- [ ] Normalize equivalent values before hashing.
- [ ] Prevent accidental cache splits caused by irrelevant host/process-level differences.
- [ ] Move toward declarative compile factors rather than a broad “hash nearly everything” strategy.
- [ ] Add validation that detects unexpected new compile-affecting knobs.
- [ ] Replace coarse whole-file compile invalidation with finer-grained fingerprints where safe.
- [ ] Hash traced symbol sets, lowered graph structure, selected custom ops, and active passes instead of always hashing entire source files.
- [ ] Ensure unrelated edits in non-executed code paths do not invalidate the whole compile cache.

## 3. Compilation Policy Simplification

- [ ] Reduce policy spread across `vllm/vllm/config/compilation.py`, `vllm/vllm/config/kernel.py`, `vllm/vllm/envs.py`, and `vllm/vllm/env_override.py`.
- [ ] Centralize compile mode selection in one resolved policy object.
- [ ] Centralize cudagraph mode selection in one resolved policy object.
- [ ] Centralize backend selection in one resolved policy object.
- [ ] Centralize custom-op compile behavior in one resolved policy object.
- [ ] Centralize warmup obligations in one resolved policy object.
- [ ] Make every fallback or policy override explicit and explainable.
- [ ] Remove policy hidden in side effects and one-off warnings where possible.
- [ ] Reduce mutually interacting knobs that create hard-to-predict behavior.

## 4. Named Compile Regions

- [ ] Promote compile regions already identified in `engine/src/vllm_adapter.rs` into first-class implementation concepts.
- [ ] Keep separate named regions for:
  - repeated transformer block body
  - decode micrograph
  - prefill micrograph
  - attention/KV update boundary
  - MoE specialty path
- [ ] Ensure each region has:
  - a stable identity
  - a stable cache namespace
  - a defined portability scope
  - a defined rank/topology scope
  - explicit warmup obligations
  - explicit closure verification criteria
- [ ] Avoid flattening distinct regions into one anonymous compile/cache surface.
- [ ] Use region boundaries to reduce over-compilation and improve cache reuse.
- [ ] Add explicit artifact sharing rules for isomorphic regions across ranks and processes.
- [ ] Reuse identical subgraph artifacts across `rank_x_y` cache namespaces via content-addressed manifests, hardlinks, or equivalent indirection.
- [ ] Detect repeated transformer-body regions before backend compilation begins, not only after serialized artifacts already exist on disk.
- [ ] Introduce a region-equivalence pass that can prove two subgraphs are compile-equivalent even when they arose from different trace positions.

## 5. Warmup Refactor

- [ ] Refactor warmup from a collection of backend-specific startup hooks into a named materialization graph.
- [ ] Keep warmup obligations explicit instead of implicit inside broad startup routines.
- [ ] Separate warmup obligations by region and backend.
- [ ] Split warmup into at least:
  - compile warmup
  - tactic/autotune warmup
  - cudagraph capture warmup
  - backend metadata warmup
  - KV-update warmup
  - multimodal/encoder warmup
- [ ] Record which warmup obligations were actually executed.
- [ ] Record which shape envelopes each warmup covered.
- [ ] Detect when production traffic falls outside the warmed envelope.
- [ ] Reduce reliance on generic dummy runs that only partially cover production paths.
- [ ] Make warmup replayable and verifiable as a first-class artifact.
- [ ] Generate warmup plans from the resolved deployment profile so unused backend families are not warmed by default.
- [ ] Split correctness warmup from expensive performance warmup so local iteration can skip low-value autotune work safely.
- [ ] Promote “closure proven” versus “closure assumed” into explicit warmup outcomes instead of leaving that distinction implicit in logs and startup behavior.

## 6. FlashInfer Autotune Artifactization

- [ ] Treat FlashInfer autotune outputs as first-class artifacts rather than incidental cache files.
- [ ] Persist autotune cache identity separately from model compile identity.
- [ ] Track exact tactic-cache provenance:
  - backend
  - GPU architecture
  - topology
  - rank role
  - shape envelope
- [ ] Record leader/follower asymmetry explicitly for distributed autotune.
- [ ] Verify that follower ranks load the same tactic decisions as the leader.
- [ ] Detect tactic-cache misses or shape coverage gaps before production execution.
- [ ] Avoid hidden tactic generation during the first real request.
- [ ] Make autotune coverage independently testable.

## 7. CUDAGraph Strategy Overhaul

- [ ] Move away from broad monolithic capture toward region-scoped and demand-driven capture.
- [ ] Persist capture descriptors as artifacts.
- [ ] Use observed production traces to choose which capture shapes are worth materializing.
- [ ] Avoid capturing shapes that do not appear in the active workload envelope.
- [ ] Support partial promotion:
  - eager only
  - piecewise graph
  - full decode graph
  - mixed strategy
- [ ] Make graph capture memory cost visible in planning.
- [ ] Track graph capture validity by topology and backend legality.
- [ ] Reduce startup latency caused by eager capture of too many shapes.
- [ ] Reduce graph memory waste by eliminating low-value captures.
- [ ] Keep graph capture ownership separate from compile-cache ownership.
- [ ] Verify “no new cudagraph capture” against real traces.
- [ ] Add capture value scoring so rarely hit shapes or low-speedup shapes are not captured automatically.
- [ ] Make capture planning aware of memory residency cost versus expected replay benefit.

## 8. GPU Model Runner Decomposition

- [ ] Split `vllm/vllm/v1/worker/gpu_model_runner.py` into smaller subsystems.
- [ ] Separate responsibilities into explicit modules such as:
  - batch planning
  - input preparation
  - warmup management
  - cudagraph management
  - speculative decode execution
  - multimodal/encoder execution
  - KV transfer coordination
  - profiling and cleanup
- [ ] Reduce branching density in the main execution path.
- [ ] Reduce Python object churn in execution-critical loops.
- [ ] Reduce accidental interactions between unrelated features.
- [ ] Improve profiling granularity by isolating subsystems.
- [ ] Improve cacheability and compile stability by reducing orchestration entropy.
- [ ] Minimize shallow copies, deep copies, and ad hoc state duplication in hot paths.
- [ ] Replace “god-object” orchestration with explicit phase boundaries.
- [ ] Prefer MRV2-style persistent state plus gather/staged-write patterns over repeated Python-side tensor reconstruction where feasible.
- [ ] Push remaining CPU metadata preparation that is shape-stable onto GPU-native or staged-update paths when correctness permits.

## 9. Scheduler Hot Path Optimization

- [ ] Profile `vllm/vllm/v1/core/sched/scheduler.py` specifically as a Python hot path.
- [ ] Identify request-state transitions that can be moved to array-backed structures.
- [ ] Reduce per-step Python container churn.
- [ ] Reduce repeated dictionary lookups in hot scheduling loops.
- [ ] Reduce request-level object mutation in the hottest path.
- [ ] Evaluate compact representations for:
  - token budgets
  - request scheduling state
  - encoder scheduling state
  - speculative decode scheduling state
- [ ] Separate cold-path policy logic from hot-path scheduling logic.
- [ ] Add metrics for scheduler-only wall time and object churn.
- [ ] Verify that scheduler simplification does not regress fairness or latency behavior.
- [ ] Split the scheduler into specialized fast paths for decode-only, mixed prefill/decode, speculative decode, and multimodal-heavy steps.
- [ ] Replace O(n) priority-preemption scans with maintained heaps or equivalent indexed priority structures.
- [ ] Prefer integer-indexed request arenas over repeated dict/list/set churn in the hottest loop segments.

## 10. KV Cache and Prefix Cache Optimization

- [ ] Preserve the low-allocation free-block queue approach in `vllm/vllm/v1/core/kv_cache_utils.py`.
- [ ] Separate prefix-cache identity hashing from runtime eviction metadata.
- [ ] Reduce overhead in block-hash generation for hot request paths.
- [ ] Ensure reproducibility modes are explicit and do not penalize non-reproducible fast paths unnecessarily.
- [ ] Consider a compact binary descriptor for block metadata used in hot loops.
- [ ] Reduce Python-level work in group-id/hash packing and unpacking where feasible.
- [ ] Make block-hash strategy configurable at a plan level instead of via incidental env behavior.
- [ ] Track prefix-cache hit/miss performance separately from block-allocation performance.
- [ ] Verify that hash strategy changes do not break cache portability or event pipelines.
- [ ] Introduce a logical-to-physical block indirection layer so identical full blocks can share physical storage without breaking append-only block-table semantics.
- [ ] Track duplicate physical KV residency caused by append-only tables and treat it as an explicit optimization target.
- [ ] Precompute or cache sparse/hybrid prefix-hit descriptors so hybrid prefix-cache scans do less repeated Python work.
- [ ] Remove or reduce whole-block recomputation on full-prefix hits where only the last-token logits are required.
- [ ] Evaluate a partial-block replay or logits-only tail path so prefix-cache-heavy requests do not overpay for block alignment constraints.

## 11. Custom Op Boundary Cleanup

- [ ] Refactor `vllm/vllm/model_executor/custom_op.py` to separate:
  - registration
  - selection
  - lowering
  - execution dispatch
- [ ] Reduce compile invalidation caused by custom-op enable/disable changes.
- [ ] Make custom-op behavior more traceable under `torch.compile`.
- [ ] Prefer explicit lowerings/decompositions over opaque fallback when practical.
- [ ] Keep custom-op boundaries visible in planning and explain output.
- [ ] Distinguish backend dispatch concerns from compile-lowering concerns.
- [ ] Remove avoidable eager-native fallback paths in performance-critical regions.
- [ ] Ensure custom-op state feeds canonical compile identity correctly.
- [ ] Narrow overly opaque custom-op islands so more of the transformer body remains visible to Dynamo/Inductor partitioning and fusion.
- [ ] Prefer IR-level or decomposition-level representations for hot-path custom ops whenever the current boundary is blocking fusion quality or compile reuse.

## 12. Inductor Patch and Fallback Governance

- [ ] Audit every patch in `vllm/vllm/env_override.py`.
- [ ] Put every monkeypatch behind precise version and capability guards.
- [ ] Record active patch set in compile artifact metadata.
- [ ] Distinguish:
  - correctness patch
  - performance patch
  - workaround for upstream bug
  - fallback behavior patch
- [ ] Detect when an upstream torch upgrade makes a local patch obsolete.
- [ ] Avoid allowing patches to silently widen the compile surface.
- [ ] Surface fallback creation as explicit evidence, not silent behavior.
- [ ] Track custom-op fallback namespace coverage as a first-class signal.

## 13. AOT Compile Artifact Improvements

- [ ] Avoid opaque pickled state as the sole long-term artifact format where possible.
- [ ] Track:
  - patch profile
- [ ] Verify “no new compile” against recorded traces.
- [ ] Reduce startup-time dependence on Python pickling and `GraphPickler` roundtrips where possible.
- [ ] Reduce artifact-load overhead from Python object reconstruction during warm start.
- [ ] Evaluate a more mmap-friendly or streaming-friendly payload format for compiled artifacts.
- [ ] Keep opaque pickled Python state off the critical path for the common cache-hit case.
- [ ] Replace “load everything eagerly” behavior with demand-driven or manifest-guided artifact hydration where practical.
- [ ] Measure duplicate-load cost across ranks and processes, not just duplicate artifact bytes on disk.
- [ ] Distinguish artifact-store identity from rank-local placement so one compiled payload can back multiple rank-local manifests.
- [ ] Reduce reliance on runtime `exec` for stitching-graph execution code where a pre-emitted module, manifest-bound callable, or equivalent static representation would work.
- [ ] Keep Python source generation and Python object rehydration off the common warm-start fast path wherever possible.
- [ ] Treat the torch.compile cache as a graph-artifact store with explicit proof metadata, not just as a directory of reusable byproducts.

## 14. Build System Simplification

- [ ] Split native build targets into:
  - core required
  - optional backend packs
  - experimental packs
  - benchmark/test-only packs
- [ ] Add a minimal local developer build profile.
- [ ] Add model/backend-focused build profiles for active development.
- [ ] Ensure local edits do not require rebuilding unrelated kernel families.
- [ ] Audit `vllm/CMakeLists.txt`, `vllm/setup.py`, and `vllm/cmake/` for always-on build cost.
- [ ] Gate rarely used external projects behind explicit opt-in flags.
- [ ] Reduce architecture fanout for local dev builds.
- [ ] Reduce target fanout when only Python-side changes are being tested.
- [ ] Reduce repeated configure cost across extension targets.
- [ ] Improve default build caching behavior for local iteration.
- [ ] Add explicit feature bundles such as `core`, `flashattn`, `deepgemm`, `flashmla`, `qutlass`, and `minimal-dev`.
- [ ] Ensure the default local profile does not build mutually exclusive backend packs unless a chosen model/profile actually needs them.
- [ ] Add selected-backend-only wheel and editable-build paths so a deployment like `hopper+flashinfer` or `blackwell+fa3` does not compile the broader CUDA extension superset.

## 15. Native Extension Surface Reduction

- [ ] Audit `vllm/csrc/` and identify:
  - always-needed files
  - model-specific files
  - backend-specific files
  - legacy compatibility files
  - test/benchmark-only files
- [ ] Remove or isolate legacy code paths that are no longer performance-competitive.
- [ ] Avoid building kernels that are not reachable under the selected SOC deployment profile.
- [ ] Add a “build only what this deployment needs” flow.
- [ ] Prefer explicit capability manifests to scattered compile-time conditionals.
- [ ] Track compile time per extension target and per external dependency.

## 16. External Dependency and FetchContent Optimization

- [ ] Audit the external project set in `vllm/cmake/external_projects/`.
- [ ] Cache external downloads and source material in a stable shared location.
- [ ] Avoid redownloading and reconfiguring external projects across local builds.
- [ ] Evaluate prebuilt internal packages for the slowest native dependencies.
- [ ] Split dependency acquisition from feature enablement where possible.
- [ ] Add visibility into which external projects are actually needed for a given build profile.
- [ ] Avoid pulling every CUDA-side external project into the default build graph when the selected deployment profile only needs a subset.

## 17. Toolchain and Build Parallelism Tuning

- [ ] Make local-dev build defaults smarter for `MAX_JOBS` and `NVCC_THREADS`.
- [ ] Tune job allocation to avoid oversubscription and UI-hostile compile storms.
- [ ] Prefer `sccache` or `ccache` automatically when available.
- [ ] Persist compiler cache directories predictably for the team.
- [ ] Measure configure time, compile time, and install time independently.
- [ ] Measure per-target build cost so the heaviest offenders are obvious.

## 18. Backend Selection Rationalization

- [ ] Reduce accidental backend churn from “auto” behavior where deterministic deployment is preferred.
- [ ] Make backend selection a resolved plan output, not a dispersed runtime consequence.
- [ ] Distinguish developer-default backend policy from production-default backend policy.
- [ ] Add an explicit deployment profile for SOC’s preferred kernels/backends.
- [ ] Avoid compiling or warming mutually exclusive backends in the same deployment profile unless required.
- [ ] Keep backend fallback chains explicit and observable.
- [ ] Build a precomputed capability table for backend legality at startup.
- [ ] Freeze backend/provider decisions into the resolved plan instead of re-evaluating legality in scattered runtime paths.
- [ ] Make the capability table explainable in terms of device, dtype, block size, quant mode, and attention shape envelope.
- [ ] Keep backend selection deterministic once the plan is resolved, even if multiple lazy-import-capable providers are present.
- [ ] Separate backend discovery cost from steady-state execution cost.
- [ ] Reduce lazy-import branching in hot or near-hot paths once startup discovery is complete.
- [ ] Emit a backend capability snapshot artifact at startup so every accepted and rejected backend decision is inspectable after the fact.
- [ ] Tie native extension enablement directly to the resolved backend choice so backend discovery and native build surface stay aligned.

## 19. Quantization and MoE Path Rationalization

- [ ] Audit quantized linear and MoE backend selection in `vllm/vllm/config/kernel.py` and related layers.
- [ ] Reduce the number of active implementation families per deployment profile.
- [ ] Identify which MoE and quantized paths are compile-pathological in practice.
- [ ] Isolate specialty MoE paths as their own compile/cache region.
- [ ] Avoid exposing every quant backend in local dev unless the workflow needs them.
- [ ] Track compile and warmup cost per quant backend family.
- [ ] Remove or demote low-value fallback implementations where feasible.
- [ ] Replace `fast_moe_cold_start`-style startup hacks with declarative MoE materialization passes that do not rely on silent ordering assumptions.
- [ ] Eliminate optimization modes that can cause silent incorrectness when speculative decode or multi-model forward interactions violate MoE startup assumptions.

## 20. Multimodal and Encoder Separation

- [ ] Separate text-only and multimodal compile/capture concerns more aggressively.
- [ ] Avoid paying multimodal complexity in text-only deployments.
- [ ] Make encoder cudagraph and encoder compile artifacts independent from decoder artifacts.
- [ ] Use separate closure criteria for encoder and decoder paths.
- [ ] Avoid broad runner-level invalidation caused by encoder-only features.
- [ ] Ensure text-only deployment profiles do not import, warm, or validate multimodal-only kernels and planner surfaces unnecessarily.

## 21. Replay, Explain, and Determinism

- [ ] Make every materialization decision replayable from the canonical plan.
- [ ] Emit a stable textual explain artifact for every resolved build/materialization plan.
- [ ] Capture rewrite and normalization traces, not just final state.
- [ ] Add round-trip tests for explain/render output.
- [ ] Ensure replay artifacts contain enough information to verify:
  - compile closure
  - warmup closure
  - cudagraph closure
  - autotune closure
- [ ] Avoid hidden runtime decisions that cannot be reconstructed later.

## 22. Validation as Executable Spec

- [ ] Add a spec-validation layer for resolved vLLM materialization plans.
- [ ] Validate:
  - backend legality
  - cache artifact compatibility
  - warmup coverage claims
  - topology-scoped artifact reuse
  - fallback expectations
  - graph-capture admissibility
- [ ] Add progressive validation levels:
  - fast structural validation
  - semantic validation
  - expensive witness validation
  - optional contradiction checks
- [ ] Treat unresolved hazard claims as plan failures, not just log messages.
- [ ] Add guarantee-plane validation for compile closure, autotune closure, cudagraph closure, and guard-policy assumptions separately.
- [ ] Require explicit contradiction evidence whenever runtime behavior depends on dropped guards or other “unsafe but faster” compile assumptions.

## 23. Observability and Measurement

- [ ] Add stable metrics for:
  - cold startup time
  - warm startup time
  - compile time
  - warmup time
  - cudagraph capture time
  - autotune time
  - scheduler wall time
  - GPU runner preprocess time
  - peak graph memory
  - cache hit/miss reasons
- [ ] Keep startup-time measurement separate from steady-state throughput measurement.
- [ ] Add artifact-level telemetry:
  - compile cache hit
  - graph cache hit
  - autotune cache hit
  - warmup coverage hit/miss
- [ ] Add a simple closure dashboard: “did anything new compile/tune/capture at runtime?”
- [ ] Add explicit measurements for:
  - artifact deserialization time
  - backend discovery time
  - startup import time
  - capability-table construction time
  - warmup overshoot versus actual runtime usage
- [ ] Add explicit measurement for duplicate subgraph compilation and duplicate subgraph artifact loading across ranks.
- [ ] Add explicit measurement for physical duplicated KV blocks caused by append-only prefix-cache behavior.

## 24. Fozzy-Backed Verification Program

- [ ] Use `fozzy` as the primary system/regression verification tool for this optimization work.
- [ ] Always run deterministic strict tests first for relevant scenarios.
- [ ] Record at least one real trace for the active optimization goal.
- [ ] Verify recorded traces with strict replay and CI checks.
- [ ] Build scenarios for:
  - cold startup
  - warm startup
  - first request after startup
  - steady-state decode
  - mixed prefill/decode
  - MoE-heavy path
  - quantized path
  - multimodal path where relevant
- [ ] Add closure checks for “no new compile,” “no new autotune,” and “no new graph capture” during replay.
- [ ] Include host-backed checks where runtime delivery is the goal.

## 25. Benchmark Matrix

- [ ] Define a benchmark matrix before major refactors land.
- [ ] Measure at least:
  - build configure time
  - build compile time
  - editable install time
  - cold startup
  - warm startup
  - first-token latency
  - steady-state tokens/sec
  - graph capture memory
  - compile cache hit rate
  - autotune cache hit rate
- [ ] Compare each phase against baseline before and after changes.
- [ ] Keep separate baselines for:
  - local developer profile
  - production deployment profile

## 26. SOC Integration Work

- [ ] Align SOC planner concepts with actual vLLM implementation concepts.
- [ ] Promote the adapter’s compile regions into concrete implementation and verification steps.
- [ ] Promote cache ownership surfaces into concrete artifact namespaces.
- [ ] Promote residual JIT surfaces into concrete runtime closure checks.
- [ ] Use SOC planning to choose which artifacts to build eagerly versus lazily.
- [ ] Ensure SOC explain output maps directly to vLLM artifact and warmup behavior.

## 27. Tinygrad-Inspired Compiler Discipline

- [ ] Adopt canonical object identity where possible in SOC’s resolved build/materialization plan.
- [ ] Add structural interning for repeated compile-plan substructures.
- [ ] Introduce named rewrite passes instead of ad hoc normalization helpers.
- [ ] Preserve rewrite traces for explain/replay.
- [ ] Tighten uncertainty monotonically across phases rather than reintroducing ambiguity later.
- [ ] Treat hazard repair as compiler work, not operational cleanup.
- [ ] Keep dependency compression explicit and verifiable.
- [ ] Prefer deleting complexity over layering more toggles onto existing complexity.
- [ ] Refactor vLLM compile/lowering policy into a clearly staged pipeline where each pass reduces uncertainty and feeds a stable next phase.
- [ ] Ensure the pass pipeline is replayable and hashable as a sequence, not just as a bag of enabled booleans.
- [ ] Preserve enough rewrite trace information to answer “why did this graph or backend choice happen?” without re-running the whole stack.
- [ ] Keep compile, link/assemble, warmup/materialize, and verify as explicit distinct phases with separate artifacts and hazard models.

## 28. Codebase Hygiene and Deletion Work

- [ ] Identify legacy code paths that remain only for broad compatibility but are not valuable for SOC’s deployment goals.
- [ ] Remove dead or low-value fallback branches when safe.
- [ ] Reduce special-case branching in startup/warmup.
- [ ] Reduce warning-only behavior that should instead be explicit plan state.
- [ ] Remove redundant warmup routines when one canonical path can cover them.
- [ ] Remove duplicate artifact identity logic spread across subsystems.
- [ ] Shrink or split giant orchestration and dispatch files that act as change magnets:
  - `vllm/vllm/v1/worker/gpu_model_runner.py`
  - `vllm/vllm/v1/core/sched/scheduler.py`
  - `vllm/vllm/_custom_ops.py`
  - `vllm/vllm/_aiter_ops.py`
- [ ] Reduce import-time work caused by giant modules that mix discovery, policy, fallback, and execution concerns.

## 29. Deployment Profiles

- [ ] Define explicit profiles such as:
  - local dev
  - CI correctness
  - CI performance
  - production inference
  - production MoE
  - production multimodal
- [ ] Bind each profile to:
  - allowed backends
  - allowed kernel families
  - compile policy
  - graph policy
  - warmup policy
  - artifact policy
  - build target set
- [ ] Stop using one giant “all capabilities all the time” shape as the default.
- [ ] Add a `minimal-dev` profile optimized for fastest rebuilds and smallest warmup surface during local kernel/compiler iteration.

## 30. Rollout Order

- [ ] Start with canonical compile identity and artifact naming.
- [ ] Then refactor warmup into explicit obligations and artifacts.
- [ ] Then restructure cudagraph capture around actual hot descriptors.
- [ ] Then decompose `gpu_model_runner.py`.
- [ ] Then optimize scheduler and KV hot paths.
- [ ] Then shrink native build surface and add build profiles.
- [ ] Then remove low-value legacy branches.

## 31. Native Extension Granularity

- [ ] Split `_C_stable_libtorch` and `_moe_C_stable_libtorch` into smaller extension domains so unrelated kernel edits do not trigger broad relink and rebuild work.
- [ ] Separate extension payloads by at least:
  - core utility kernels
  - attention kernels
  - quantization kernels
  - MoE kernels
  - experimental or rapidly changing kernels
- [ ] Reduce “one giant shared object” rebuild behavior for local iteration on a single kernel family.
- [ ] Track relink scope as its own DX metric, not just raw compile time.
- [ ] Prefer extension boundaries that align with actual SOC deployment profiles and ownership surfaces.
- [ ] Ensure per-extension manifests make it obvious which models and backends can reach each binary.

## 32. Compile Cache Metadata and Serialization

- [ ] Replace Python-text compile cache metadata with a more structured format than `ast.literal_eval` over generated Python data.
- [ ] Prefer a cache metadata format that is:
  - faster to load
  - safer under corruption
  - friendlier to concurrent readers
  - easier to inspect programmatically
- [ ] Separate metadata lookup from payload hydration so warm-starts do not pay unnecessary Python reconstruction cost.
- [ ] Track compile-cache metadata parse time separately from payload load time.
- [ ] Keep cache metadata versioned explicitly so format upgrades are controlled.
- [ ] Support partial cache reads so a rank can load only the subgraph metadata it actually needs.
- [ ] Make cache-metadata failure modes explicit:
  - parse failure
  - schema mismatch
  - missing payload
  - stale compatibility factors
- [ ] Ensure cache metadata can represent one shared artifact backing multiple rank-local placements.

## 33. Compile Key and Torch Integration Hardening

- [ ] Remove dependence on monkey-patching torch internals to discover or intercept compile cache keys.
- [ ] Introduce a stable vLLM-owned graph fingerprint for subgraphs and compile regions.
- [ ] Make compile deduplication work even if upstream torch internals rename or move cache-key functions.
- [ ] Keep cache-key derivation inspectable and testable without invoking backend compilation.
- [ ] Distinguish clearly between:
  - vLLM region identity
  - FX graph structural identity
  - backend compiler cache key
  - rank-local artifact placement key
- [ ] Add tests proving that isomorphic graphs with different transient names reuse the same vLLM-level artifact identity.
- [ ] Reduce hidden reliance on upstream private APIs in the common compile path.

## 34. vLLM IR Expansion Program

- [ ] Treat vLLM IR as a primary optimization surface, not only as an incremental migration aid.
- [ ] Expand vLLM IR coverage around the ops and boundaries that most affect compile closure and kernel choice.
- [ ] Prefer late kernel/provider selection through IR when it reduces pattern explosion and backend branching.
- [ ] Use IR-level identities to support per-op artifactization and per-op autotune persistence.
- [ ] Add explicit IR-level benchmarking so provider comparisons happen at the op family level, not only at end-to-end model level.
- [ ] Make IR lowering traces available in explain output so kernel/provider choice can be audited after startup.
- [ ] Use IR to reduce duplicate fusion logic across eager, Dynamo-only, and fully compiled paths.
- [ ] Add a future-facing slot for persistent per-op autotune keyed by:
  - op identity
  - dtype
  - shape bucket
  - backend/provider
  - GPU architecture
- [ ] Prefer moving backend-specific pattern complexity behind IR implementations instead of growing Python-side orchestration branches.

## 35. Optimization Level and Startup Policy

- [ ] Turn `-O3` into a meaningfully different mode instead of leaving it operationally equivalent to `-O2`.
- [ ] Define `-O3` around explicit extra work such as:
  - broader shape materialization
  - more aggressive autotune
  - profile-guided capture promotion
  - persistent per-op tactic optimization
- [ ] Ensure each optimization level maps to a concrete startup budget and artifact budget.
- [ ] Make optimization levels choose from named plan templates rather than loosely toggling many booleans.
- [ ] Emit an explain artifact that says exactly why a given `-O` level caused specific compile, warmup, autotune, and graph actions.
- [ ] Prevent “surprise expensive startup work” from being enabled implicitly by a level whose documentation suggests otherwise.
- [ ] Tie optimization levels to explicit compile-envelope and cudagraph-envelope budgets so startup cost is predictable before materialization begins.
- [ ] Make `-O3` the home for genuinely broader region materialization and profile-guided capture promotion rather than only additional boolean enablement.

## 36. Process Topology and Python Overhead

- [ ] Treat process topology cost as part of the optimization surface, not just kernel throughput.
- [ ] Measure Python and IPC overhead separately across:
  - API server
  - engine core
  - GPU worker
  - DP coordinator
- [ ] Add explicit profiling for ZMQ, shared-memory message queues, future resolution, and worker-monitor overhead.
- [ ] Reduce repeated Python-side routing and bookkeeping once backend, rank, and capability decisions are already known.
- [ ] Keep rank/topology metadata in compact precomputed tables so the engine hot path does less dynamic recomputation.
- [ ] Distinguish process-startup cost from model-startup cost from compile-startup cost.
- [ ] Make it easy to answer:
  - how much time was spent in process bring-up
  - how much time was spent in IPC setup
  - how much time was spent in model materialization
  - how much time was spent in compile and warmup
- [ ] Prefer topology-aware fast paths for common single-node and fixed-profile SOC deployments instead of always paying for the most general orchestration path.
- [ ] Then harden replay/verification for runtime closure.

## 31. Immediate First Batch

- [ ] Inventory the compile-affecting env/config surface and freeze a first canonical plan schema.
- [ ] Implement explicit artifact namespaces for:
  - compile cache
  - cudagraph cache
  - autotune cache
  - warmup coverage proof
- [ ] Split warmup into named obligations matching the adapter’s region model.
- [ ] Add closure verification that detects runtime compile/tune/capture after startup.
- [ ] Create a minimal local developer build profile for the vendored vLLM fork.
- [ ] Add a benchmark harness for build time, startup time, and first-request latency.
- [ ] Add `fozzy` scenarios and recorded traces for the active deployment path.
- [ ] Add a first backend capability snapshot artifact so a run can explain exactly why each backend/provider was or was not chosen.
- [ ] Add instrumentation to prove how much startup time is spent in:
  - import/module initialization
  - backend discovery
  - compile artifact loading
  - warmup execution
  - graph capture

## 32. Platform Activation and Import-Time Cleanup

- [ ] Separate platform detection from platform activation.
- [ ] Make platform discovery pure and side-effect free before any kernel module import occurs.
- [ ] Replace import-time kernel activation with an explicit activation phase owned by the resolved plan.
- [ ] Reduce startup work caused by `current_platform.import_kernels()` side effects during broad module import.
- [ ] Make platform plugin resolution deterministic once the deployment profile is selected.
- [ ] Cache backend and platform capability discovery results for the lifetime of the process.
- [ ] Measure startup time spent in:
  - platform detection
  - extension import
  - provider capability checks
  - backend activation
- [ ] Ensure text-only or CPU-only flows do not pay CUDA/ROCm/XPU activation costs unnecessarily.

## 33. Import Surface and Module Boundary Reduction

- [ ] Audit import-time work in:
  - `vllm/vllm/_custom_ops.py`
  - `vllm/vllm/platforms/`
  - `vllm/vllm/config/`
  - `vllm/vllm/compilation/`
- [ ] Split `_custom_ops.py` by backend or capability family so unused operator groups are not imported eagerly.
- [ ] Split registration-only logic from execution wrappers in `_custom_ops.py`.
- [ ] Delay fake-op registration for optional backends until the backend is actually part of the resolved plan where practical.
- [ ] Reduce import-time coupling between custom-op registration, platform discovery, and extension loading.
- [ ] Track import wall time per major module so startup regressions are visible immediately.

## 34. Runtime Executor Specialization

- [ ] Replace broad feature-flag branching in the GPU runner with pre-specialized executor objects.
- [ ] Pre-resolve execution modes for combinations such as:
  - eager text-only
  - piecewise cudagraph decode
  - full cudagraph decode
  - speculative decode
  - multimodal encoder plus decoder
  - ubatched execution
- [ ] Keep the hot execute path branch-thin once the mode is resolved.
- [ ] Separate executor construction cost from steady-state execute cost.
- [ ] Avoid re-evaluating the same legality and routing decisions every step when the active mode has not changed.
- [ ] Introduce explicit executor modules such as:
  - input-prep executor
  - attention-metadata executor
  - cudagraph executor
  - speculative-decode executor
  - multimodal executor
  - sampler executor
- [ ] Make executor choice visible in explain/replay artifacts.

## 35. Scheduler Strategy Object Refactor

- [ ] Split scheduler policy from scheduler state mutation.
- [ ] Introduce strategy objects or passes for:
  - running-request scheduling
  - waiting-request admission
  - preemption policy
  - encoder-cache admission
  - KV-connector coordination
  - stats emission
- [ ] Keep the hottest scheduling loop focused on compact state and token-budget arithmetic.
- [ ] Precompute policy-specific helper tables instead of re-deriving them every step.
- [ ] Evaluate structure-of-arrays or other compact state layouts for the hottest scheduler-owned request fields.
- [ ] Reduce Python branching in the hot loop for features that are disabled in the active deployment profile.
- [ ] Preserve fairness and latency behavior with dedicated regression checks while simplifying the scheduler.

## 36. Model Loading Manifest and Staging

- [ ] Introduce a resolved weight-manifest artifact for model loading.
- [ ] Resolve weight format, shard list, index files, and local-versus-remote origin once per plan instead of re-deriving them during load.
- [ ] Avoid repeated globbing, repo listing, and safetensors index discovery during normal startup.
- [ ] Stage model loading into explicit phases:
  - manifest resolution
  - artifact acquisition
  - tensor streaming
  - weight transformation or quant processing
  - post-load finalization
  - bind-to-runtime
- [ ] Keep model-load manifest identity separate from compile identity while still linking them through the resolved plan.
- [ ] Measure and report model-load time separately from post-load transformation time.
- [ ] Support cached manifest reuse across warm starts when the weight set is unchanged.

## 37. Compile Cache Hashing and Code Snapshot Cleanup

- [ ] Stop rereading full traced file contents in multiple places during compile-cache key construction.
- [ ] Capture one canonical code snapshot manifest per resolved compile plan and reuse it across cache-key producers.
- [ ] Distinguish:
  - config hash
  - env/policy hash
  - code snapshot hash
  - compiler/toolchain hash
  - artifact payload hash
- [ ] Remove duplicated code-hash construction logic across `compilation/backends.py` and `compilation/caching.py`.
- [ ] Ensure dynamically generated code fragments have stable identity even when they cannot be mapped to a normal source file.
- [ ] Make cache misses attributable to a specific changed factor rather than a generic “hash changed” result.
- [ ] Measure code-hash construction time and traced-file scan time explicitly.
- [ ] Separate “source changed” invalidation from “pass pipeline changed” invalidation so unrelated source edits do not wipe otherwise reusable compiled subgraphs.

## 38. Native Build Profile and Feature-Gating Hardening

- [ ] Add explicit build profiles such as:
  - `dev-min`
  - `runtime-cuda-core`
  - `runtime-rocm-core`
  - `full-release`
  - `benchmark-kernels`
- [ ] Ensure optional external projects are excluded from configure time unless selected by profile or resolved capability needs.
- [ ] Move external dependency selection earlier so CMake only sees the features required by the resolved build profile.
- [ ] Default local builds to current-machine architecture targeting rather than broad release fanout.
- [ ] Keep release-matrix builds as an explicit mode, not the implicit default for local iteration.
- [ ] Record which extension targets and external projects were enabled for a build as part of the build artifact metadata.
- [ ] Add a “Python-only iteration” path that skips unrelated native rebuild work whenever possible.

## 39. Piecewise Compile and Range Materialization

- [ ] Audit piecewise compilation to ensure it does not eagerly compile or eagerly load more shape ranges than the active workload envelope actually needs.
- [ ] Prefer demand-driven range compilation and demand-driven range hydration when full eager materialization is not justified by startup goals.
- [ ] Record compile-range coverage at the subgraph level so unused ranges can be pruned from future startup plans.
- [ ] Detect when many ranges are structurally identical and can share one compiled payload behind multiple range descriptors.
- [ ] Measure cold-start cost and warm-start cost caused specifically by piecewise range fanout.
- [ ] Treat `use_inductor_graph_partition` as a strategic unification path and measure where it can replace current piecewise-versus-whole-graph tradeoffs cleanly.
- [ ] Compare post-Dynamo splitting against Inductor-native partitioning on compile time, artifact reuse, cudagraph compatibility, and pass effectiveness before cementing long-term policy.

## 40. Success Criteria

- [ ] Cold startup is materially lower.
- [ ] Warm startup is materially lower.
- [ ] Local build/iteration time is materially lower.
- [ ] First-request latency no longer hides large compile/tune/capture work.
- [ ] Runtime no longer surprises us with new specialization for covered scenarios.
- [ ] The active deployment profile uses a smaller, clearer backend/kernel surface.
- [ ] Compile/warmup/capture behavior is explainable from a canonical plan.
- [ ] Engineers can iterate on the fork without paying the full global complexity tax each time.
- [ ] Warm starts spend much less time deserializing Python-heavy compile artifacts.
- [ ] Production-covered workloads do not trigger surprise backend discovery, warmup, or graph-shape expansion after startup.

## 41. Frozen Startup Policy

- [ ] Freeze env-derived behavior into a single startup policy object before model materialization begins.
- [ ] Stop allowing modules to consult `envs.py` opportunistically after the resolved plan has been created.
- [ ] Distinguish “startup-only policy” from “live runtime controls” explicitly.
- [ ] Ensure the frozen startup policy is included in explain/replay artifacts.
- [ ] Detect when a late env read would create behavior that is not represented in the canonical plan.

## 42. Trace-Driven Warmup and Capture Minimization

- [ ] Base warmup and cudagraph coverage on real observed traces, not just broad heuristic candidate sets.
- [ ] Persist a compact “observed shape envelope” artifact for each deployment profile.
- [ ] Use that envelope to decide which compile shapes, capture shapes, and warmup shapes are worth materializing on the next run.
- [ ] Track warmup overshoot: shapes and kernels warmed but never used.
- [ ] Track warmup undershoot: shapes and kernels used in production but not covered during startup.
- [ ] Prefer narrow fallback ladders over capturing every plausible token-count variant up front.

## 43. MRV2-Style Persistent State and GPU-Native Prep

- [ ] Push more per-step metadata preparation toward persistent GPU-resident state with compact diff application.
- [ ] Audit which step inputs are still effectively rebuilt from Python every iteration and prioritize them for staged-write or gather-style refactors.
- [ ] Separate persistent request state from per-step gathered model inputs more aggressively.
- [ ] Reduce CPU-side tensor reordering work in consecutive-step steady-state execution.
- [ ] Prefer GPU-native derivation of positions, sequence lengths, and similar batch metadata when it lowers CPU jitter without harming determinism.
- [ ] Measure scheduler-to-runner handoff overhead independently from model execution.

## 44. Attention and KV Planning Freeze

- [ ] Precompute attention backend, KV layout, and block-size decisions into a stable planning artifact.
- [ ] Avoid repeating backend/layout legality work during later runner initialization phases.
- [ ] Make KV layout choice a visible part of the deployment profile and replay artifact.
- [ ] Ensure backend-driven KV layout mutations do not happen implicitly after plan resolution.
- [ ] Track how many layers/groups share identical attention metadata so duplicated builder work can be reduced.

## 45. Compile Pipeline as a First-Class Product Surface

- [ ] Treat the vLLM compile pipeline itself as a supported product surface with explicit artifacts, not just an internal implementation detail.
- [ ] Emit a machine-readable manifest of enabled passes, lowered regions, and custom patches for each compiled artifact set.
- [ ] Make it easy to diff two compile manifests and see which exact pass, factor, or source file changed the cache identity.
- [ ] Ensure compile-range selection and region splitting are visible in explain output and benchmark results.
- [ ] Reduce “same workload, different artifact set” surprises by making compile-surface drift measurable.

## 46. Build-Surface Hard Pruning for SOC

- [ ] Define a SOC-specific supported hardware and backend matrix and prune the vendored build accordingly.
- [ ] Disable or split out extension targets that are unreachable for that matrix by default.
- [ ] Reduce CUDA/ROCm architecture fanout for local iteration to the architectures SOC actually ships against.
- [ ] Separate “upstream-compatibility build” from “SOC fast local build” so developers do not pay upstream breadth by default.
- [ ] Track which native targets dominate local build time and whether they are actually needed for the active deployment profile.

## 47. Priority Targets for Early Refactor

- [ ] Prioritize deep cleanup in these high-leverage files first:
  - `vllm/vllm/compilation/backends.py`
  - `vllm/vllm/compilation/caching.py`
  - `vllm/vllm/compilation/compiler_interface.py`
  - `vllm/vllm/v1/worker/gpu/cudagraph_utils.py`
  - `vllm/vllm/v1/worker/gpu/input_batch.py`
  - `vllm/vllm/v1/attention/selector.py`
  - `vllm/vllm/model_executor/warmup/deep_gemm_warmup.py`
- [ ] For each target, capture:
  - why it is hot or strategically important
  - what compile/startup/throughput tax it currently imposes
  - whether the first pass should optimize, split, or delete

## 48. Net-New Source-Grounded Additions

These items came from an additional direct review of the vendored `vllm` tree
and were deduped against the existing checklist before being added here.

### 48.1 Make Inductor Graph Partition the Default Unification Path

- [ ] Revisit optimization-level defaults so `use_inductor_graph_partition` is enabled by default on supported stacks instead of remaining off across the current `O1`/`O2`/`O3` presets.
- [ ] Treat Inductor-native partitioning as the primary long-term path for combining full-graph compile visibility with piecewise/runtime cudagraph legality.
- [ ] Add a benchmark matrix comparing current default splitting behavior versus `use_inductor_graph_partition=True` for compile wall time, compile artifact count, graph reuse rate, cudagraph legality, and steady-state throughput.
- [ ] Make the chosen partition strategy visible in explain output and startup logs as a first-class policy decision.
- [ ] Prefer one canonical partition strategy per supported deployment profile rather than allowing multiple semi-overlapping default behaviors.

### 48.2 Replace General-Purpose `_dummy_run` Warmup With A Narrower Materialization Executor

- [ ] Stop using the general-purpose `_dummy_run` path as the main substrate for warmup, profiling, and cudagraph capture when a narrower descriptor executor would suffice.
- [ ] Introduce a dedicated capture/warmup executor that reuses persistent tensors, precomputed metadata, and stable request descriptors instead of rebuilding them through the full dummy-run orchestration path.
- [ ] Precompute and reuse dummy batch descriptors for known capture sizes instead of regenerating request layouts for every warmup/capture step.
- [ ] Reuse device-side and pinned-host control tensors for dummy runs rather than rebuilding small tensors and NumPy arrays each time.
- [ ] Reduce repeated attention-metadata preparation inside dummy-run-driven warmup when the shape envelope is identical across iterations.
- [ ] Split “needs full execution semantics” dummy runs from “only need kernel materialization” warmups so the latter path stays much smaller.
- [ ] Add dedicated profiling for `_dummy_run` setup overhead versus actual model execution so orchestration cost is no longer hidden inside warmup totals.

### 48.3 Unify FlashInfer Workspace And Wrapper State

- [ ] Unify process-global TRTLLM workspace allocation and builder-local FlashInfer workspace allocation under one backend-owned workspace manager.
- [ ] Eliminate duplicate workspace sizing logic between global helper paths and per-builder wrapper initialization.
- [ ] Add explicit accounting for workspace memory by backend family, capture mode, kv-cache dtype, and decode/prefill path.
- [ ] Reuse wrapper planning state across metadata builders when the backend, dtype, and shape envelope are identical.
- [ ] Make wrapper-cache ownership explicit so decode wrapper reuse does not stay hidden inside backend-specific builder instances.
- [ ] Persist wrapper-plan compatibility metadata separately from autotune cache payloads.
- [ ] Detect when multiple FlashInfer builder instances are materializing equivalent wrapper state and collapse them behind shared backend artifacts.

### 48.4 Split Startup Attribution Into Real Phases

- [ ] Replace the current combined “profile, create kv cache, warmup model” startup logging with phase-separated accounting.
- [ ] Emit separate timings for available-memory profiling, kv-cache config resolution, kv-cache tensor creation, compile, backend warmup, autotune, cudagraph capture, and encoder capture.
- [ ] Make startup phase attribution machine-readable so SOC can diff startup regressions by phase rather than by one combined duration.
- [ ] Record phase-level memory deltas alongside phase-level wall times for the main startup stages.
- [ ] Distinguish one-time startup materialization from repeated startup deserialization/load costs.

### 48.5 Break Up Monolithic Custom-Op Import And Registration Cost

- [ ] Split `vllm/vllm/_custom_ops.py` into smaller backend- or domain-scoped registration modules so import-time cost is not paid for the entire custom-op surface at once.
- [ ] Lazily register custom-op families based on selected backend and resolved deployment profile rather than unconditionally importing broad registration surfaces.
- [ ] Measure import-time and registration-time overhead for custom-op loading as an explicit startup metric.
- [ ] Separate fake-kernel registrations needed for tracing/compilation from runtime-op dispatch surfaces so developer and CPU-heavy paths do less work.
- [ ] Ensure custom-op registration scope tracks the resolved build profile so unused op families do not widen startup and compile surfaces.
- [ ] Avoid nested compile surfaces where per-op native fallbacks are compiled opportunistically without a canonical artifact plan.

### 48.6 Remove Editable-Build Copy-Back As A Default Iteration Tax

- [ ] Remove or reduce the editable-install pattern that copies generated Python and vendored third-party trees back into the source tree after builds.
- [ ] Prefer a manifest- or symlink-based editable-build strategy over repeated tree copy-back for `vllm_flash_attn`, `triton_kernels`, `deep_gemm`, and `fmha_sm100`.
- [ ] Track copy-back time and copied-byte volume as explicit local-DX metrics.
- [ ] Ensure editable installs only materialize the generated assets actually required by the selected local build profile.
- [ ] Avoid source-tree mutation as a default side effect of local extension builds whenever a non-mutating alternative is viable.
- [ ] Preserve reproducible packaging behavior while shrinking local iteration overhead for developers working on a narrow kernel subset.

### 48.7 Tighten Extension-Surface Gating To The Real SOC Deployment Profile

- [ ] Tighten native extension selection so build target enablement follows the resolved SOC backend profile more directly than broad platform-level gating.
- [ ] Add finer-grained enablement for CUDA extension families such as FA2/FA3/FA4, FlashMLA, DeepGEMM, Qutlass, and stable-libtorch custom ops.
- [ ] Prevent local builds from enabling large optional CUDA families just because the platform could support them in theory.
- [ ] Record which optional extension families were reachable from the selected deployment profile and which were merely platform-admissible but skipped.
- [ ] Separate “runtime reachable” from “build technically possible” in build manifests and explain output.

### 48.8 Remove Unsafe Cold-Start Heuristics

- [ ] Replace `fast_moe_cold_start`-style assumption-based startup shortcuts with deterministic region-aware materialization plans.
- [ ] Remove startup optimizations that can cause silent incorrectness when their hidden ordering assumptions fail.
- [ ] Require explicit proof or scoped legality checks before enabling any cold-start shortcut that depends on decoder-forward ordering assumptions.
- [ ] Emit explicit evidence whenever a cold-start optimization depends on model-structure assumptions that are not universally valid.
- [ ] Prefer artifact- and region-level MoE warmup planning over “usually safe” startup shortcuts.

### 48.9 Target The Concrete Scheduler Pathologies

- [ ] Replace repeated O(n) preemption selection scans in priority scheduling with maintained priority data structures.
- [ ] Reduce repeated container churn in the scheduler’s running/waiting path around scheduled request maps, encoder scheduling maps, speculative token maps, and skipped-waiting transfers.
- [ ] Add a scheduler microbenchmark specifically for preemption-heavy and connector-heavy scenarios where container churn is likely to dominate.
- [ ] Separate the “request cannot be scheduled” control path from the core “request fits and advances” path so the common case is less branchy.
- [ ] Add per-step counters for preemption scans, waiting-queue skips, connector deferrals, and block-allocation retries.

### 48.10 Reuse Control Tensors In The Runner More Aggressively

- [ ] Cache and reuse small index/control tensors created in hot runner paths instead of rebuilding them from Python or NumPy repeatedly.
- [ ] Prefer staged updates into persistent pinned buffers over creating fresh `torch.tensor` or `torch.from_numpy` objects for recurring control-plane operations.
- [ ] Track control-plane tensor allocation count separately from model tensor allocation count during startup and steady-state execution.
- [ ] Move shape-stable control metadata fully onto persistent device or pinned host buffers where correctness permits.
- [ ] Audit runner-side small-tensor creation in sample index handling, speculative decode metadata, dummy-run setup, logit selection, and multimodal scheduling glue.

### 48.11 Finish The V2 Adoption Program

- [ ] Make “increase V2 default coverage” a primary performance program, not a secondary cleanup effort.
- [ ] Build a concrete fallback matrix for SOC workloads showing which real deployment profiles still fall back from V2 to V1 and why.
- [ ] Rank unsupported V2 features by actual deployment frequency and startup/throughput impact.
- [ ] Prioritize eliminating V2 blockers that keep common SOC profiles on V1, especially where the blocker preserves large Python hot paths or broad warmup surfaces.
- [ ] Treat V1 retention as an explicit compatibility cost with measured startup, compile, and orchestration tax.
- [ ] Track V2 adoption rate across the benchmark matrix as a first-class success metric.

### 48.12 Remove Oversized Persistent CPU Token Surfaces

- [ ] Refactor `InputBatch` so large request-by-`max_model_len` CPU token matrices are no longer the default persistent representation for common workloads.
- [ ] Replace oversized always-live CPU token buffers with more compact representations where only scheduled-token gathers are required.
- [ ] Reduce memory footprint and mutation cost from `token_ids_cpu_tensor` and adjacent per-request CPU-side token bookkeeping.
- [ ] Eliminate redundant request-state duplication where `CachedRequestState` and persistent batch state encode overlapping information.
- [ ] Prefer MRV2-style persistent row ownership plus gathered step inputs over broad CPU source-of-truth tensors when correctness allows.
- [ ] Measure the CPU memory tax and copy tax of persistent token matrices independently from block-table and sampler metadata costs.

### 48.13 Remove Slow Incremental Detokenization From Common Serving Paths

- [ ] Audit when serving flows still fall back to `SlowIncrementalDetokenizer` and treat those cases as explicit performance exceptions.
- [ ] Ensure SOC-default tokenizer choices land on the fast incremental detokenizer path whenever possible.
- [ ] Measure detokenization wall time separately from model execution and scheduler time for streaming-heavy scenarios.
- [ ] Add regression checks for tokenizer/backend combinations that unexpectedly lose fast incremental detokenization support.
- [ ] Reduce Python string-building overhead in output streaming paths where repeated token-by-token concatenation is still the common case.
- [ ] Make detokenizer path choice visible in explain and observability artifacts.

### 48.14 Special-Case Third-Party Source Acquisition That Is Obviously Too Broad

- [ ] Replace broad full-repository `FetchContent` usage where only a narrow subtree or packaged payload is actually needed.
- [ ] Stop fetching the full Triton repository just to vendor `python/triton_kernels/triton_kernels` when a narrower acquisition path is viable.
- [ ] Prefer pinned archives, vendored snapshots, or subtree-only mirrors for Python-only dependency payloads used during build/install.
- [ ] Measure external source acquisition time and extracted-byte volume separately from compile time.
- [ ] Keep third-party acquisition manifests explicit so local iteration can reuse already-resolved payloads without refetch or broad reconfigure.

### 48.15 Finish The DeepGEMM Packaging Refactor

- [ ] Replace the current per-Python DeepGEMM extension packaging model with a design that can converge toward one stable runtime-facing artifact family where feasible.
- [ ] Evaluate a `TORCH_LIBRARY` plus shim binding path to remove the per-interpreter `_C.cpython-*` build and install tax.
- [ ] Move DeepGEMM toward AOT kernel materialization so runtime JIT header payloads and toolkit-at-runtime requirements shrink over time.
- [ ] Separate DeepGEMM Python-package vendoring cost from DeepGEMM binary-build cost in build telemetry.
- [ ] Ensure DeepGEMM enablement follows the resolved SOC backend profile rather than broad platform admissibility alone.
- [ ] Track how much of local build and startup cost is attributable specifically to DeepGEMM packaging, JIT support payloads, and multi-interpreter artifact production.

### 48.16 Make Faster Weight Loading A Resolved Plan Decision

- [ ] Promote model-loading strategy selection into the resolved plan instead of leaving it mostly as a loose loader-format and extra-config combination.
- [ ] Decide weight-loading mode explicitly per deployment profile, including:
  - lazy safetensors
  - multithread safetensors
  - fastsafetensors
  - instanttensor
  - sharded state
- [ ] Measure weight-manifest resolution, file discovery, iterator setup, and actual tensor streaming as separate subphases.
- [ ] Revisit whether multithread weight loading should remain mostly opt-in for SOC-default local and production profiles.
- [ ] Record why a chosen loader strategy was selected and why other admissible strategies were rejected.
- [ ] Treat weight-loading policy as part of cold-start optimization, not just as a model-format implementation detail.
