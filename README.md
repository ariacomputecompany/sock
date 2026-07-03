# sock

sock is a compiler and build system for inference engines.

It takes an engine such as `vLLM`, determines exactly which compiled regions, kernels, caches, warmups, and backend artifacts are required for a specific inference use-case, and builds that closure deterministically.

The point is simple:

- build exactly what you need
- reuse exactly what is admissible
- prove exactly what is ready to serve
- refuse hidden compile work and stale cache assumptions

## What sock does

sock compiles inference-engine startup the way a normal compiler handles a program:

- it parses a requested serving intent
- it resolves engine-specific compile regions and backend choices
- it computes the minimal valid artifact closure
- it materializes only the required compiled outputs and caches
- it executes the required warmup closure for the requested readiness target
- it emits a replayable build bundle with strict verification

In practice, that means:

- if you only need a prefill path, sock builds the prefill closure
- if a requested subset seam is semantically real in vendored `vLLM`, sock builds only that closure
- if a requested subset seam actually depends on broader worker startup, sock fails closed instead of pretending it is standalone
- if multiple bundle outputs share a cache root, sock reuses only admissible artifacts across them instead of treating each output directory as its own isolated cache universe
- if a distributed startup needs leader/follower artifact fanout, sock plans and executes that explicitly
- if an artifact is already valid for reuse, sock proves that and skips rebuilding it
- if you ask for early-serve readiness, sock materializes the serveable closure without pretending deferred performance warmups already happened
- if a requested scope would still leak runtime JIT, sock surfaces that as a bounded contract instead of hiding it

## Product model

sock treats inference-engine startup as a deterministic compilation problem.

The canonical pipeline is:

`RawRequest -> NormalizedRequest -> ResolvedBuildPlan -> ArtifactClosure -> VerificationReport`

From that pipeline, sock produces:

- a `ResolvedBuildPlan` describing the exact closure to build
- an `ArtifactClosure` describing the concrete compiled artifacts and caches
- a `VerificationReport` proving structural correctness, admissibility, warmup coverage, and runtime-JIT bounds
- a replay bundle that can be verified and replayed without new compile work

## Why it exists

Inference engines are usually asked to do too much in one opaque startup lifecycle:

- graph compilation
- backend selection
- Triton codegen
- FlashInfer setup
- cache shaping
- CUDA graph capture
- warmup
- topology fanout

When all of that is coupled together, operators get long cold starts, bad visibility, and very little control over what is actually being built.

sock separates those concerns into explicit build units so startup becomes:

- inspectable
- deterministic
- partially buildable
- replay-safe
- measurable

## V1 scope

sock goes deep before it goes broad.

V1 is:

- engine: `vLLM`
- hardware: NVIDIA
- platform: Linux

This is not a shallow abstraction layer over many engines.
It is a deep compiler-grade integration with one real production engine first, so the abstractions are proven against actual complexity.

## What sock is

- a compiler for inference-engine startup
- a deterministic build system for serving closures
- a planner and executor for compiled regions, kernels, caches, and warmup work
- a proof surface for artifact reuse, readiness, and no-surprise-JIT claims
- a source-aligned integration layer over vendored engine code

## What sock is not

- an inference engine
- a serving control plane
- a model training framework
- a replacement for CUDA, Triton, FlashInfer, TensorRT, or engine-native runtimes

sock compiles and materializes the engine-specific startup world.
It does not replace the runtimes that ultimately execute the model.

## How it builds less

The core design goal is to avoid broad “build everything” behavior.

sock does that by making the engine legible in terms of:

- compile-equivalent regions
- cache ownership boundaries
- backend-specific artifact scopes
- topology-sensitive materialization paths
- warmup scopes
- runtime-JIT risk surfaces

Those are then used to build only the minimal valid closure for the requested purpose.

Examples:

- `prefill_attention` can be planned and materialized separately from `decode_attention`
- `decode_attention` and `kv_cache_update` are tracked as explicit `vLLM` surfaces, but subset builds fail closed when vendored startup paths still require mixed-batch worker context
- backend autotune caches can be treated as first-class artifacts instead of hidden side effects
- CUDA graph captures can be handled as topology-scoped rank-local outputs
- leader/follower artifact fanout can be chosen explicitly instead of emerging accidentally at runtime
- invalidation evicts only stale siblings in the affected cache namespace and invalidation domain instead of blowing away unrelated artifact closures

## vLLM integration

sock vendors `vLLM` locally and integrates with it directly.

That integration preserves real `vLLM` structure rather than flattening it into fake-generic abstractions.

sock models:

- canonical compile region identity
- backend binding per region
- cache ownership surfaces
- topology-sensitive warmup paths
- backend-specific runtime-JIT triggers
- materialization and reuse boundaries derived from `vLLM` source

This is what lets sock compile parts of the engine intentionally instead of treating `vLLM` startup as one opaque side effect.

## Operator workflow

The CLI surface is:

- `cargo run --bin sock -- prepare prefill-path --out /tmp/sock-bundle`
- `cargo run --bin sock -- prepare replay-safe-closure --out /tmp/sock-bundle`
- `cargo run --bin sock -- plan`
- `cargo run --bin sock -- explain`
- `cargo run --bin sock -- build --out /tmp/sock-bundle`
- `cargo run --bin sock -- verify --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- replay --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- doctor`

The workflow is:

1. describe the serving intent
2. inspect the requested scope, expanded closure, and estimated work
3. build the required subset
4. verify the emitted bundle
5. replay the result without new compile work

## Verification model

Verification is strict and fail-closed.

sock verifies:

- artifact admissibility
- plan identity consistency
- cache and bundle integrity
- warmup proof coverage
- runtime-JIT evidence bounds
- observed runtime-JIT contradictions from the live build path
- compile-free verify/replay operator gates

Replay bundles are content-digested, identity-checked, and verification-checked.
Invalid reuse, stale artifacts, and mismatched plan state are rejected instead of being repaired implicitly.

## Repository layout

- `app/`: CLI and shared application contract surface
- `core/`: canonical schemas, identity, validation, bundle logic, verification, and rendering
- `engine/`: planner, executor integration, and `vLLM`-specific build logic
- `vllm/`: vendored `vLLM` source tree used for direct integration

## End goal

sock is the system you use when you want an inference engine to behave like a compiler target instead of a startup mystery.

It gives operators deterministic control over:

- what gets built
- what gets reused
- what gets warmed
- what is safe to serve
- what can still trigger runtime specialization

The result is faster builds, smaller closures, cleaner cold starts, and a much better production DX for inference-engine deployment.
