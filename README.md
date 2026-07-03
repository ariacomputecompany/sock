# sock

sock is a compiler for inference-engine startup.

Its job is to take an engine like `vLLM`, figure out exactly which compiled regions, kernels, caches, warmups, and backend-specific artifacts are needed for a given inference use-case, and build only that closure deterministically.

In plain terms:

- if you only need a specific part of an inference engine, you should not have to build the whole engine
- if a cache or kernel binary is already admissible for reuse, `sock` should prove that and skip rebuilding it
- if a runtime path can still trigger surprise compile work, `sock` should expose that explicitly instead of hiding it behind startup scripts

The end goal is fast, deterministic, replay-safe partial builds for inference engines.

## What it is

- a build system and compiler layer for inference-engine startup
- a deterministic closure planner for compiled regions, kernels, caches, topology, and warmup work
- a way to compile or materialize only the parts of an engine needed for a specific serving intent
- a schema-governed producer of `ResolvedBuildPlan`, `ArtifactClosure`, and `VerificationReport`
- a proof system for what has been built, reused, warmed, and still remains risky

## What it is not

- an inference engine
- a scheduler or serving control plane
- a frontend for model authoring or training
- a replacement for CUDA, Triton, FlashInfer, TensorRT, or engine-native runtimes

## V1 scope

V1 is intentionally narrow and deep:

- engine: `vLLM`
- hardware: NVIDIA
- platform: Linux

The goal is to get deep closure quality for one real production engine before expanding breadth.

## Why it exists

Inference-engine startup is usually treated like one big opaque side effect:

- import the engine
- load the model
- let torch compile what it wants
- let Triton JIT what it wants
- let backend autotune happen
- let CUDA graph capture happen
- hope requests do not trigger more work later

That is bad for both build time and operator trust.

sock exists to break that apart into explicit build units so startup becomes something we can compile intentionally instead of something we merely observe after the fact.

The central idea is simple:

- discover compile-equivalent regions
- discover cache ownership boundaries
- discover warmup and runtime-JIT boundaries
- compute the minimal valid closure for a requested use-case
- build, verify, and replay that closure deterministically

## Current code shape

The canonical semantic pipeline is:

`RawRequest -> NormalizedRequest -> ResolvedBuildPlan -> ArtifactClosure -> VerificationReport`

The Rust workspace is organized as:

- `app/`: CLI entrypoint
- `core/`: canonical semantic core, identities, validation, and replay helpers
- `engine/`: engine-facing planner glue
- `vllm/`: vendored `vLLM` source tree, pinned locally for deep integration work

## Status

Today the project has the control plane for this compiler:

- a compiling Rust workspace
- a canonical `RawRequest -> NormalizedRequest -> ResolvedBuildPlan -> ArtifactClosure -> VerificationReport` contract
- a deterministic `vLLM` planning path for NVIDIA/Linux with source-anchored compile regions
- `vLLM` adapter truth for canonical region identity, cache ownership, topology-sensitive warmup surfaces, and machine-checkable residual JIT triggers
- replay bundle emission and fail-closed bundle validation
- CLI-visible explain, verify, replay, and doctor surfaces
- compile-free `verify` and `replay` operator gates rendered in the verification surface
- vendored `vLLM` source for source-aligned adapter work

What it does not have yet is the full data plane:

- `sock build` does not yet perform real scoped materialization of vendored `vLLM` artifacts
- partial-build execution is not finished yet
- measured build-time reduction from live scoped execution is not finished yet

So the current repo is already a serious compiler/planner architecture for inference-engine startup, but it is not yet the completed selective build executor.

## Operator workflow

The production CLI surface is:

- `cargo run --bin sock -- plan`
- `cargo run --bin sock -- explain`
- `cargo run --bin sock -- build --out /tmp/sock-bundle`
- `cargo run --bin sock -- verify --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- replay --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- doctor`

Right now these commands are about planning, explaining, bundling, verifying, and replaying the build contract.
The next implementation step is making `sock build` perform real scoped materialization against vendored `vLLM`.

Replay bundles are intentionally strict:

- all emitted contract files are content-digested
- plan identity must agree across the bundle
- verification reports must exactly match the loaded build plan
- invalid artifact reuse inside an otherwise well-formed bundle is rejected during verification
- mismatches fail closed instead of being repaired implicitly

`sock verify` and `sock replay` are currently contract-validation paths, not materialization paths:

- they load an emitted bundle
- they prove structural identity and verification consistency
- they render bounded runtime-JIT evidence and compile-free operator gates
- they do not perform new compile, warmup, or artifact materialization work

That restriction is deliberate.
Once the real executor lands, verify and replay should still stay compile-free.

## Verification

Engineer-facing regression coverage lives in:

- Rust unit and integration tests under the workspace crates
- Fozzy scenarios under `tests/*.fozzy.json`
- replay artifacts emitted by `sock build`, which are then verified and replayed without silent recompilation
- host-backed Fozzy trace verify/replay/ci passes recorded from the real CLI workflow

## Integration shape

The binary and library defaults now share one path:

- `app/src/lib.rs` owns the default production host snapshot, request, planning entrypoint, diagnostics, and replay-bundle construction
- `app/src/main.rs` is only the CLI shell over that shared contract
- `engine/` owns planner and `vLLM` integration
- `core/` owns the canonical schemas, validation, bundle strictness, rendering, and identity logic

## End state

The product we are building is not “a nicer wrapper around `vLLM`.”

It is:

- a compiler for inference-engine startup
- a deterministic builder for exact serving closures
- a way to compile only the parts of an inference engine you actually need
- a proof surface for reuse, warmup, and no-surprise-JIT claims

`vLLM` is the first backend because it is complex enough to force the abstractions to be real.

## Repository map

- `SPEC.md`: product spec
- `FIRST_PRINCIPLES_REPORT.md`: architectural reasoning
