# sock

sock is an inference artifact-closure optimizer for production startup paths.

It turns inference startup, warmup, backend selection, kernel materialization, cache reuse, and runtime-compile risk into a deterministic build product with replayable artifacts and evidence-backed guarantees.

## What it is

- a semantic orchestration layer above inference engines
- a deterministic planner for backend, kernel, cache, topology, and warmup strategy
- a schema-governed producer of `BuildPlan`, `ArtifactClosure`, and `VerificationReport`
- a system for proving what has and has not been materialized before serving

## What it is not

- an inference engine
- a scheduler or serving control plane
- a universal ML compiler
- a replacement for CUDA, Triton, FlashInfer, TensorRT, or engine-native runtimes

## V1 scope

V1 is intentionally narrow and deep:

- engine: `vLLM`
- hardware: NVIDIA
- platform: Linux

The goal is to get deep closure quality for one real production engine before expanding breadth.

## Thesis

The build-speed problem in inference is not mainly a compiler-speed problem.
It is a closure-design problem.

The fastest build system is the one that can prove most of the required world is already compiled, packaged, linked, and admissible for reuse.

sock exists to make that world explicit.

## Current code shape

The canonical semantic pipeline is:

`RawRequest -> NormalizedRequest -> ResolvedBuildPlan -> ArtifactClosure -> VerificationReport`

The Rust workspace is organized as:

- `app/`: CLI entrypoint
- `core/`: canonical semantic core, identities, validation, and replay helpers
- `engine/`: engine-facing planner glue
- `vllm/`: vendored `vLLM` source tree, pinned locally for deep integration work

## Status

This project is early, but it already has:

- a compiling Rust workspace
- a canonical `RawRequest -> NormalizedRequest -> ResolvedBuildPlan -> ArtifactClosure -> VerificationReport` contract
- a deterministic `vLLM` planning path for NVIDIA/Linux with source-anchored compile regions
- `vLLM` adapter truth for canonical region identity, cache ownership, topology-sensitive warmup surfaces, and machine-checkable residual JIT triggers
- replay bundle emission and fail-closed bundle validation
- CLI-visible explain, verify, replay, and doctor surfaces
- compile-free `verify` and `replay` operator gates rendered in the verification surface
- vendored `vLLM` source for source-aligned adapter work

## Operator workflow

The production CLI surface is:

- `cargo run --bin sock -- plan`
- `cargo run --bin sock -- explain`
- `cargo run --bin sock -- build --out /tmp/sock-bundle`
- `cargo run --bin sock -- verify --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- replay --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- doctor`

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

## Repository map

- `SPEC.md`: product spec
- `FIRST_PRINCIPLES_REPORT.md`: architectural reasoning
