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
- a canonical plan/closure/verification scaffold
- a minimal deterministic planner
- vendored `vLLM` source for source-aligned adapter work

## Repository map

- `SPEC.md`: product spec
- `FIRST_PRINCIPLES_REPORT.md`: architectural reasoning
