# sock

sock is a deterministic build driver for inference engines.

It is focused on turning inference startup, warmup, backend selection, kernel materialization, cache reuse, and runtime-compile risk into a first-class build product with strong diagnostics and replayable artifacts.

## What it is

- a semantic orchestration layer above inference engines
- an artifact-closure optimizer for inference startup
- a deterministic planner for backend, kernel, cache, and warmup strategy
- a system for proving what has and has not been materialized before serving

## What it is not

- an inference engine
- a scheduler or serving control plane
- a universal ML compiler
- a replacement for CUDA, Triton, FlashInfer, TensorRT, or engine-native runtimes

## V1 focus

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

## Current repo contents

- `SPEC.md`
- `FIRST_PRINCIPLES_REPORT.md`

## Status

This project is early.
The repository currently contains the product/specification foundation and implementation roadmap for the first build of the system.
