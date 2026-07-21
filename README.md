# sock

sock is an inference-engine graph compiler built around a vendored `vLLM` fork.

The short version: sock cleans up and tightens `vLLM` internals, then wraps them in a lazy compilation graph so you can get to inference with as little ceremony, waiting, and runtime guesswork as possible.

The thesis is simple: inference engines should behave more like compiler targets. Given a serving intent, sock should build the smallest valid graph needed to serve that workload instead of forcing every request through one broad, opaque startup path.

That means:

- build exactly what you need
- reuse exactly what is admissible
- prove exactly what is ready to serve
- make hidden compile work and stale cache assumptions impossible to ignore

## What sock does

sock turns inference-engine startup into a deterministic, demand-driven build graph.

The canonical pipeline is:

`RawRequest -> NormalizedRequest -> ResolvedBuildPlan -> ArtifactClosure -> VerificationReport`

For each requested serving intent, sock:

- resolves engine-specific compile regions and backend choices
- computes the minimal valid artifact closure
- materializes the required compiled outputs, caches, and warmups
- verifies artifact admissibility, readiness, and runtime-JIT bounds
- emits a replayable bundle that can be checked without new compile work

In practice, that means:

- if you only need a prefill path, sock builds the prefill closure
- if a requested subset is semantically real in vendored `vLLM`, sock builds only that closure
- if a requested subset actually depends on broader worker startup, sock fails closed instead of pretending it is standalone
- if an artifact is already valid for reuse, sock proves that and skips rebuilding it
- if a requested scope would still leak runtime JIT, sock surfaces that as a bounded contract instead of hiding it

## Why it exists

Modern inference engines are powerful, but getting from "I want to serve this model in this mode" to "the runtime is actually ready" often means suffering through too much upfront machinery:

- graph compilation
- backend selection
- Triton codegen
- FlashInfer setup
- cache shaping
- CUDA graph capture
- warmup
- topology fanout

When all of that is coupled into one startup lifecycle, operators get long cold starts, bad visibility, and very little control over what is actually being built.

sock separates those concerns into explicit build units: compile-equivalent regions, cache ownership boundaries, backend-specific artifact scopes, topology-sensitive materialization paths, warmup scopes, and runtime-JIT risk surfaces. Those units are then compiled only when the requested serving graph actually needs them.

## vLLM integration

The vendored fork preserves real `vLLM` structure rather than flattening it into fake-generic abstractions. sock models canonical compile region identity, backend binding, cache ownership, topology-sensitive warmup paths, backend-specific runtime-JIT triggers, and materialization boundaries derived from `vLLM` source.

This is what lets sock compile parts of the engine intentionally instead of treating `vLLM` startup as one opaque side effect.

V1 goes deep before it goes broad:

- engine: `vLLM`
- hardware: AMD/ROCm and NVIDIA/CUDA
- platform: Linux

## Scope

sock is:

- a deterministic build system for serving closures
- a planner and executor for compiled regions, kernels, caches, and warmup work
- a proof surface for artifact reuse, readiness, and no-surprise-JIT claims

sock is not:

- a new inference engine from scratch
- a serving control plane
- a model training framework
- a replacement for CUDA, Triton, FlashInfer, TensorRT, or engine-native runtimes

sock compiles and materializes the engine-specific startup world around `vLLM`.
It does not replace the low-level runtimes that ultimately execute the model.

## Examples

- `prefill_attention` can be planned and materialized separately from `decode_attention`
- `decode_attention` and `kv_cache_update` are tracked as explicit `vLLM` surfaces, but subset builds fail closed when vendored startup paths still require mixed-batch worker context
- backend autotune caches are treated as first-class artifacts instead of hidden side effects
- CUDA graph captures are handled as topology-scoped rank-local outputs
- leader/follower artifact fanout is planned explicitly instead of emerging accidentally at runtime
- invalidation evicts only stale siblings in the affected cache namespace and invalidation domain instead of blowing away unrelated artifact closures

## Operator workflow

The CLI surface is:

- `cargo run --bin sock -- install-runtime --profile auto`
- `cargo run --bin sock -- install-runtime --profile cuda --build-profile minimal-dev --recreate-venv`
- `cargo run --bin sock -- install-runtime --profile cuda --build-profile minimal-dev --dry-run --format json`
- `cargo run --bin sock -- prepare prefill-path --out /tmp/sock-bundle`
- `cargo run --bin sock -- prepare replay-safe-closure --out /tmp/sock-bundle`
- `cargo run --bin sock -- measure prefill-path --out /tmp/sock-measure`
- `cargo run --bin sock -- plan`
- `cargo run --bin sock -- explain`
- `cargo run --bin sock -- build --out /tmp/sock-bundle`
- `cargo run --bin sock -- verify --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- replay --bundle /tmp/sock-bundle`
- `cargo run --bin sock -- doctor`

Runtime installation is resolved from `runtime.buildplan.json` and the
backend-neutral top-level `requirements.txt`. The build plan then selects the
single valid accelerator dependency set for the host, because CUDA and ROCm
torch/runtime wheels are mutually exclusive install universes:

- CUDA installs `requirements.txt`, `vllm/requirements/build/cuda.txt`, and `vllm/requirements/cuda.txt`.
- ROCm installs `requirements.txt`, `vllm/requirements/build/rocm.txt`, and `vllm/requirements/rocm.txt`.
- Host packages are limited to compiler/toolchain, Git, Python venv support, Python headers, and the vendor driver/runtime probe. Python build tools such as CMake and Ninja come from `requirements.txt` and are resolved inside `vllm/.venv`.
- The installer emits the resolved environment, native build profile, CMake defines, requirements, and exact command steps in JSON before doing work when run with `--dry-run --format json`.
- `--preflight-only` fails closed on missing build tools, Python headers, or accelerator probes.
- `--recreate-venv` removes `vllm/.venv` before installation for clean-machine validation.

CUDA production serving should use the default compiled/CUDA-graph vLLM path
when the target model/profile has enough KV headroom. On the RTX 4090 validation
host, compiled mode materially improved Qwen3-4B throughput and still improved
Qwen3-8B TTFT/throughput. Use `--enforce-eager` for deterministic bring-up,
debugging, or tight-memory profiles where compile/CUDA-graph reservations reduce
KV capacity too far.

The workflow is:

1. install the deterministic accelerator runtime with `sock install-runtime`
2. describe the serving intent
3. inspect the requested scope, expanded closure, and estimated work
4. measure the scoped closure against a broad build on the live executor path
5. build the required subset
6. verify the emitted bundle
7. replay the result without new compile work

`measure` records three concrete executions:

- a broad cold build
- a scoped cold build for the requested intent
- a scoped warm build that reuses the same cache root

That report is written to `measurement_report.json` and proves, with the same executor used for production bundles, whether scoped closure and cache reuse are actually reducing work.

## Verification model

Verification is strict and fail-closed. sock verifies:

- artifact admissibility
- plan identity consistency
- cache and bundle integrity
- warmup proof coverage
- runtime-JIT evidence bounds
- observed runtime-JIT contradictions from the live build path
- compile-free verify/replay operator gates
- measured scoped-vs-broad materialization reduction
- measured warm-cache reuse decisions

Replay bundles are content-digested, identity-checked, and verification-checked.
Invalid reuse, stale artifacts, and mismatched plan state are rejected instead of being repaired implicitly.

## Repository layout

- `app/`: CLI and shared application contract surface
- `core/`: canonical schemas, identity, validation, bundle logic, verification, and rendering
- `engine/`: planner, executor integration, and `vLLM`-specific build logic
- `vllm/`: vendored `vLLM` source tree used for direct integration
