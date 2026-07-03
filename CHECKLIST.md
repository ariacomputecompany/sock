# sock checklist

This is the active sober implementation checklist for the remaining Lane A work.
It is intentionally one-by-one, production-oriented, and limited to the highest-confidence build-safe path.

Scope:

- production `sock`
- modular engine architecture
- deep V1 focus on vendored `vLLM`
- V1 deployment target: NVIDIA on Linux
- north star: deterministic partial builds, fast cold start, great operator DX

Working rule:

- execute in order unless a later item becomes a strict prerequisite
- remove completed items instead of annotating them
- treat any runtime-adjacent drift as an escalation out of Lane A

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

- [ ] Audit every external project and fetched dependency in the vendored `vllm` build.
- [ ] Eliminate redundant fetch, unpack, and configure work across repeated local builds.
- [ ] Prefer vendored-resolved source manifests or local mirrors over incidental network fetch during normal iteration.
- [ ] Gate heavyweight external dependency preparation behind the selected backend/build profile.
- [ ] Cache external dependency resolution results in a way that survives normal local rebuilds.
- [ ] Keep third-party acquisition manifests explicit so local iteration can reuse already-resolved payloads without refetch or broad reconfigure.
- [ ] Prefer a manifest- or symlink-based editable-build strategy over repeated tree copy-back for `vllm_flash_attn`, `triton_kernels`, `deep_gemm`, and `fmha_sm100`.

## 17. Toolchain and Build Parallelism Tuning

- [ ] Audit current compiler, linker, and Python build orchestration for serialized bottlenecks.
- [ ] Reduce redundant single-threaded phases during local iteration.
- [ ] Ensure parallelism settings are explicit and tuned for local workstation builds.
- [ ] Reduce avoidable Python-side orchestration overhead between native targets.
- [ ] Distinguish configure-time parallelism, compile-time parallelism, and packaging-time parallelism in observability output.
- [ ] Make build concurrency choices visible in explain output so slow builds can be diagnosed from artifacts instead of guesswork.
