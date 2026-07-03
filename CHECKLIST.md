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
