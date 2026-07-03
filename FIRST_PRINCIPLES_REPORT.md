# sock first-principles report

## Goal

This report asks a narrower and more important question than "how do we wrap `vLLM` nicely?"

The real question is:

How do we turn a messy, partially-lazy, multi-backend inference startup stack into a build product with:

- very low cold-start latency
- strong closure guarantees
- reproducible artifact identity
- fast cache reuse
- great operator DX

The target is not just "understand `vLLM` compilation."
The target is to make the whole startup and materialization path legible enough that we can optimize it aggressively and systematically.

References used:

- [SPEC.md](/Users/deepsaint/Desktop/sock/SPEC.md)
- [TINYGRADREF.md](/Users/deepsaint/Desktop/sock/TINYGRADREF.md)
- `jello` references in `/Users/deepsaint/Desktop/jello`
- [vLLM torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/)
- [vLLM compilation config](https://docs.vllm.ai/en/stable/api/vllm/config/compilation/)
- [vLLM CUDA Graphs design](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [vLLM IR design](https://docs.vllm.ai/en/v0.23.0/design/vllm_ir/)
- [vLLM Triton JIT issue #43009](https://github.com/vllm-project/vllm/issues/43009)
- [vLLM Triton JIT hang issue #45198](https://github.com/vllm-project/vllm/issues/45198)
- [PyTorch compile caching tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [PyTorch caching configuration tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)
- [PyTorch regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [PyTorch regional AoT compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)
- [PyTorch AOTInductor](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
- [PyTorch PT2 archive spec](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/export/pt2_archive.html)
- [PyTorch ahead-of-time compilation with torch.compile](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_compile.html)
- [FlashInfer installation](https://docs.flashinfer.ai/installation.html)
- [FlashInfer CLI](https://docs.flashinfer.ai/cli.html)

---

## Reality

### What is physically happening

Inference startup latency is the sum of multiple distinct costs:

- graph tracing and graph lowering
- kernel code generation
- Triton compilation
- autotuning
- extension loading or building
- cache lookup and cache population
- CUDA Graph capture
- distributed startup coordination
- warmup coverage execution

Those are different phenomena, with different invalidation rules.
They are often treated as one opaque blob called "compile time."

### What current systems actually do

`vLLM` already has serious compilation machinery:

- it integrates with `torch.compile`
- it uses compile caches
- it exposes `compile_sizes`
- it exposes `cudagraph_capture_sizes`
- it supports piecewise versus full CUDA graphs
- it has custom op wrapping and an emerging `vLLM IR`

`vLLM` also explicitly states that its `torch.compile` integration aims to finish compilation before serving requests, so requests do not trigger new compilations at runtime. Source: [vLLM torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/).

But `vLLM` also documents deep complexity:

- it drops many dynamic-shape guards
- it wraps attention as a custom op because the internals are too messy to trace directly
- it has many config interactions around `splitting_ops`, graph partitioning, and CUDA Graph modes
- it constrains `max_cudagraph_capture_size` partly to avoid startup cost explosions

Source: [vLLM torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/), [vLLM compilation config](https://docs.vllm.ai/en/stable/api/vllm/config/compilation/), [vLLM CUDA Graphs design](https://docs.vllm.ai/en/latest/design/cuda_graphs/).

### Evidence that the startup surface still leaks

Recent issues show Triton kernel JIT compilation still occurring during inference, with latency spikes and even hangs in multi-process startup paths. Sources:

- [Issue #43009](https://github.com/vllm-project/vllm/issues/43009)
- [Issue #45198](https://github.com/vllm-project/vllm/issues/45198)

This means the closure is not complete in practice, even if parts of the stack already aim for full pre-serve compilation.

### External compiler reality

PyTorch now supports:

- compile-time caches
- configurable cache layers
- regional compilation to reduce cold-start cost
- ahead-of-time compilation with `torch.compile().aot_compile()`
- AOTInductor packaging into `.pt2` archives
- multi-model packaging inside one artifact

Sources:

- [compile caching tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [caching config tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)
- [regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [regional AoT compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)
- [AOTInductor](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
- [AOT torch.compile](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_compile.html)
- [PT2 archive spec](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/export/pt2_archive.html)

FlashInfer also already offers optional precompiled binaries and prebuilt JIT cache packages, explicitly to eliminate compilation and download overhead at runtime. Source: [FlashInfer installation](https://docs.flashinfer.ai/installation.html).

---

## Interpretation

The common interpretation is:

"`vLLM` compile is slow because LLM inference is inherently complex."

That is true but incomplete.

The deeper interpretation is:

`vLLM` startup feels bad because too many semantically different compile and materialization events are cohabiting one unstructured lifecycle.

The problem is not only too much compilation.
The problem is that the system does not present a stable, minimal, reusable compilation closure as a first-class artifact.

Today the stack behaves more like:

- discover ad hoc
- compile lazily or semi-lazily
- fuse policy with execution
- mix cache lookup with codegen
- mix backend selection with warmup
- mix correctness closure with performance optimization

That creates four pathologies:

1. Over-compilation.
2. Duplicate compilation.
3. Unprovable closure.
4. Bad UX because startup behavior is not explainable.

---

## Contradictions

### Contradiction 1

Expectation:
If `vLLM` says compilation finishes before serving, runtime JIT should not appear.

Evidence:
Recent Triton JIT warnings still appear during inference in the field. Sources:

- [Issue #43009](https://github.com/vllm-project/vllm/issues/43009)
- [Issue #45198](https://github.com/vllm-project/vllm/issues/45198)

Implication:
The closure being achieved is narrower than the closure users think they are getting.

### Contradiction 2

Expectation:
Dynamic-shape flexibility should reduce operational pain.

Evidence:
`vLLM` guard dropping exists precisely because default dynamic guard behavior conflicts with its compile strategy. Source: [vLLM torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/).

Implication:
Dynamic generality and deterministic startup are in tension.
The system needs explicit shape envelopes, not infinite flexibility.

### Contradiction 3

Expectation:
Full-graph compilation should be the cleanest and fastest path.

Evidence:
PyTorch's regional compilation guidance shows smaller repeated regions can dramatically reduce cold-start compile time while often preserving most performance gains. Sources:

- [regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [regional AoT compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)

Implication:
Whole-graph maximalism is likely the wrong default coordinate system for inference build speed.

### Contradiction 4

Expectation:
Runtime kernel stacks should be unavoidable because kernels are inherently dynamic.

Evidence:
FlashInfer already distributes precompiled binary packages and prebuilt JIT caches. Source: [FlashInfer installation](https://docs.flashinfer.ai/installation.html).

Implication:
A meaningful fraction of "runtime compilation" is a packaging failure, not a physics requirement.

### Contradiction 5

Expectation:
Startup is one thing, so a single cache is enough.

Evidence:
PyTorch documents several different compile caches; `vLLM` has separate compile cache semantics; FlashInfer has its own kernel binary or JIT cache surface. Sources:

- [compile caching tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [caching config tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)
- [vLLM compilation config](https://docs.vllm.ai/en/stable/api/vllm/config/compilation/)
- [FlashInfer installation](https://docs.flashinfer.ai/installation.html)

Implication:
One cache key and one cache directory is the wrong abstraction.

---

## What The Lamp Reveals

The lamp is not "vLLM compile is slow."

The lamp is:

Inference startup is actually a distributed artifact-linking problem with dynamic-shape envelopes.

That reframing matters.

If startup is artifact linking, then the key questions become:

- what exact artifacts are required?
- which are interchangeable?
- which are shape-specialized?
- which are backend-specialized?
- which can be prebuilt versus generated?
- which are reusable across models, revisions, or hardware families?
- which are identical but currently rebuilt because the system lacks canonical identity?

From there, the hidden opportunity appears:

The fastest build system is not the one that compiles fastest.
It is the one that proves most of the needed world is already compiled, packaged, linked, and admissible for reuse.

So the primary optimization target is not compiler speed alone.
It is closure minimization plus maximal reuse.

---

## Better Abstraction

The better abstraction is not "compile vLLM."

The better abstraction is:

`sock` should be an inference artifact closure optimizer.

That means each build has three distinct outputs:

1. A `BuildPlan`.
2. An `ArtifactClosure`.
3. A `GuaranteeEnvelope`.

### 1. BuildPlan

The plan chooses:

- engine path
- backend family
- kernel families
- shape envelopes
- warmup obligations
- compile regions
- CUDA Graph capture envelopes
- cache reuse candidates
- verification work

### 2. ArtifactClosure

This is the actual compiled world:

- compiled graph artifacts
- Triton binaries
- extension `.so` files
- prepackaged backend kernels
- autotune results
- CUDA Graph captures
- topology-scoped cache artifacts

### 3. GuaranteeEnvelope

This says:

- what request shapes are covered
- what kernels are covered
- what topology is covered
- what still may compile or specialize later

Once these are separate, optimization becomes much clearer.

---

## Optimization Program

Below is the concrete engineering program I would pursue.

### A. Stop optimizing "startup" as one blob

Split startup into separately measurable phases:

- environment discovery
- artifact identity resolution
- reusable artifact lookup
- graph trace/export
- graph lowering
- Triton compilation
- autotune
- extension load/build
- CUDA Graph capture
- warmup execution
- verification

Unique optimization:

Make every phase independently cacheable and independently skippable.

Why this matters:

If one warmup shape misses cache, we should not pay graph trace or backend resolution again.

### B. Build shape-envelope lattices instead of open-ended dynamic support

Current systems often pretend they support "dynamic shapes" broadly.
That usually means hidden recompiles or specialization debt.

Unique optimization:

Represent coverage as a finite lattice:

- symbolic range nodes
- exact hot-size nodes
- fallback range nodes
- uncovered residual nodes

Then optimize for:

- highest traffic coverage per compiled artifact
- lowest number of exact-shape specializations
- smallest residual runtime-JIT region

This turns compile planning into a weighted set-cover problem rather than a vague warmup heuristic.

### C. Use regional compilation as a first-class vLLM strategy

PyTorch's regional compilation result is directly relevant because Transformer stacks are repetitive. Sources:

- [regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [regional AoT compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)

Unique optimization:

Teach `sock` to discover compile-equivalence regions inside `vLLM` workloads:

- repeated block bodies
- repeated decode micrographs
- repeated prefill micrographs
- repeated MoE dispatch regions

Then compile those regions once and reuse them across repeated instances whenever correctness allows.

This is likely one of the highest-upside compile-time reductions.

### D. Hybrid JIT/AoT planning instead of ideology

PyTorch now supports AOT packaging of compiled artifacts, including Triton compilation and autotuning in the ahead-of-time path. Source: [AOT torch.compile](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_compile.html).

Unique optimization:

Use three tiers:

- Tier 0: prebuilt vendor/backend artifacts
- Tier 1: ahead-of-time compiled stable regions
- Tier 2: bounded JIT only for truly irreducible dynamic tails

This is better than "all JIT" or "all AoT."

### E. Treat precompiled backend packages as build inputs, not deployment trivia

FlashInfer explicitly offers:

- `flashinfer-cubin`
- `flashinfer-jit-cache`

to eliminate runtime compile and download overhead. Source: [FlashInfer installation](https://docs.flashinfer.ai/installation.html).

Unique optimization:

`sock` should resolve backend artifact packages as first-class dependencies and prefer them automatically when compatible.

That means:

- detect whether precompiled FlashInfer artifacts exist
- pull them into the closure if valid
- refuse to silently fall back to JIT unless policy allows it

This should generalize:

- precompiled Triton bundles when feasible
- prebuilt extension wheels
- pre-baked PT2 archives

### F. Canonicalize artifact identity far more aggressively than the underlying engines do

Most duplicate compile work happens because semantically equivalent requests are not recognized as equivalent early enough.

Unique optimization:

Introduce multi-layer content-addressed identity:

- request identity
- shape-envelope identity
- graph-region identity
- backend capability identity
- ABI identity
- compiled artifact identity
- warmup evidence identity

This should be stricter than path-based caches and more semantic than raw CLI hashing.

### G. Make compile cache portability explicit and graded

`vLLM` already uses model-related info to derive cache directories and supports saving caches in binary or unpacked formats. Source: [vLLM compilation config](https://docs.vllm.ai/en/stable/api/vllm/config/compilation/).

Unique optimization:

Classify reuse scopes:

- process-local
- host-local
- identical-driver identical-CUDA host pool
- architecture-family portable
- topology-portable
- non-portable

This lets `sock` safely import caches across machines instead of treating cache portability as binary.

### H. Separate "compile closure" from "performance closure"

A system can be functionally ready before it is fully CUDA-graph captured or fully autotuned.

Unique optimization:

Define two closure planes:

- correctness closure
- performance closure

Then allow operator policies like:

- serve when correctness closure is complete
- continue background optimization until performance closure is complete

This can massively improve perceived startup without lying.

### I. Leader/follower distributed compilation

Multi-worker startup can amplify compilation waste and contention.
Recent issue evidence suggests this can interact badly with Triton JIT and distributed startup. Source: [Issue #45198](https://github.com/vllm-project/vllm/issues/45198).

Unique optimization:

Compile once per equivalence class, then fan out artifacts:

- elect one leader per artifact family
- followers wait on artifact availability, not on redundant compilation
- verify checksums and ABI fingerprints before adoption

This is much stronger than "each rank warms itself up."

### J. Compile-range planning, not only compile-size planning

`vLLM` already exposes exact compile sizes and compile range endpoints. Source: [vLLM compilation config](https://docs.vllm.ai/en/stable/api/vllm/config/compilation/).

Unique optimization:

Choose ranges using workload priors:

- frequent decode bucket sizes
- common prefill batch sizes
- model-specific MoE routing burst shapes
- topology-specific all2all boundaries

This should be traffic-aware, not static.

### K. Replace monolithic warmup with coverage witnesses

Today warmup often means "run some requests."

Unique optimization:

Warmup should emit machine-checkable witnesses:

- which graph regions executed
- which kernels compiled
- which exact shapes were covered
- which CUDA Graph captures completed
- which residual kernel families remain uncovered

This turns warmup from ritual into proof.

### L. Use vLLM IR as a leverage point, not just a vLLM detail

`vLLM IR` exists specifically to separate operator semantics from implementation and dispatching. Source: [vLLM IR design](https://docs.vllm.ai/en/v0.23.0/design/vllm_ir/).

Unique optimization:

Sock should reason at the highest stable semantic layer available.

When possible:

- plan at IR/operator-family level
- lower to backend-specific artifacts later

This reduces cache fragmentation and enables backend substitution.

### M. Build a compile daemon, not just a CLI

Cold startup is partially a process-lifetime problem.

Unique optimization:

Keep a resident `sockd` process that maintains:

- discovered capability fingerprints
- open cache indices
- hot compiler workers
- compile trace stats
- artifact availability maps

This can dramatically reduce repeated Python and compiler process startup overhead.

### N. Make autotune a build asset

PyTorch AOT compilation now explicitly includes Triton kernel compilation and autotuning in the ahead-of-time path. Source: [AOT torch.compile](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_compile.html).

Unique optimization:

Persist autotune results as first-class artifacts with strict applicability metadata.

That lets us:

- separate code generation from parameter search
- reuse tuning across repeated builds
- ship topology-aware tuning bundles

### O. Introduce a cost model for compile ROI

Not every additional compiled shape is worth the startup cost.

Unique optimization:

Score each candidate artifact by:

- startup cost
- expected traffic mass covered
- expected latency saved
- expected duplication reduced
- cache portability
- residual risk eliminated

Then choose the optimal frontier per policy:

- fastest cold start
- lowest runtime jitter
- maximum offline closure
- CI validation mode

### P. Multi-model packaging should be a first-class future target

PyTorch PT2 archive supports multiple model definitions in one artifact. Source: [PT2 archive spec](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/export/pt2_archive.html).

Unique optimization:

Eventually `sock` should package:

- one model
- one engine revision
- one backend family
- multiple shape envelopes
- multiple compile regions

inside one portable closure bundle.

That gives us a clean future for fleet-wide reuse and staged deployment.

---

## What Is Proven

These claims are strongly supported by current evidence.

### Proven 1

The build problem is multi-layered, and treating it as one cache or one compile step is wrong.

### Proven 2

Runtime JIT still leaks in real `vLLM` deployments, so full startup closure is not guaranteed in practice today.

### Proven 3

Regional compilation is a real and current lever for reducing cold-start compile time in repeated-structure models.

### Proven 4

AOT packaging is becoming practical enough that `sock` should target it for stable subgraphs or stable regions.

### Proven 5

Precompiled backend assets can meaningfully erase runtime compile work, at least for some kernel families.

### Proven 6

`vLLM` already contains enough internal structure that `sock` should model compilation as a graph of regions, ops, and backend choices rather than a flat startup script.

---

## What Is Still Unknown

### Unknown 1

How much of real `vLLM` startup time is graph tracing versus Triton codegen versus autotune versus CUDA Graph capture versus backend-specific warmup?

This needs direct measurement.

### Unknown 2

How much reuse is available across models of the same architecture family, not just exact model revisions?

### Unknown 3

How well can `vLLM` workloads be partitioned into compile-equivalent repeated regions without losing meaningful runtime performance?

### Unknown 4

How much of attention and KV-cache update complexity can be lifted into a reusable semantic planning layer before backend choice?

### Unknown 5

Which backend stacks offer the highest prebuild leverage:

- FlashInfer
- Triton
- custom C++/CUDA extensions
- AOTInductor regions

### Unknown 6

What is the best serving policy boundary between correctness closure and performance closure?

---

## Next Experiments

These are the most valuable falsifiable experiments to run first.

### Experiment 1: Compile-phase attribution

Instrument a representative `vLLM` startup and split time into:

- import and engine boot
- graph capture
- inductor lowering
- Triton compile
- autotune
- CUDA Graph capture
- warmup execution

Goal:

Determine where the real time is going before optimizing blindly.

### Experiment 2: Runtime JIT witness collection

Create a kernel-JIT monitor pass that records:

- kernel name
- triggering shape
- backend family
- request phase
- rank and topology

Goal:

Build a residual-JIT dataset and stop arguing from anecdotes.

### Experiment 3: Region equivalence mining

Mine repeated graph regions or IR regions across:

- repeated transformer blocks
- decode paths
- prefill paths
- MoE branches

Goal:

Quantify whether regional compilation can dramatically shrink compile work for real `vLLM` deployments.

### Experiment 4: FlashInfer-prebuilt-first policy

Run the same build with:

- default FlashInfer setup
- `flashinfer-cubin`
- `flashinfer-jit-cache`

Goal:

Measure how much startup work disappears when backend artifacts are resolved upfront.

### Experiment 5: AoT stable-region prototype

Take one stable repeated region and compile it with:

- ordinary `torch.compile`
- regional compilation
- AOTInductor or `torch.compile().aot_compile()` path if admissible

Goal:

See whether stable-region AoT is viable for `sock` closures.

### Experiment 6: Distributed leader/follower compilation

In multi-rank startup, compile artifacts on one leader and distribute them.

Goal:

Measure whether redundant rank-local compilation is a major source of startup pain.

### Experiment 7: Shape-envelope optimizer

Use production-like traffic histograms and select:

- compile sizes
- compile ranges
- cudagraph capture sizes

by optimization instead of fixed defaults.

Goal:

Find the Pareto frontier between startup time and runtime jitter.

### Experiment 8: Correctness-closure early serve

Split startup into:

- minimum closure needed for safe serve
- deferred performance closure

Goal:

Test whether perceived startup can be slashed without lying about residual performance risk.

---

## Honest Thesis

The most important conclusion is this:

The build-speed problem is not primarily a compiler-speed problem.
It is a closure-design problem.

If we design `sock` as a nicer wrapper over existing startup flows, we will improve DX but not fundamentally solve cold start.

If we design `sock` as a closure optimizer, we unlock a much stronger path:

- minimize the number of distinct things that need compiling
- package or pre-resolve everything that can be pre-resolved
- compile only the smallest semantically sufficient regions
- treat runtime JIT as a measurable residual debt
- separate correctness closure from performance closure
- distribute artifacts instead of duplicating work
- make shape coverage explicit and optimized

That is the path to both:

- clean DX
- extremely fast builds

The best version of `sock` is not "Nix for inference engines" in the package-manager sense.

It is:

`jello` for inference startup plus a closure optimizer for dynamic compiler stacks.

That is a stronger and more defensible abstraction.

---

## Practical recommendation

If we want the highest-probability path to a great V1, we should prioritize work in this order:

1. Measure and classify real startup cost.
2. Build canonical artifact and shape-envelope identity.
3. Implement residual-JIT witnesses.
4. Add backend artifact resolution for prebuilt assets.
5. Prototype regional compilation for `vLLM` repeated regions.
6. Add leader/follower distributed materialization.
7. Prototype AoT packaging only for the stable subgraphs that actually justify it.

This sequencing is important.
It avoids premature commitment to one compiler ideology and keeps us aligned with evidence.
