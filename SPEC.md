# sock

## Working Title

sock is a deterministic build, compile, and startup materialization driver for inference engines.

It sits above engines like `vLLM`, `SGLang`, `TensorRT-LLM`, and similar runtimes, and turns messy engine-specific startup, warmup, kernel selection, cache behavior, custom tensor-op backends, and backend configuration into a clean, inspectable, reproducible build artifact.

The project is not an inference engine.
The project is not a new tensor compiler IR from scratch.
The project is not a model serving control plane.

The project is a semantic build driver for inference engine compilation, startup materialization, and runtime-compile risk closure.

---

## Core Problem

Modern inference serving stacks have a bad operational shape:

- runtime JIT appears unexpectedly during serving
- warmup behavior is inconsistent across backends and architectures
- kernel and backend selection logic is scattered across engine internals
- custom tensor-op enablement is spread across flags, environment variables, imports, extension builds, and runtime fallbacks
- cache behavior is opaque
- configuration surfaces are large, overlapping, and under-validated
- startup failures are often rooted in hidden backend assumptions or ABI mismatches
- production operators only learn what the system is really doing after it starts failing

This creates "config hell" at exactly the wrong time: live deployment, interview environments, production bring-up, and high-pressure debugging.

The specific pain is not merely "compilation is slow."
The real pain is:

- compilation happens at the wrong time
- compilation behavior is not legible
- compilation outcomes are not deterministic enough
- engine startup does not provide strong guarantees about what has already been materialized versus what can still compile lazily
- custom kernels and extensions are present in practice but not represented explicitly in planning artifacts

sock exists to move that chaos left.

---

## Product Thesis

Inference engines should be buildable through a deterministic compile driver in the same way linkers can be mediated through a deterministic linker driver.

Given:

- an engine target, such as `vllm` or `sglang`
- a model
- a hardware profile
- a serving topology
- a backend policy
- a warmup policy
- a cache policy
- an execution guarantee target

sock should produce a first-class build artifact that answers:

- what engine backend paths will be used
- what kernels, extensions, or engine-specific components will be materialized
- what warmup coverage will be executed
- what caches will be created or reused
- what remains capable of runtime compilation
- what configuration assumptions were applied
- what incompatibilities were detected and repaired
- how to replay the exact build on another machine
- what guarantee envelope was actually achieved

This artifact is the core value.

The ambition is effectively "Nix for inference engine startup and compilation surfaces," but constrained to inference-engine build closure rather than general package management:

- closed-world identity for the build inputs that matter
- explicit artifact graphs
- deterministic materialization
- evidence-backed compatibility and replay

---

## Direct Inspiration From jello

The closest internal precedent is `jello`, which already solves the same class of problem for the linker layer.

`jello`'s most important reusable ideas are:

1. It accepts real-world chaotic invocations instead of demanding a clean abstract API.
2. It normalizes them into a canonical internal model.
3. It resolves environment and backend selection as an explicit phase.
4. It produces an immutable plan artifact before execution.
5. It separates planning, execution, diagnostics, and artifact emission.
6. It turns folklore and side effects into data.

Relevant `jello` concepts to reuse:

- `LinkPlan` as a first-class IR artifact
- layered config resolution: env > project > user > defaults
- explicit discover/resolve phases
- explain mode
- replayable emitted artifacts
- confidence-scored auto-fixes
- deterministic backend selection
- structured diagnostics instead of raw tool output

Relevant `jello` files and ideas:

- `/Users/deepsaint/Desktop/jello/README.md`
  - defines the core value of a deterministic plan artifact
- `/Users/deepsaint/Desktop/jello/lib/driver.ml`
  - defines the orchestration pipeline:
    - parse
    - normalize
    - discover
    - resolve
    - reorder/fix
    - plan
    - execute
    - diagnose
    - emit
- `/Users/deepsaint/Desktop/jello/lib/types.ml`
  - shows the value of making core concepts explicit in the type system
- `/Users/deepsaint/Desktop/jello/lib/config.ml`
  - defines the layered config model
- `/Users/deepsaint/Desktop/jello/lib/backend/emit.ml`
  - defines replay and artifact emission

The most important reuse is conceptual, but several structural patterns can be lifted almost directly.

---

## Important Engineering Lessons From tinygrad

The strongest import from `TINYGRADREF.md` is not tensor math or codegen logic. It is compiler engineering discipline.

Key ideas to adopt:

1. One canonical resolved build IR with stable structural identity.
2. Named rewrite passes instead of config-phase soup.
3. Validation as an executable spec inside the pipeline.
4. Separate compile, link/assemble, and execute/verify phases.
5. Semantic caches for different pipeline layers, not one giant cache.
6. Dependency and hazard tracking as first-class data.
7. Replayability and renderability in the core, not tacked on later.
8. Queue-graph style materialization planning instead of only flat task lists.

sock should copy tinygrad's engineering substrate, not tinygrad's tensor semantics.

---

## What sock Is

sock is a deterministic inference build driver.

It should:

- ingest engine-oriented build requests
- normalize configuration into a canonical `BuildPlan`
- discover the actual machine and backend capabilities
- resolve engine-specific backend, extension, and kernel choices
- precompute and materialize startup-sensitive compile work
- emit artifacts and diagnostics that make behavior reproducible
- provide evidence-backed guarantees about what has and has not been materialized
- verify whether the claimed runtime-compile closure actually holds for a declared envelope

sock should make an engine build understandable before the engine starts serving traffic.

---

## What sock Is Not

sock is not:

- a replacement for `vLLM`, `SGLang`, or `TensorRT-LLM`
- a universal model execution runtime
- a scheduler or autoscaler
- a replacement for CUDA, Triton, TileLang, CUTLASS, DeepGemm, FlashInfer, or TensorRT
- a general ML compiler framework in the TVM sense
- a serving API server
- a cluster manager

sock is a mediation layer between operator intent and engine/backend startup reality.

That said, it must model much more of the runtime compilation surface than a thin wrapper would.
If an engine's correctness or startup closure depends on custom kernels, extension builds, cache identities, tensor-op dispatch, or shape-specialized codegen, those must appear in sock's modeled artifact graph.

---

## Project Positioning

The space already contains:

- inference engines
- engine-specific compilers
- machine learning compiler frameworks
- serving orchestrators

But the specific cross-engine layer appears underserved:

- deterministic build orchestration across inference engines
- compile, warmup, and cache planning as a first-class product
- artifact-closure modeling across engine-specific kernel stacks
- explicit guarantees against surprise runtime compilation
- unified diagnostics across engine and backend combinations

This suggests a narrow and defensible wedge:

sock is not trying to beat every engine at inference throughput.
sock is trying to make engine build and startup deterministic, legible, and operationally safe.

---

## Core Artifact

The central artifact should be something like:

- `BuildPlan`
- `EngineBuildPlan`
- `ServeBuildPlan`

Working name in this spec: `BuildPlan`.

This is the inference analogue of `jello`'s `LinkPlan`.

It should be an immutable, serializable snapshot of intent, resolution, evidence, and planned materialization.

### BuildPlan requirements

The `BuildPlan` must be:

- stable
- inspectable
- diffable
- replayable
- renderable into a human-readable deterministic form
- structurally hashable

### BuildPlan fields

At minimum:

- target engine
- engine version or source revision
- engine extensions and plugin surface
- model identifier
- model revision
- model format
- hardware profile
- GPU architecture and capabilities
- topology
- selected backend family
- selected kernel families
- selected cache locations
- warmup plan
- compilation coverage assumptions
- runtime fallbacks still possible
- environment overrides applied
- fixes applied
- diagnostics and warnings
- replay instructions
- artifact manifest
- guarantee envelope
- guarantee evidence
- ABI and environment fingerprint

Potential detailed schema:

```text
BuildPlan {
  raw_request
  normalized_request
  resolved_identity

  engine
  engine_revision
  engine_mode
  engine_extensions

  model
  model_revision
  model_format

  topology
  hardware
  abi_fingerprint

  capability_witnesses
  backend_selection
  kernel_selection
  dispatch_graph

  cache_policy
  cache_identity

  compile_policy
  warmup_policy
  shape_envelope

  artifact_graph
  materialization_waves
  artifact_manifest

  guarantee_target
  guarantee_level
  guarantee_envelope
  guarantee_evidence
  residual_runtime_risks

  assumptions
  fixes_applied
  diagnostics

  replay_steps
  render_key
  structural_key
}
```

The key requirement is that it be stable, inspectable, and diffable.

---

## Artifact Closure Model

The most important addition to the spec is that sock must model the compiled artifact closure, not merely high-level backend selection.

sock should represent:

- engine binaries and Python package identity
- compiled extensions and custom ops
- backend libraries and their ABI compatibility
- kernel family selection and expected specialization domains
- cache artifacts produced during warmup or ahead-of-time compilation
- topology-scoped artifacts such as per-rank or per-device materializations
- any runtime-generated artifact that may still be emitted lazily

For each artifact, sock should know:

- producer
- inputs
- identity key
- invalidation boundary
- placement
- portability scope
- verification method

This is the difference between a startup wrapper and a true build driver.

---

## Guarantee Envelope

sock must make guarantees in terms of explicit envelopes, not vague claims.

Important guarantee dimensions:

- environment closure
  - required binaries, libraries, Python packages, extensions, and ABI versions are compatible
- kernel closure
  - selected kernel families and custom ops are materialized or proven available
- shape closure
  - the declared request shape and dtype envelope has been covered
- runtime closure
  - no new compile, autotune, extension build, or codegen should occur inside the envelope
- topology closure
  - the same claim holds for the declared device, worker, and parallelism topology

Sock should expose guarantee levels such as:

- `strict-aot`
- `shape-bounded-aot`
- `mostly-aot`
- `runtime-jit-possible`
- `runtime-jit-likely`

These should be produced from evidence, not branding.

---

## Canonical Internal IR

sock should have a single canonical resolved build IR, analogous to the discipline described in `TINYGRADREF.md`.

Recommended internal objects:

- `RawRequest`
- `NormalizedRequest`
- `ResolvedBuildPlan`
- `MaterializationGraph`
- `VerificationReport`

Requirements:

- structural hashing for canonical identity
- interning of repeated substructures where useful
- round-trip rendering and parsing
- identity-affecting fixes recorded into the plan

Every cache key, explain artifact, replay artifact, warmup plan, and execution wave should hang off this canonical resolved form.

---

## Primary User Story

A user wants to run:

- `vLLM` on Blackwell with FP8
- `vLLM` with custom kernels and a declared no-surprise-JIT envelope
- `SGLang` on H100 with a specific backend combination
- `TensorRT-LLM` with a known deployment shape

Today the user is forced to trust engine internals to:

- pick valid backends
- select valid kernel families
- materialize custom extensions correctly
- avoid invalid warmups
- reuse caches correctly
- avoid surprise JIT
- explain failures clearly

With sock, the user should instead:

1. declare intent
2. generate a `BuildPlan`
3. inspect exactly what will happen
4. materialize artifacts
5. verify the resulting guarantee envelope
6. deploy with known guarantees

That is the entire product.

---

## Design Principles

### 1. Planning before execution

Execution should never be the first time behavior becomes legible.

### 2. Determinism over cleverness

If a build can vary across equivalent machines without an explicit reason, that is a bug.

### 3. Explainability is a feature

The system should not merely succeed.
It should make clear why it succeeded, what it chose, and what remains risky.

### 4. Auto-fix only when confidence is high

If a repair is not deterministic and safe, sock should suggest instead of silently mutating behavior.

### 5. Runtime JIT should be explicit debt

If runtime compilation remains possible after build, that must be surfaced as a structured warning, not an implicit surprise.

### 6. Engine heterogeneity is normal

sock must treat backend inconsistency as the default reality, not as an edge case.

### 7. Modularity without fake universality

sock should be modular across engines and backends, but should not collapse real engine differences into a misleading pseudo-universal model.

The correct approach is:

- shared abstractions where semantics truly match
- engine adapters where behaviors genuinely differ
- backend shims where materialization and verification paths differ

### 8. Normalize before lowering

High-level intent should be converted into explicit obligations before engine-specific emission begins.

### 9. Validation belongs inside the pipeline

Unresolved hazards are not just test failures.
They are plan invariant failures.

---

## Modular Architecture

sock should be deeply modular even while V1 is focused on `vLLM`.

The architecture should separate:

- common planning abstractions
- engine adapter interfaces
- backend capability tables
- artifact identity and cache logic
- materialization and verification executors

### Shared abstractions

The following should be engine-agnostic wherever possible:

- request normalization
- structural plan identity
- config layering
- capability witness representation
- artifact graph representation
- cache identity and invalidation logic
- guarantee envelope representation
- diagnostics and explain artifacts
- replay artifact generation

### Engine-specific shims

The following should be adapter-driven:

- extraction of engine-effective configuration
- enumeration of eligible execution paths
- enumeration of materializable artifacts
- engine-specific warmup invocation
- observation of runtime compile events
- mapping from engine outputs to shared diagnostics

This keeps the system reusable without pretending all engines behave the same.

---

## Engine Adapter Contract

Each engine should plug into sock through an explicit adapter contract.

Suggested adapter responsibilities:

- `discover_engine_surface`
- `extract_effective_config`
- `enumerate_execution_paths`
- `enumerate_materializable_artifacts`
- `resolve_backend_options`
- `build_warmup_coverage`
- `observe_runtime_materialization`
- `verify_closure_claims`
- `render_engine_explain`

Each adapter should publish:

- supported engine revisions
- feature flags and extension points
- known backend families
- known custom kernel families
- unsupported combinations
- guarantee limitations

The adapter contract should be strong enough that adding a new engine primarily means implementing a shim, not forking the whole system.

---

## Backend and Kernel Capability Model

sock will need a backend capability model that spans multiple engines.

Examples:

- Triton
- TileLang
- CUTLASS
- DeepGemm
- FlashInfer
- TensorRT engine generation
- eager fallback paths
- engine-specific custom CUDA or C++ extensions

This does not mean abstracting them into a fake universal execution model.
It means modeling the decision and compatibility surface cleanly.

Capability data should be centralized and declarative, not scattered across one-off resolver branches.

Important modeled facts include:

- backend availability
- supported architectures
- supported dtypes and layouts
- whether ahead-of-time materialization is supported
- whether cache artifacts are persistent and portable
- whether residual lazy specialization remains
- what evidence is required for a closure claim

---

## Warmup Planning

Warmup is currently ad hoc and engine-specific.

sock should make it explicit:

- what is warmable
- what shape and dtype coverage is intended
- what remains lazily specialized
- what compile cost is being paid now versus deferred
- what evidence will prove the warmup actually covered the intended surface

Warmup should be modeled as a graph of coverage obligations, not just a list of sample requests.

---

## Cache Policy

sock must understand:

- local versus shared caches
- persistent versus ephemeral caches
- invalidation boundaries
- versioned artifact identity
- host portability and topology portability
- engine-private versus shared backend caches

sock should maintain separate semantic cache layers for:

- normalized requests
- capability discovery
- resolved plans
- materialized artifacts
- warmup results
- runtime-compile witnesses
- explain and render outputs

Do not collapse all of these into one "build cache."

---

## Hazard and Dependency Model

sock should explicitly model:

- stale cache hazards
- warmup dependency hazards
- artifact invalidation hazards
- runtime fallback hazards
- backend handoff hazards
- overlapping writes into shared artifact spaces

Dependency tracking should be subresource-aware where useful:

- cache directories
- model shards
- backend-specific compile artifacts
- per-rank warmup state
- kernel-family manifests

Materialization should be planned as waves or a queue graph with explicit admissibility checks, not just as a flat task list.

---

## The Main Pipeline

The architecture should closely mirror `jello`'s driver pipeline while importing tinygrad's stronger normalization, validation, and phase discipline.

### Proposed sock pipeline

1. Ingest
2. Normalize
3. Discover
4. Resolve
5. Validate
6. Plan
7. Compile
8. Assemble
9. Materialize
10. Verify
11. Diagnose
12. Emit

### 1. Ingest

Input sources may include:

- CLI invocations
- project manifest files
- engine-specific config files
- environment variables
- hardware profile declarations
- deployment presets

The system must accept the messy shape the world already uses.

### 2. Normalize

Convert heterogeneous config into a canonical request.

Examples:

- different naming conventions for the same model format
- different backend toggles that imply the same effect
- engine-specific flags that map to common semantic concepts
- duplicate or conflicting cache settings
- vague guarantee asks like "no runtime JIT" into explicit closure obligations

Output:

- a normalized request object

### 3. Discover

Discover real environment facts:

- GPU architecture
- available driver and runtime stack
- installed backend libraries
- engine version
- engine extension surface
- availability of Triton, TileLang, CUTLASS, DeepGemm, FlashInfer, TensorRT, and other relevant stacks
- local cache directories
- filesystem constraints
- topology facts
- ABI-relevant package and extension fingerprints

This phase should be first-class and explicit.

### 4. Resolve

Resolve semantic choices:

- which engine backend path is valid
- which kernels and extensions are even eligible
- which cache strategy is possible
- which warmup strategies apply
- whether ahead-of-time materialization is supported for the selected path
- which guarantee target is actually achievable

This is the inference equivalent of `jello`'s backend discovery and library resolution.

### 5. Validate

Before execution, validate:

- incompatible architecture and backend combinations
- known crash combinations
- unsupported model and backend pairings
- missing prerequisites
- cache invalidation hazards
- likely runtime JIT hotspots
- closure claims lacking sufficient evidence
- replay artifacts that would be insufficient for deterministic reproduction

This phase is where sock earns trust.

Validation should support levels:

- fast structural validation
- semantic consistency validation
- expensive witness validation
- optional solver-backed contradiction checks

### 6. Plan

Produce immutable `BuildPlan`.

This is the product's IR.

### 7. Compile

Compile or otherwise pre-materialize buildable artifacts:

- engine-side compile steps
- extension build steps
- backend-specific codegen triggers
- kernel-family precompilation where supported

### 8. Assemble

Assemble the artifact graph:

- connect compiled artifacts to their runtime use sites
- finalize cache identity and manifest ownership
- construct materialization waves and dependency edges

### 9. Materialize

Execute the build and startup materialization pipeline:

- warmup passes
- cache generation
- artifact movement
- manifest writing
- topology-scoped placement

This phase must be constrained by the plan.

### 10. Verify

Verification should confirm:

- planned artifacts actually exist
- cache directories contain expected entries
- warmup coverage completed as planned
- runtime observation matches claimed closure
- startup can proceed without additional unplanned compile work where guarantees claim so

### 11. Diagnose

Translate failures into high-signal diagnostics.

Examples:

- "DeepGemm auto-disabled, but warmup path still invoked backend-specific kernels"
- "TileLang kernel family is still lazily shape-specialized under this request profile"
- "This engine and backend combination cannot provide no-runtime-JIT guarantees for these shape ranges"
- "Cache path configured but not persistent across workers"
- "Compiled extension ABI fingerprint does not match the active PyTorch and CUDA stack"

### 12. Emit

Write build artifacts:

- `buildplan.json`
- `diagnostics.json`
- `replay.sh`
- `artifact_manifest.json`
- `verification_report.json`

Potentially:

- `risk_report.json`
- `warmup_manifest.json`
- `backend_resolution.json`
- `rewrite_trace.json`
- `guarantee_evidence.json`

---

## Proposed Product Surface

### CLI

Potential commands:

- `sock plan`
- `sock build`
- `sock verify`
- `sock replay`
- `sock doctor`
- `sock explain`
- `sock diff`
- `sock render`
- `sock trace`

Example flow:

```bash
sock plan --engine vllm --model Qwen/Qwen3.6-27B-FP8 --target blackwell-sm120 --guarantee strict-aot
sock build buildplan.json
sock verify buildplan.json
sock explain buildplan.json
```

### Manifest-driven flow

Project manifest file:

- `.sock.json`
- `sock.toml`

Config hierarchy should mirror `jello`:

1. environment variables
2. project config
3. user config
4. defaults

This is a direct reusable pattern from `jello`.

---

## Reusable jello Patterns

The following should be copied aggressively in spirit and, where useful, in implementation approach.

### Pattern 1: Immutable plan artifact

From `jello`:

- `LinkPlan` is the central truth

For sock:

- `BuildPlan` should be the central truth

### Pattern 2: Explicit orchestration phases

From `jello`:

- parse
- normalize
- discover
- resolve
- plan
- execute
- diagnose
- emit

For sock:

- same architecture, domain-translated and extended into compile, assemble, materialize, and verify

### Pattern 3: Layered config resolution

From `jello` config:

- env > project > user > defaults

For sock:

- keep this exactly

### Pattern 4: Replay artifacts

From `jello`:

- `linkplan.json`
- `linkplan.sh`
- `diagnostics.json`

For sock:

- `buildplan.json`
- `replay.sh`
- `diagnostics.json`
- `artifact_manifest.json`
- `verification_report.json`

### Pattern 5: Confidence-scored fixing

From `jello`:

- auto-fix when safe
- suggest when ambiguous
- fail when correctness is unclear

For sock:

- same model for backend toggles, warmup strategies, cache choices, closure claims, and runtime-JIT risk

### Pattern 6: Explain mode

From `jello`:

- show chosen backend, reasoning, inputs, fixes, command

For sock:

- show backend family, kernel paths, extension paths, cache assumptions, warmup coverage, guarantee evidence, and remaining runtime risks

---

## V1 Scope

V1 should be deliberately narrow in engine breadth but deep in closure quality.

### Recommended V1 target

- engines:
  - `vLLM` only
- hardware:
  - NVIDIA only
- environment:
  - Linux first
- guarantees:
  - deterministic planning
  - artifact emission
  - backend validation
  - warmup planning
  - cache planning
  - artifact closure modeling
  - risk reporting for remaining runtime JIT

### Why `vLLM` only first

The real V1 challenge is not breadth.
It is proving that sock can deeply model one production engine, including:

- backend family selection
- custom kernel and tensor-op surfaces
- extension and ABI constraints
- shape-bounded warmup closure
- replayable and verifiable materialization

If the adapter contract is correct, other engines can slot in later with targeted shims rather than parallel implementations.

### V1 must-have features

1. Engine request normalization
2. Hardware, runtime, and backend discovery
3. Backend, kernel, and extension compatibility validation
4. Warmup coverage graph generation
5. Cache plan generation
6. Immutable `BuildPlan`
7. Artifact graph and manifest generation
8. Replay artifacts
9. High-signal diagnostics
10. Verification report
11. Guarantee envelope reporting

### V1 nice-to-have features

1. BuildPlan diffing
2. Shared cache manifest format
3. Environment doctor mode
4. Preset profiles by hardware family
5. Rewrite trace visualization

---

## Non-Goals For V1

- universal support for every inference engine
- inventing a new tensor compiler
- cluster orchestration
- replacing engine internals
- deep scheduler integration
- auto-optimizing every model architecture
- writing a full runtime

The first win is determinism and legibility with deep `vLLM` closure, not total stack ownership.

---

## Example Guarantees

sock should eventually be able to say things like:

- "This build fully materialized the planned backend artifacts for the declared warmup coverage."
- "No additional compile work is expected for requests within this shape envelope."
- "TileLang remains lazily specialized beyond these sequence-length buckets."
- "This plan requires runtime fallback if request shapes exceed the warmup envelope."
- "DeepGemm was disabled and no DeepGemm warmup path will be executed."
- "The active custom extension set is ABI-compatible with the detected CUDA, PyTorch, and engine revision."
- "This guarantee applies only to tensor-parallel degree 4 on `sm120` devices under the recorded cache identity."

Those are the kinds of statements the user actually needs.

---

## Why This Matters

Today, inference serving startup is often a black box.

The operator wants:

- confidence
- reproducibility
- clear failure modes
- environment portability
- fewer runtime surprises

The engines usually provide:

- partial logs
- partial warmup
- hidden backend decisions
- backend-specific failure surfaces
- opaque kernel and extension dispatch behavior

sock should bridge that gap.

The value is not merely better speed.
The value is turning startup, warmup, custom-kernel selection, and compilation into something that can be reasoned about ahead of time.

---

## Canonical Project Statement

sock is a deterministic build driver for inference engines.

It ingests messy engine, model, hardware, backend, and extension configuration; normalizes it into an immutable `BuildPlan`; resolves backend, kernel, cache, and warmup strategy deterministically; materializes reproducible artifacts; verifies the achieved guarantee envelope; and emits explainable diagnostics and replayable build outputs.

It is the inference analogue of what `jello` is for linking:

- not a replacement for the underlying backend
- not a new engine
- but a semantic orchestration layer that makes a chaotic system legible, reproducible, and reliable

Its implementation should be modular across engines and backends, but V1 should go deep on `vLLM` first so the abstractions are proven against a real production engine rather than only against a shallow common denominator.

---

## Immediate Next Step

The next concrete step should be to define:

- the first version of the canonical `BuildPlan` schema
- the engine adapter contract
- the artifact closure and guarantee-envelope types
- the exact V1 pipeline for:
  - `vLLM`
  - NVIDIA
  - Linux

That schema and adapter boundary are the foundation of the entire project.
