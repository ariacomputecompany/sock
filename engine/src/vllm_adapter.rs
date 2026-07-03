use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use sock_core::{
    AdapterBackendBinding, AdapterBoundary, AdapterCompileRegion, AdapterDiagnostic, AdapterError,
    AdapterHook, AdapterResult, AdapterSurvey, ArtifactPortability, CacheOwnershipSurface,
    CompileAffectingKnob, CompileRegionKind, ConfigInputSource, CoveragePlane, DiagnosticSeverity,
    EffectiveConfigInput, EngineAdapter, EngineAdapterContract, JitRiskLevel,
    PreservedEngineAbstraction, RankDisposition, ResidualRuntimeJitSurface, SourceAnchor,
    SourceEvidence, TargetEngine,
};

use crate::vllm;

pub struct VllmAdapter {
    root: PathBuf,
    source_index: RefCell<SourceIndex>,
}

impl Default for VllmAdapter {
    fn default() -> Self {
        Self::new(vllm::root())
    }
}

impl VllmAdapter {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            source_index: RefCell::new(SourceIndex::new(root.clone())),
            root,
        }
    }

    fn contract(&self) -> EngineAdapterContract {
        EngineAdapterContract {
            hooks: vec![
                AdapterHook::DiscoverEngineSurface,
                AdapterHook::ExtractEffectiveConfig,
                AdapterHook::EnumerateExecutionPaths,
                AdapterHook::EnumerateMaterializableArtifacts,
                AdapterHook::ResolveBackendOptions,
                AdapterHook::BuildWarmupCoverage,
                AdapterHook::ObserveRuntimeMaterialization,
                AdapterHook::VerifyClosureClaims,
                AdapterHook::RenderExplain,
            ],
            boundaries: vec![
                AdapterBoundary::CompileGraphBoundary,
                AdapterBoundary::CompileRegionBoundary,
                AdapterBoundary::CustomOpBoundary,
                AdapterBoundary::CacheOwnershipBoundary,
                AdapterBoundary::CUDAGraphBoundary,
                AdapterBoundary::WarmupBoundary,
                AdapterBoundary::TopologyBoundary,
            ],
            guarantee_limitations: vec![
                "Closure claims remain topology-bounded because full CUDA graphs and some backend warmups are invalidated by context-parallel configuration.".to_owned(),
                "Residual runtime specialization must be tracked as evidence, because vLLM still contains explicit warmup and fallback paths for JIT-sensitive regions.".to_owned(),
            ],
        }
    }

    fn config_inputs(&self) -> AdapterResult<Vec<EffectiveConfigInput>> {
        Ok(vec![
            self.config_input(
                "--cudagraph-capture-sizes",
                ConfigInputSource::CliFlag,
                "Shapes the explicit CUDA graph capture envelope for mixed and decode execution paths.",
                true,
                "vllm/vllm/engine/arg_utils.py",
                &["--cudagraph-capture-sizes", "CompilationConfig"],
            )?,
            self.config_input(
                "--max-cudagraph-capture-size",
                ConfigInputSource::CliFlag,
                "Caps full-graph coverage and directly changes the decode capture frontier.",
                true,
                "vllm/vllm/engine/arg_utils.py",
                &["--max-cudagraph-capture-size", "CompilationConfig"],
            )?,
            self.config_input(
                "--enable-flashinfer-autotune",
                ConfigInputSource::CliFlag,
                "Turns backend tactic materialization into an explicit startup contract instead of a hidden runtime side effect.",
                true,
                "vllm/vllm/engine/arg_utils.py",
                &["--enable-flashinfer-autotune", "KernelConfig"],
            )?,
            self.config_input(
                "CompilationConfig",
                ConfigInputSource::PythonApi,
                "Python-side config owns compile mode, cudagraph mode, guard policy, and cache semantics.",
                true,
                "vllm/vllm/config/__init__.py",
                &["CompilationConfig", "CUDAGraphMode"],
            )?,
            self.config_input(
                "VLLM_DISABLE_COMPILE_CACHE",
                ConfigInputSource::EnvironmentVariable,
                "Disables compile-cache reuse and therefore changes artifact admissibility.",
                true,
                "vllm/vllm/envs.py",
                &["VLLM_DISABLE_COMPILE_CACHE"],
            )?,
            self.config_input(
                "VLLM_COMPILE_CACHE_SAVE_FORMAT",
                ConfigInputSource::EnvironmentVariable,
                "Changes compile cache persistence layout and multiprocess safety.",
                true,
                "vllm/vllm/envs.py",
                &["VLLM_COMPILE_CACHE_SAVE_FORMAT"],
            )?,
            self.config_input(
                "VLLM_HAS_FLASHINFER_CUBIN",
                ConfigInputSource::EnvironmentVariable,
                "Signals whether prebuilt FlashInfer cubins are present and admissible for reuse.",
                true,
                "vllm/vllm/envs.py",
                &["VLLM_HAS_FLASHINFER_CUBIN"],
            )?,
            self.config_input(
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                ConfigInputSource::EnvironmentVariable,
                "Relocates tactic cache ownership and affects startup reuse behavior across launches.",
                true,
                "vllm/vllm/envs.py",
                &["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"],
            )?,
            self.config_input(
                "current_platform",
                ConfigInputSource::RuntimeState,
                "Platform/runtime checks gate backend legality, full CUDA graph admissibility, and extension behavior.",
                true,
                "vllm/vllm/platforms/cuda.py",
                &["in_wsl()", "cudagraph_mode"],
            )?,
        ])
    }

    fn compile_knobs(&self) -> AdapterResult<Vec<CompileAffectingKnob>> {
        Ok(vec![
            self.knob(
                "Compilation mode",
                "compile-pipeline",
                "vLLM distinguishes stock torch.compile, single-trace Dynamo, and its own Inductor-backed compile mode.",
                "vllm/vllm/config/compilation.py",
                &["class CompilationMode", "VLLM_COMPILE = 3"],
            )?,
            self.knob(
                "CUDA graph mode",
                "cudagraph-policy",
                "NONE, PIECEWISE, FULL, and FULL_AND_PIECEWISE are separate planning modes and cannot be flattened without losing semantics.",
                "vllm/vllm/config/compilation.py",
                &["class CUDAGraphMode", "FULL_AND_PIECEWISE"],
            )?,
            self.knob(
                "Optimization presets",
                "preset-policy",
                "O0/O1/O2 presets materially change cudagraph behavior and backend autotune policy.",
                "vllm/vllm/config/vllm.py",
                &[
                    "OPTIMIZATION_LEVEL_00",
                    "OPTIMIZATION_LEVEL_01",
                    "OPTIMIZATION_LEVEL_02",
                ],
            )?,
            self.knob(
                "Dynamic-shape guard policy",
                "guard-policy",
                "Guard evaluation and guard dropping change what runtime specialization leakage means.",
                "vllm/vllm/config/compilation.py",
                &["evaluate_guards: bool = False", "DynamicShapesType"],
            )?,
            self.knob(
                "Custom-op dispatch",
                "custom-op-boundary",
                "Custom ops can be enabled, disabled, or compiled natively and therefore affect both legality and identity.",
                "vllm/vllm/model_executor/custom_op.py",
                &[
                    "class CustomOp(nn.Module):",
                    "compilation_config.enabled_custom_ops.update",
                ],
            )?,
            self.knob(
                "FlashInfer autotune",
                "backend-autotune",
                "FlashInfer startup may benchmark multiple implementations and broadcast tactic caches across ranks.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &["def flashinfer_autotune(", "Broadcast autotune cache from rank 0"],
            )?,
        ])
    }

    fn preserved_abstractions(&self) -> AdapterResult<Vec<PreservedEngineAbstraction>> {
        Ok(vec![
            self.abstraction(
                "Graph and region boundaries",
                "Compile planning should preserve explicit graph-partition and CUDA-graph boundaries because vLLM models them as distinct execution surfaces.",
                "vllm/vllm/config/compilation.py",
                &["requires_piecewise_compilation", "has_full_cudagraphs"],
            )?,
            self.abstraction(
                "Custom-op boundaries",
                "Custom ops are a native vLLM boundary and should remain visible instead of being flattened into anonymous kernels.",
                "vllm/vllm/model_executor/custom_op.py",
                &[
                    "class CustomOp(nn.Module):",
                    "Dispatches the forward method to the appropriate backend.",
                ],
            )?,
            self.abstraction(
                "Cache ownership boundaries",
                "Compile-cache ownership and cache-key shaping are explicit in vLLM env configuration and must stay separate from warmup evidence.",
                "vllm/vllm/envs.py",
                &["compile_factors()", "VLLM_COMPILE_CACHE_SAVE_FORMAT"],
            )?,
            self.abstraction(
                "Layer identity in static forward context",
                "Layer identity already exists in the compile-time forward context and should anchor region extraction.",
                "vllm/vllm/config/vllm.py",
                &[
                    "def get_layers_from_vllm_config(",
                    "forward_context = vllm_config.compilation_config.static_forward_context",
                ],
            )?,
        ])
    }

    fn compile_regions(&self) -> AdapterResult<Vec<AdapterCompileRegion>> {
        Ok(vec![
            self.region(
                "repeated_transformer_block_pattern",
                "transformer_block_body",
                CompileRegionKind::RepeatedTransformerBlockBody,
                AdapterBackendBinding::Primary,
                true,
                true,
                vec![
                    "Repeated layer patterns are already explicit in KV cache grouping.".to_owned(),
                    "Static forward-context naming can stabilize region identity.".to_owned(),
                ],
                "Repeated layer-pattern structure is strong evidence that transformer-body regions can be compiled once and reused across placements.",
                "repeated_transformer_block_pattern",
                vec![CoveragePlane::Correctness, CoveragePlane::Performance],
                ArtifactPortability::GpuArchitectureFamilyPortable,
                RankDisposition::Shared,
                false,
                "compile-cache",
                "transformer_block_body",
                "vllm/vllm/v1/core/kv_cache_utils.py",
                &["The layers in the models are repeated with some patterns"],
            )?,
            self.region(
                "decode_micrograph",
                "decode_attention",
                CompileRegionKind::DecodeMicrograph,
                AdapterBackendBinding::Fixed(sock_core::BackendFamily::CudaGraphs),
                true,
                true,
                vec![
                    "Decode batch shape is bounded by cudagraph capture limits.".to_owned(),
                    "Speculative decode deterministically expands the decode envelope.".to_owned(),
                ],
                "Decode metadata builders preallocate against a bounded capture size, making decode a natural regional compile target.",
                "decode_micrograph",
                vec![
                    CoveragePlane::Correctness,
                    CoveragePlane::Performance,
                    CoveragePlane::CudaGraph,
                ],
                ArtifactPortability::TopologyScoped,
                RankDisposition::RankLocal,
                true,
                "cuda-graph-cache",
                "decode_attention",
                "vllm/vllm/v1/attention/backends/gdn_attn.py",
                &["self.decode_cudagraph_max_bs", "self.use_full_cuda_graph"],
            )?,
            self.region(
                "prefill_micrograph",
                "prefill_attention",
                CompileRegionKind::PrefillMicrograph,
                AdapterBackendBinding::Primary,
                true,
                true,
                vec![
                    "Prefill metadata kernels are warmed separately from decode.".to_owned(),
                    "Chunked prefill metadata already exists as a distinct kernel surface.".to_owned(),
                ],
                "Sparse MLA warmup paths show that prefill has its own specialization surface and should remain a first-class compile region.",
                "prefill_micrograph",
                vec![CoveragePlane::Correctness, CoveragePlane::Performance],
                ArtifactPortability::GpuArchitectureFamilyPortable,
                RankDisposition::Shared,
                false,
                "compile-cache",
                "prefill_attention",
                "vllm/vllm/model_executor/warmup/sparse_mla_triton_warmup.py",
                &[
                    "_warm_sparse_swa_prefill_metadata_kernel",
                    "_warm_prefill_chunk_metadata_kernel",
                ],
            )?,
            self.region(
                "attention_kv_update_boundary",
                "kv_cache_update",
                CompileRegionKind::AttentionKvBoundary,
                AdapterBackendBinding::Primary,
                true,
                true,
                vec![
                    "Mixed prefill+decode attention warmup is explicit.".to_owned(),
                    "KV update and attention setup are already surfaced as separate warmup obligations.".to_owned(),
                ],
                "Mixed-batch attention warmup shows a real execution seam between attention/KV setup and the broader model path.",
                "attention_kv_update_boundary",
                vec![
                    CoveragePlane::Correctness,
                    CoveragePlane::BackendSpecialization,
                ],
                ArtifactPortability::GpuArchitectureFamilyPortable,
                RankDisposition::Shared,
                true,
                "flashinfer-autotune-cache",
                "kv_cache_update",
                "vllm/vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py",
                &[
                    "Warm DSv4 sparse-MLA mixed prefill+decode attention",
                    "run_mixed_prefill_decode_warmup",
                ],
            )?,
            self.region(
                "moe_specialty_path",
                "moe_specialty_path",
                CompileRegionKind::MoeSpecialtyPath,
                AdapterBackendBinding::Fixed(sock_core::BackendFamily::AotInductor),
                true,
                true,
                vec![
                    "MoE custom ops can trigger special fallback behavior in Inductor.".to_owned(),
                    "Deep MoE/TP graphs are already called out as compile-pathological.".to_owned(),
                ],
                "Deep MoE and TP graphs are handled as a special compile-risk class already, which makes them a distinct specialty region.",
                "moe_specialty_path",
                vec![CoveragePlane::BackendSpecialization],
                ArtifactPortability::TopologyScoped,
                RankDisposition::Shared,
                true,
                "compile-cache",
                "moe_specialty_path",
                "vllm/vllm/env_override.py",
                &["deep MoE/TP graphs", "_VLLM_FALLBACK_NAMESPACE_PREFIXES"],
            )?,
        ])
    }

    fn cache_ownership_surfaces(&self) -> AdapterResult<Vec<CacheOwnershipSurface>> {
        Ok(vec![
            self.cache_surface(
                "compile-cache",
                vec![
                    "transformer_block_body".to_owned(),
                    "prefill_attention".to_owned(),
                    "moe_specialty_path".to_owned(),
                ],
                vec![
                    "VLLM_DISABLE_COMPILE_CACHE".to_owned(),
                    "VLLM_COMPILE_CACHE_SAVE_FORMAT".to_owned(),
                    "CompilationConfig".to_owned(),
                ],
                ArtifactPortability::AbiClusterPortable,
                RankDisposition::Shared,
                false,
                "vLLM compile-cache ownership is explicit and must remain separate from warmup proofs.",
                "vllm/vllm/envs.py",
                &["compile_factors()", "VLLM_COMPILE_CACHE_SAVE_FORMAT"],
            )?,
            self.cache_surface(
                "flashinfer-autotune-cache",
                vec!["kv_cache_update".to_owned()],
                vec![
                    "--enable-flashinfer-autotune".to_owned(),
                    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR".to_owned(),
                    "VLLM_HAS_FLASHINFER_CUBIN".to_owned(),
                ],
                ArtifactPortability::TopologyScoped,
                RankDisposition::Shared,
                true,
                "FlashInfer tactic ownership is rank-asymmetric and topology-scoped because rank 0 tunes and followers consume broadcast caches.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &["Tuning is performed only on rank 0.", "Broadcast autotune cache from rank 0"],
            )?,
            self.cache_surface(
                "cuda-graph-cache",
                vec!["decode_attention".to_owned()],
                vec![
                    "--cudagraph-capture-sizes".to_owned(),
                    "--max-cudagraph-capture-size".to_owned(),
                    "current_platform".to_owned(),
                ],
                ArtifactPortability::TopologyScoped,
                RankDisposition::RankLocal,
                true,
                "CUDA-graph capture ownership is rank-local and topology-scoped because capture legality depends on the exact runtime topology.",
                "vllm/vllm/config/compilation.py",
                &["class CUDAGraphMode", "has_full_cudagraphs"],
            )?,
        ])
    }

    fn residual_jit_surfaces(&self) -> AdapterResult<Vec<ResidualRuntimeJitSurface>> {
        Ok(vec![
            self.jit_surface(
                "flashinfer_tactic_autotune",
                JitRiskLevel::High,
                "Enabled when FlashInfer is present on Hopper or Blackwell and autotune is not disabled.",
                "flashinfer",
                "Leader/follower distributed startup where rank 0 benchmarks and followers consume broadcast tactic caches.",
                "Warmup only covers the dummy-run envelope, so untouched tactics can still appear later.",
                "Persist the autotune cache as a first-class artifact and verify replay against the exact cache file.",
                vec![
                    "--enable-flashinfer-autotune".to_owned(),
                    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR".to_owned(),
                    "VLLM_HAS_FLASHINFER_CUBIN".to_owned(),
                ],
                vec!["kv_cache_update".to_owned(), "prefill_attention".to_owned()],
                vec!["kv_cache_update".to_owned()],
                vec!["kv_cache_update".to_owned()],
                true,
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &["FlashInfer autotune for Hopper", "Tuning is performed only on rank 0."],
            )?,
            self.jit_surface(
                "triton_attention_warmup_gap",
                JitRiskLevel::Medium,
                "Appears when the dummy mixed batch does not match the production prefill/decode combination.",
                "triton",
                "Single-rank and distributed attention backends that rely on warmup dummy runs.",
                "Warmup is intentionally selective and some hybrid attention backends are still excluded by current capture-path limits.",
                "Promote region-scoped warmup obligations into the canonical plan instead of relying on one generic dummy run.",
                vec![
                    "--cudagraph-capture-sizes".to_owned(),
                    "--max-cudagraph-capture-size".to_owned(),
                    "CompilationConfig".to_owned(),
                ],
                vec!["decode_attention".to_owned(), "prefill_attention".to_owned()],
                vec!["decode_attention".to_owned(), "prefill_attention".to_owned()],
                vec!["decode_attention".to_owned(), "prefill_attention".to_owned()],
                true,
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &[
                    "This should be `any` instead of `all`",
                    "Warmup with mixed batch containing both prefill and decode tokens",
                ],
            )?,
            self.jit_surface(
                "inductor_custom_op_fallback",
                JitRiskLevel::High,
                "Triggered by custom ops without lowerings or decompositions in deep MoE/TP graphs.",
                "inductor",
                "Topology-sensitive TP and MoE graphs where fallback logging becomes compile-pathological.",
                "The first encounter of a custom op can still be expensive enough to require vLLM-side patching.",
                "Track fallback namespace coverage as explicit guarantee evidence instead of assuming it is harmless.",
                vec!["CompilationConfig".to_owned(), "current_platform".to_owned()],
                vec!["moe_specialty_path".to_owned()],
                vec!["moe_specialty_path".to_owned()],
                vec!["moe_specialty_path".to_owned()],
                true,
                "vllm/vllm/env_override.py",
                &[
                    "When Inductor encounters a custom op without a registered lowering",
                    "effectively hanging compilation",
                ],
            )?,
            self.jit_surface(
                "guard_dropped_dynamic_shapes",
                JitRiskLevel::Medium,
                "Occurs whenever non-stock compile modes run with guard dropping enabled.",
                "torch.compile",
                "All topologies; impact grows when real traffic escapes the planned shape envelope.",
                "Dropped guards suppress recompilation triggers, so runtime specialization leakage must be detected through witnesses rather than assumed absent.",
                "Treat guard policy as a guarantee-plane input and require contradiction evidence when evaluate-guards mode is requested.",
                vec!["CompilationConfig".to_owned()],
                vec![
                    "transformer_block_body".to_owned(),
                    "prefill_attention".to_owned(),
                    "decode_attention".to_owned(),
                ],
                vec![
                    "transformer_block_body".to_owned(),
                    "prefill_attention".to_owned(),
                    "decode_attention".to_owned(),
                ],
                vec![
                    "transformer_block_body".to_owned(),
                    "prefill_attention".to_owned(),
                    "decode_attention".to_owned(),
                ],
                false,
                "vllm/vllm/compilation/wrapper.py",
                &[
                    "it ensures that all guards are dropped",
                    "Dynamo should never be traced again after that",
                ],
            )?,
        ])
    }

    fn diagnostics(&self) -> AdapterResult<Vec<AdapterDiagnostic>> {
        Ok(vec![
            self.diagnostic(
                DiagnosticSeverity::Risk,
                "Guard-dropping assumptions",
                "vLLM deliberately drops Dynamo guards outside stock torch.compile mode, so sock must surface this as an explicit assumption in the guarantee model.",
                "vllm/vllm/compilation/wrapper.py",
                &["it ensures that all guards are dropped", "skip_all_guards_unsafe"],
            )?,
            self.diagnostic(
                DiagnosticSeverity::Warning,
                "Unsupported full CUDA graph paths",
                "Full CUDA graphs are downgraded to piecewise mode when decode or prefill context parallelism is enabled.",
                "vllm/vllm/platforms/rocm.py",
                &[
                    "incompatible with full CUDA graphs",
                    "Overriding cudagraph_mode to PIECEWISE.",
                ],
            )?,
            self.diagnostic(
                DiagnosticSeverity::Info,
                "Piecewise versus full CUDA graph tradeoff",
                "vLLM’s optimization presets already distinguish piecewise and full graph policies, so sock should preserve both planes in planning.",
                "vllm/vllm/config/vllm.py",
                &["CUDAGraphMode.PIECEWISE", "CUDAGraphMode.FULL_AND_PIECEWISE"],
            )?,
            self.diagnostic(
                DiagnosticSeverity::Warning,
                "Missing prebuilt backend assets",
                "FlashInfer cubins and autotune caches are first-class reuse inputs; when missing, the plan should explain the extra startup work directly.",
                "vllm/vllm/envs.py",
                &["VLLM_HAS_FLASHINFER_CUBIN", "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"],
            )?,
            self.diagnostic(
                DiagnosticSeverity::Risk,
                "Residual Triton JIT risk",
                "Warmup code exists specifically to avoid JIT during execution, which means uncovered regions remain a real production risk.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &[
                    "This is useful specifically for JIT'ed kernels",
                    "happen during model execution.",
                ],
            )?,
            self.diagnostic(
                DiagnosticSeverity::Warning,
                "Topology-specific startup hazard",
                "Distributed startup is intentionally asymmetric because rank 0 may materialize tactics and broadcast them to followers.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &[
                    "Tuning is performed only on rank 0.",
                    "Broadcast autotune cache from rank 0 to all other ranks",
                ],
            )?,
        ])
    }

    fn config_input(
        &self,
        name: &str,
        source: ConfigInputSource,
        compile_relevance: &str,
        identity_affecting: bool,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<EffectiveConfigInput> {
        Ok(EffectiveConfigInput {
            name: name.to_owned(),
            source,
            compile_relevance: compile_relevance.to_owned(),
            identity_affecting,
            evidence: self.evidence(file, compile_relevance, anchor_patterns)?,
        })
    }

    fn knob(
        &self,
        name: &str,
        category: &str,
        description: &str,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<CompileAffectingKnob> {
        Ok(CompileAffectingKnob {
            name: name.to_owned(),
            category: category.to_owned(),
            description: description.to_owned(),
            evidence: self.evidence(file, description, anchor_patterns)?,
        })
    }

    fn abstraction(
        &self,
        name: &str,
        description: &str,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<PreservedEngineAbstraction> {
        Ok(PreservedEngineAbstraction {
            name: name.to_owned(),
            description: description.to_owned(),
            evidence: self.evidence(file, description, anchor_patterns)?,
        })
    }

    fn region(
        &self,
        name: &str,
        canonical_name: &str,
        kind: CompileRegionKind,
        backend_binding: AdapterBackendBinding,
        repeated: bool,
        regional_compile_candidate: bool,
        boundaries: Vec<String>,
        rationale: &str,
        invalidation_domain: &str,
        shape_planes: Vec<CoveragePlane>,
        artifact_portability: ArtifactPortability,
        rank_disposition: RankDisposition,
        topology_sensitive: bool,
        cache_namespace: &str,
        warmup_scope: &str,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<AdapterCompileRegion> {
        Ok(AdapterCompileRegion {
            name: name.to_owned(),
            canonical_name: canonical_name.to_owned(),
            kind,
            backend_binding,
            repeated,
            regional_compile_candidate,
            boundaries,
            rationale: rationale.to_owned(),
            invalidation_domain: invalidation_domain.to_owned(),
            shape_planes,
            artifact_portability,
            rank_disposition,
            topology_sensitive,
            cache_namespace: cache_namespace.to_owned(),
            warmup_scope: warmup_scope.to_owned(),
            evidence: self.evidence(file, rationale, anchor_patterns)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn jit_surface(
        &self,
        name: &str,
        risk: JitRiskLevel,
        trigger_shape_or_config: &str,
        backend_family: &str,
        topology_context: &str,
        warmup_gap: &str,
        mitigation: &str,
        trigger_inputs: Vec<String>,
        affected_regions: Vec<String>,
        required_artifacts: Vec<String>,
        required_warmup_scopes: Vec<String>,
        topology_sensitive: bool,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<ResidualRuntimeJitSurface> {
        Ok(ResidualRuntimeJitSurface {
            name: name.to_owned(),
            risk,
            trigger_shape_or_config: trigger_shape_or_config.to_owned(),
            backend_family: backend_family.to_owned(),
            topology_context: topology_context.to_owned(),
            warmup_gap: warmup_gap.to_owned(),
            mitigation: mitigation.to_owned(),
            trigger_inputs,
            affected_regions,
            required_artifacts,
            required_warmup_scopes,
            topology_sensitive,
            evidence: self.evidence(file, mitigation, anchor_patterns)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn cache_surface(
        &self,
        name: &str,
        artifact_scopes: Vec<String>,
        ownership_inputs: Vec<String>,
        portability: ArtifactPortability,
        rank_disposition: RankDisposition,
        topology_sensitive: bool,
        rationale: &str,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<CacheOwnershipSurface> {
        Ok(CacheOwnershipSurface {
            name: name.to_owned(),
            artifact_scopes,
            ownership_inputs,
            portability,
            rank_disposition,
            topology_sensitive,
            rationale: rationale.to_owned(),
            evidence: self.evidence(file, rationale, anchor_patterns)?,
        })
    }

    fn diagnostic(
        &self,
        severity: DiagnosticSeverity,
        title: &str,
        message: &str,
        file: &'static str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<AdapterDiagnostic> {
        Ok(AdapterDiagnostic {
            severity,
            title: title.to_owned(),
            message: message.to_owned(),
            evidence: self.evidence(file, message, anchor_patterns)?,
        })
    }

    fn evidence(
        &self,
        file: &'static str,
        summary: &str,
        anchor_patterns: &[&'static str],
    ) -> AdapterResult<SourceEvidence> {
        let mut index = self.source_index.borrow_mut();
        let anchors = anchor_patterns
            .iter()
            .map(|pattern| index.anchor(file, pattern))
            .collect::<AdapterResult<Vec<_>>>()?;

        Ok(SourceEvidence {
            summary: summary.to_owned(),
            anchors,
        })
    }
}

impl EngineAdapter for VllmAdapter {
    fn target(&self) -> TargetEngine {
        TargetEngine::Vllm
    }

    fn survey(&self) -> AdapterResult<AdapterSurvey> {
        Ok(AdapterSurvey {
            engine: TargetEngine::Vllm,
            engine_root: self.root.display().to_string(),
            engine_revision: vllm::revision().to_owned(),
            contract: self.contract(),
            config_inputs: self.config_inputs()?,
            compile_knobs: self.compile_knobs()?,
            preserved_abstractions: self.preserved_abstractions()?,
            compile_regions: self.compile_regions()?,
            cache_ownership_surfaces: self.cache_ownership_surfaces()?,
            residual_jit_surfaces: self.residual_jit_surfaces()?,
            diagnostics: self.diagnostics()?,
        })
    }

    fn render_explain(&self, survey: &AdapterSurvey) -> String {
        let mut out = String::new();
        out.push_str("sock vLLM adapter survey\n");
        out.push_str(&format!("revision: {}\n", survey.engine_revision));
        out.push_str(&format!("config inputs: {}\n", survey.config_inputs.len()));
        out.push_str(&format!(
            "compile regions: {}\n",
            survey.compile_regions.len()
        ));
        out.push_str(&format!(
            "cache ownership surfaces: {}\n",
            survey.cache_ownership_surfaces.len()
        ));
        out.push_str(&format!(
            "residual jit surfaces: {}\n",
            survey.residual_jit_surfaces.len()
        ));
        out.push_str("regions:\n");
        for region in &survey.compile_regions {
            out.push_str(&format!(
                "- {} -> {} ({:?}) candidate={} repeated={} cache={} topology_sensitive={}\n",
                region.name,
                region.canonical_name,
                region.kind,
                region.regional_compile_candidate,
                region.repeated,
                region.cache_namespace,
                region.topology_sensitive
            ));
        }
        out.push_str("cache surfaces:\n");
        for surface in &survey.cache_ownership_surfaces {
            out.push_str(&format!(
                "- {} scopes={} portability={:?} rank_disposition={:?}\n",
                surface.name,
                surface.artifact_scopes.join(","),
                surface.portability,
                surface.rank_disposition
            ));
        }
        out.push_str("diagnostics:\n");
        for diagnostic in &survey.diagnostics {
            out.push_str(&format!(
                "- {:?}: {} - {}\n",
                diagnostic.severity, diagnostic.title, diagnostic.message
            ));
        }
        out
    }
}

struct SourceIndex {
    root: PathBuf,
    cache: HashMap<&'static str, String>,
}

impl SourceIndex {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            cache: HashMap::new(),
        }
    }

    fn anchor(&mut self, file: &'static str, pattern: &'static str) -> AdapterResult<SourceAnchor> {
        let content = self.content(file)?;
        let line = content
            .lines()
            .position(|line| line.contains(pattern))
            .map(|idx| idx + 1)
            .ok_or_else(|| AdapterError::MissingPattern {
                file: file.to_owned(),
                pattern: pattern.to_owned(),
            })?;

        Ok(SourceAnchor {
            file: file.to_owned(),
            line,
        })
    }

    fn content(&mut self, file: &'static str) -> AdapterResult<&str> {
        if !self.cache.contains_key(file) {
            let path = self.root.join(file.strip_prefix("vllm/").unwrap_or(file));
            let content = fs::read_to_string(&path).map_err(|source| AdapterError::ReadSource {
                file: path.display().to_string(),
                source,
            })?;
            self.cache.insert(file, content);
        }

        Ok(self.cache.get(file).expect("cached content must exist"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survey_anchors_resolve_to_real_source_lines() {
        let adapter = VllmAdapter::default();
        let survey = adapter.survey().expect("survey should build");

        assert_eq!(survey.engine, TargetEngine::Vllm);
        assert!(!survey.config_inputs.is_empty());
        assert!(!survey.compile_regions.is_empty());
        assert!(!survey.cache_ownership_surfaces.is_empty());
        assert!(!survey.residual_jit_surfaces.is_empty());

        for evidence in survey
            .config_inputs
            .iter()
            .map(|entry| &entry.evidence)
            .chain(survey.compile_knobs.iter().map(|entry| &entry.evidence))
            .chain(
                survey
                    .preserved_abstractions
                    .iter()
                    .map(|entry| &entry.evidence),
            )
            .chain(survey.compile_regions.iter().map(|entry| &entry.evidence))
            .chain(
                survey
                    .cache_ownership_surfaces
                    .iter()
                    .map(|entry| &entry.evidence),
            )
            .chain(
                survey
                    .residual_jit_surfaces
                    .iter()
                    .map(|entry| &entry.evidence),
            )
            .chain(survey.diagnostics.iter().map(|entry| &entry.evidence))
        {
            assert!(!evidence.anchors.is_empty());
            for anchor in &evidence.anchors {
                let path =
                    vllm::root().join(anchor.file.strip_prefix("vllm/").unwrap_or(&anchor.file));
                let content = fs::read_to_string(path).expect("source file should exist");
                assert!(anchor.line >= 1);
                assert!(anchor.line <= content.lines().count());
            }
        }
    }
}
