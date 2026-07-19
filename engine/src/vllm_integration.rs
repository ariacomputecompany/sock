use std::collections::BTreeSet;

use sock_core::{
    AdapterError, AdapterSurvey, ArtifactRequirement, CacheOwnershipSurface, CompileRegion,
    IntegrationScopeKind, SourceAnchor, SourceEvidence, VllmCallableTarget,
    VllmIntegrationDocument, VllmIntegrationSurface, VllmIsolationContract,
    VllmIsolationDisposition, VllmReplayRoot, VllmReplayRootKind,
};
use thiserror::Error;

use crate::{BuildScope, PlanningOutcome, vllm};

#[derive(Debug, Error)]
pub enum VllmIntegrationError {
    #[error("adapter survey failed: {0}")]
    Adapter(#[from] AdapterError),
    #[error("missing vLLM integration mapping for scope {scope}")]
    MissingSurface { scope: String },
    #[error(
        "scoped subset build is not semantically valid for {surface_id}: requires {required_context}; blockers: {blockers}"
    )]
    NonIsolatableSubset {
        surface_id: String,
        required_context: String,
        blockers: String,
    },
}

pub fn build_vllm_integration_document(
    outcome: &PlanningOutcome,
) -> Result<VllmIntegrationDocument, VllmIntegrationError> {
    let survey = &outcome.adapter_survey;
    let mut surface_ids = BTreeSet::new();
    let mut surfaces = Vec::new();

    for region in &outcome.plan.compile_regions {
        let surface = integration_surface_for_region(region, survey)?;
        if surface_ids.insert(surface.id.clone()) {
            surfaces.push(surface);
        }
    }

    for cache_surface in relevant_cache_surfaces(&outcome.plan.artifact_requirements, &survey) {
        let surface = integration_surface_for_cache(cache_surface, survey)?;
        if surface_ids.insert(surface.id.clone()) {
            surfaces.push(surface);
        }
    }

    surfaces.sort_by(|left, right| left.id.cmp(&right.id));
    let mut replay_roots = surfaces
        .iter()
        .map(|surface| replay_root_for_surface(outcome, surface))
        .collect::<Vec<_>>();
    replay_roots.sort_by(|left, right| left.id.cmp(&right.id));

    Ok(VllmIntegrationDocument {
        schema_version: sock_core::SchemaVersion::current(),
        plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
        engine_root: vllm::root().display().to_string(),
        engine_revision: vllm::revision().to_owned(),
        surfaces,
        replay_roots,
    })
}

pub fn validate_scoped_vllm_subset(
    scope: &BuildScope,
    integration: &VllmIntegrationDocument,
) -> Result<(), VllmIntegrationError> {
    if !scope.has_subset_selectors() {
        return Ok(());
    }

    for surface in integration
        .surfaces
        .iter()
        .filter(|surface| surface.scope_kind == IntegrationScopeKind::CompileRegion)
    {
        if surface.isolation.subset_build_valid {
            continue;
        }
        return Err(VllmIntegrationError::NonIsolatableSubset {
            surface_id: surface.id.clone(),
            required_context: surface.isolation.required_context.join(", "),
            blockers: surface.isolation.blockers.join("; "),
        });
    }

    Ok(())
}

fn integration_surface_for_region(
    region: &CompileRegion,
    survey: &AdapterSurvey,
) -> Result<VllmIntegrationSurface, VllmIntegrationError> {
    let integrated = match region.name.as_str() {
        "transformer_block_body" => VllmIntegrationSurface {
            id: format!("compile-region:{}", region.name),
            scope_kind: IntegrationScopeKind::CompileRegion,
            scope_name: region.name.clone(),
            backend: Some(region.family),
            cache_namespace: Some("compile-cache".to_owned()),
            warmup_scope: Some("transformer_block_body".to_owned()),
            rationale: "Compiled transformer-body partitions should follow vLLM's own piecewise compilation and standalone artifact storage path.".to_owned(),
            preserved_inputs: vec![
                "CompilationConfig".to_owned(),
                "VLLM_DISABLE_COMPILE_CACHE".to_owned(),
                "VLLM_COMPILE_CACHE_SAVE_FORMAT".to_owned(),
            ],
            preserved_abstractions: vec![
                "Graph and region boundaries".to_owned(),
                "Cache ownership boundaries".to_owned(),
                "Layer identity in static forward context".to_owned(),
            ],
            isolation: isolation(
                VllmIsolationDisposition::ContextBound,
                true,
                false,
                &["PiecewiseBackend context"],
                &[],
                "Subset compilation for transformer blocks is real, but the seam is only callable through vLLM's piecewise backend context.",
                "vllm/vllm/compilation/piecewise_backend.py",
                &["class PiecewiseBackend", "def compile_all_ranges(self) -> None:"],
            )?,
            primary: callable(
                "vllm.compilation.piecewise_backend",
                "PiecewiseBackend.compile_all_ranges",
                "Compile every selected range through vLLM's piecewise backend instead of flattening region compilation into a sock-owned pipeline.",
                "vllm/vllm/compilation/piecewise_backend.py",
                &["def compile_all_ranges(self) -> None:"],
            )?,
            auxiliary: vec![
                callable(
                    "vllm.compilation.caching",
                    "StandaloneCompiledArtifacts.insert",
                    "Deduplicate standalone compiled artifacts exactly the way vLLM stores them.",
                    "vllm/vllm/compilation/caching.py",
                    &["class StandaloneCompiledArtifacts:", "def insert(self, submod_name: str, shape: str, entry: bytes) -> None:"],
                )?,
                callable(
                    "vllm.compilation.caching",
                    "build_aot_compile_plan",
                    "Shape compile-cache identity with vLLM's own normalized AOT compile plan.",
                    "vllm/vllm/compilation/caching.py",
                    &["def build_aot_compile_plan(", "\"canonical_aot_plan_id\""],
                )?,
            ],
        },
        "prefill_attention" => VllmIntegrationSurface {
            id: format!("compile-region:{}", region.name),
            scope_kind: IntegrationScopeKind::CompileRegion,
            scope_name: region.name.clone(),
            backend: Some(region.family),
            cache_namespace: Some("compile-cache".to_owned()),
            warmup_scope: Some("prefill_attention".to_owned()),
            rationale: "Prefill specialization should preserve vLLM's sparse-MLA Triton warmup path rather than substituting a sock-owned prefill kernel path.".to_owned(),
            preserved_inputs: vec![
                "CompilationConfig".to_owned(),
                "--cudagraph-capture-sizes".to_owned(),
            ],
            preserved_abstractions: vec![
                "Graph and region boundaries".to_owned(),
                "Custom-op boundaries".to_owned(),
            ],
            isolation: isolation(
                VllmIsolationDisposition::ContextBound,
                true,
                false,
                &["Worker context"],
                &[],
                "Prefill warmup remains subset-build valid because vLLM exposes a worker-scoped warmup seam without forcing decode-owned mixed-batch capture.",
                "vllm/vllm/model_executor/warmup/sparse_mla_triton_warmup.py",
                &["def sparse_mla_triton_warmup_if_needed(worker: \"Worker\") -> None:"],
            )?,
            primary: callable(
                "vllm.model_executor.warmup.sparse_mla_triton_warmup",
                "sparse_mla_triton_warmup_if_needed",
                "Use vLLM's native sparse-MLA Triton warmup orchestrator for prefill-owned metadata kernels.",
                "vllm/vllm/model_executor/warmup/sparse_mla_triton_warmup.py",
                &["def sparse_mla_triton_warmup_if_needed(worker: \"Worker\") -> None:"],
            )?,
            auxiliary: vec![
                callable(
                    "vllm.model_executor.warmup.sparse_mla_triton_warmup",
                    "_warm_sparse_swa_prefill_metadata_kernel",
                    "Warm sparse SWA prefill metadata kernels on the exact vendored path.",
                    "vllm/vllm/model_executor/warmup/sparse_mla_triton_warmup.py",
                    &["_warm_sparse_swa_prefill_metadata_kernel"],
                )?,
                callable(
                    "vllm.model_executor.warmup.sparse_mla_triton_warmup",
                    "_warm_prefill_chunk_metadata_kernel",
                    "Warm chunked prefill metadata kernels using vLLM's own indexing path.",
                    "vllm/vllm/model_executor/warmup/sparse_mla_triton_warmup.py",
                    &["_warm_prefill_chunk_metadata_kernel"],
                )?,
            ],
        },
        "decode_attention" => VllmIntegrationSurface {
            id: format!("compile-region:{}", region.name),
            scope_kind: IntegrationScopeKind::CompileRegion,
            scope_name: region.name.clone(),
            backend: Some(region.family),
            cache_namespace: Some("cuda-graph-cache".to_owned()),
            warmup_scope: Some("decode_attention".to_owned()),
            rationale: "Decode specialization should stay tied to vLLM's CUDA-graph and mixed-batch dummy-run path so capture boundaries match the real engine.".to_owned(),
            preserved_inputs: vec![
                "--cudagraph-capture-sizes".to_owned(),
                "--max-cudagraph-capture-size".to_owned(),
                "CompilationConfig".to_owned(),
            ],
            preserved_abstractions: vec![
                "Graph and region boundaries".to_owned(),
                "Layer identity in static forward context".to_owned(),
            ],
            isolation: isolation(
                VllmIsolationDisposition::NonIsolatable,
                false,
                false,
                &["Full worker startup context"],
                &[
                    "Decode warmup forces mixed-batch dummy runs during kernel warmup.",
                    "CUDA-graph capture sizing is derived from broader decode metadata builder state.",
                ],
                "Decode materialization is not a semantically real standalone subset build because the vendored warmup path crosses broader worker startup and mixed-batch capture boundaries.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &["def kernel_warmup(worker: \"Worker\"):", "create_mixed_batch=True"],
            )?,
            primary: callable(
                "vllm.model_executor.warmup.kernel_warmup",
                "kernel_warmup",
                "Run decode-owned startup through vLLM's kernel warmup orchestrator.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &["def kernel_warmup(worker: \"Worker\"):", "create_mixed_batch=True"],
            )?,
            auxiliary: vec![
                callable(
                    "vllm.v1.worker.gpu_model_runner",
                    "GPUModelRunner._dummy_run",
                    "Preserve vLLM's mixed prefill/decode dummy-run path for decode capture materialization.",
                    "vllm/vllm/v1/worker/gpu_model_runner.py",
                    &["def _dummy_run(", "create_mixed_batch: bool = False,"],
                )?,
                callable(
                    "vllm.v1.attention.backends.gdn_attn",
                    "GDNAttentionMetadataBuilder.__init__",
                    "Decode CUDA-graph buffers and capture bounds come from the vendored attention metadata builder.",
                    "vllm/vllm/v1/attention/backends/gdn_attn.py",
                    &["self.use_full_cuda_graph: bool =", "self.decode_cudagraph_max_bs: int = ("],
                )?,
            ],
        },
        "kv_cache_update" => VllmIntegrationSurface {
            id: format!("compile-region:{}", region.name),
            scope_kind: IntegrationScopeKind::CompileRegion,
            scope_name: region.name.clone(),
            backend: Some(region.family),
            cache_namespace: Some("flashinfer-autotune-cache".to_owned()),
            warmup_scope: Some("kv_cache_update".to_owned()),
            rationale: "FlashInfer KV-update specialization should remain on vLLM's mixed prefill/decode autotune and warmup path so tactic ownership stays engine-native.".to_owned(),
            preserved_inputs: vec![
                "--enable-flashinfer-autotune".to_owned(),
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR".to_owned(),
                "VLLM_HAS_FLASHINFER_CUBIN".to_owned(),
            ],
            preserved_abstractions: vec![
                "Cache ownership boundaries".to_owned(),
                "Graph and region boundaries".to_owned(),
            ],
            isolation: isolation(
                VllmIsolationDisposition::NonIsolatable,
                false,
                false,
                &["Full worker startup context"],
                &[
                    "FlashInfer KV-update warmup reuses mixed prefill/decode warmup orchestration.",
                    "Autotune cache resolution is runner-scoped rather than standalone-region scoped.",
                ],
                "KV-update materialization is not a semantically real standalone subset build because the vendored autotune seam depends on mixed-batch worker startup state.",
                "vllm/vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py",
                &[
                    "def deepseek_v4_sparse_mla_attention_warmup(worker: \"Worker\") -> None:",
                    "run_mixed_prefill_decode_warmup",
                ],
            )?,
            primary: callable(
                "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup",
                "deepseek_v4_sparse_mla_attention_warmup",
                "Use vLLM's DSv4 sparse-MLA warmup as the concrete integration seam for KV-update specialization.",
                "vllm/vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py",
                &["def deepseek_v4_sparse_mla_attention_warmup(worker: \"Worker\") -> None:", "run_mixed_prefill_decode_warmup"],
            )?,
            auxiliary: vec![
                callable(
                    "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup",
                    "run_mixed_prefill_decode_warmup",
                    "Preserve vLLM's mixed-batch warmup path for KV update and attention setup.",
                    "vllm/vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py",
                    &["run_mixed_prefill_decode_warmup"],
                )?,
                callable(
                    "vllm.model_executor.warmup.flashinfer_autotune_cache",
                    "resolve_flashinfer_autotune_file",
                    "Resolve FlashInfer tactic cache paths exactly the way vLLM does.",
                    "vllm/vllm/model_executor/warmup/flashinfer_autotune_cache.py",
                    &["def resolve_flashinfer_autotune_file(runner: \"GPUModelRunner\") -> Path:"],
                )?,
            ],
        },
        "moe_specialty_path" => VllmIntegrationSurface {
            id: format!("compile-region:{}", region.name),
            scope_kind: IntegrationScopeKind::CompileRegion,
            scope_name: region.name.clone(),
            backend: Some(region.family),
            cache_namespace: Some("compile-cache".to_owned()),
            warmup_scope: Some("moe_specialty_path".to_owned()),
            rationale: "MoE specialty compilation should preserve vLLM's own Inductor fallback policy and namespace handling instead of inventing a parallel fallback model.".to_owned(),
            preserved_inputs: vec!["CompilationConfig".to_owned(), "current_platform".to_owned()],
            preserved_abstractions: vec![
                "Custom-op boundaries".to_owned(),
                "Cache ownership boundaries".to_owned(),
            ],
            isolation: isolation(
                VllmIsolationDisposition::Standalone,
                true,
                true,
                &[],
                &[],
                "MoE fallback patching is a true standalone vendored seam because it is exposed as a module function with no worker-owned runtime context.",
                "vllm/vllm/env_override.py",
                &["def _patch_inductor_fallback_allow_list() -> None:"],
            )?,
            primary: callable(
                "vllm.env_override",
                "_patch_inductor_fallback_allow_list",
                "Keep vLLM's custom-op fallback allow-list patch as the concrete seam for MoE specialty fallback behavior.",
                "vllm/vllm/env_override.py",
                &["def _patch_inductor_fallback_allow_list() -> None:", "_VLLM_FALLBACK_NAMESPACE_PREFIXES"],
            )?,
            auxiliary: vec![callable(
                "vllm.env_override",
                "_VllmFallbackAllowList.__contains__",
                "Preserve vendored namespace membership checks for vLLM custom-op fallbacks.",
                "vllm/vllm/env_override.py",
                &[
                    "def __contains__(self, item):",
                    "for prefix in _VLLM_FALLBACK_NAMESPACE_PREFIXES:",
                ],
            )?],
        },
        other => {
            return Err(VllmIntegrationError::MissingSurface {
                scope: other.to_owned(),
            })
        }
    };

    Ok(apply_survey_context(integrated, survey))
}

fn integration_surface_for_cache(
    cache_surface: &CacheOwnershipSurface,
    survey: &AdapterSurvey,
) -> Result<VllmIntegrationSurface, VllmIntegrationError> {
    let integrated = match cache_surface.name.as_str() {
        "compile-cache" => VllmIntegrationSurface {
            id: "cache-surface:compile-cache".to_owned(),
            scope_kind: IntegrationScopeKind::CacheSurface,
            scope_name: cache_surface.name.clone(),
            backend: None,
            cache_namespace: Some(cache_surface.name.clone()),
            warmup_scope: None,
            rationale: cache_surface.rationale.clone(),
            preserved_inputs: vec![
                "CompilationConfig".to_owned(),
                "VLLM_DISABLE_COMPILE_CACHE".to_owned(),
                "VLLM_COMPILE_CACHE_SAVE_FORMAT".to_owned(),
            ],
            preserved_abstractions: vec![
                "Graph and region boundaries".to_owned(),
                "Cache ownership boundaries".to_owned(),
            ],
            isolation: isolation(
                VllmIsolationDisposition::ContextBound,
                true,
                false,
                &["PiecewiseBackend context"],
                &[],
                "Compile-cache ownership is subset-build valid, but write paths still run through vLLM's compilation backend context.",
                "vllm/vllm/compilation/caching.py",
                &["def build_aot_compile_plan(", "\"canonical_aot_plan_id\""],
            )?,
            primary: callable(
                "vllm.compilation.caching",
                "build_aot_compile_plan",
                "Compile-cache identity must follow vLLM's own normalized AOT compile plan.",
                "vllm/vllm/compilation/caching.py",
                &["def build_aot_compile_plan(", "\"canonical_aot_plan_id\""],
            )?,
            auxiliary: vec![
                callable(
                    "vllm.compilation.caching",
                    "StandaloneCompiledArtifacts.insert",
                    "Compiled artifact storage should preserve vLLM's standalone artifact layout.",
                    "vllm/vllm/compilation/caching.py",
                    &[
                        "class StandaloneCompiledArtifacts:",
                        "def insert(self, submod_name: str, shape: str, entry: bytes) -> None:",
                    ],
                )?,
                callable(
                    "vllm.compilation.backends",
                    "CompilationConfig.cache_dir",
                    "Keep vLLM's compilation backend cache-dir ownership intact.",
                    "vllm/vllm/compilation/backends.py",
                    &["self.compilation_config.cache_dir = cache_dir"],
                )?,
            ],
        },
        "flashinfer-autotune-cache" => VllmIntegrationSurface {
            id: "cache-surface:flashinfer-autotune-cache".to_owned(),
            scope_kind: IntegrationScopeKind::CacheSurface,
            scope_name: cache_surface.name.clone(),
            backend: None,
            cache_namespace: Some(cache_surface.name.clone()),
            warmup_scope: Some("kv_cache_update".to_owned()),
            rationale: cache_surface.rationale.clone(),
            preserved_inputs: vec![
                "--enable-flashinfer-autotune".to_owned(),
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR".to_owned(),
                "VLLM_HAS_FLASHINFER_CUBIN".to_owned(),
            ],
            preserved_abstractions: vec!["Cache ownership boundaries".to_owned()],
            isolation: isolation(
                VllmIsolationDisposition::NonIsolatable,
                false,
                false,
                &["GPUModelRunner context"],
                &[
                    "FlashInfer autotune cache paths are resolved from the live runner.",
                    "Autotune contents are produced by mixed-batch warmup rather than a standalone cache operation.",
                ],
                "FlashInfer autotune cache ownership is real, but standalone subset builds are not: the cache is produced from runner-scoped warmup state.",
                "vllm/vllm/model_executor/warmup/flashinfer_autotune_cache.py",
                &["def resolve_flashinfer_autotune_file(runner: \"GPUModelRunner\") -> Path:"],
            )?,
            primary: callable(
                "vllm.model_executor.warmup.flashinfer_autotune_cache",
                "resolve_flashinfer_autotune_file",
                "FlashInfer autotune cache location and ownership should stay on vLLM's native path.",
                "vllm/vllm/model_executor/warmup/flashinfer_autotune_cache.py",
                &["def resolve_flashinfer_autotune_file(runner: \"GPUModelRunner\") -> Path:"],
            )?,
            auxiliary: vec![callable(
                "vllm.model_executor.warmup.flashinfer_autotune_cache",
                "write_flashinfer_autotune_cache",
                "Persist FlashInfer tactic results through vLLM's atomic cache writer.",
                "vllm/vllm/model_executor/warmup/flashinfer_autotune_cache.py",
                &[
                    "def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:",
                ],
            )?],
        },
        "cuda-graph-cache" => VllmIntegrationSurface {
            id: "cache-surface:cuda-graph-cache".to_owned(),
            scope_kind: IntegrationScopeKind::CacheSurface,
            scope_name: cache_surface.name.clone(),
            backend: None,
            cache_namespace: Some(cache_surface.name.clone()),
            warmup_scope: Some("decode_attention".to_owned()),
            rationale: cache_surface.rationale.clone(),
            preserved_inputs: vec![
                "--cudagraph-capture-sizes".to_owned(),
                "--max-cudagraph-capture-size".to_owned(),
                "CompilationConfig".to_owned(),
            ],
            preserved_abstractions: vec!["Graph and region boundaries".to_owned()],
            isolation: isolation(
                VllmIsolationDisposition::NonIsolatable,
                false,
                false,
                &["Full worker startup context"],
                &[
                    "Decode graph-cache sizing lives inside attention metadata builder state.",
                    "Cache materialization is driven by decode kernel warmup rather than a standalone cache writer.",
                ],
                "CUDA-graph cache ownership is explicit, but standalone subset builds are not semantically real because capture state is created by broader decode startup.",
                "vllm/vllm/v1/attention/backends/gdn_attn.py",
                &[
                    "self.use_full_cuda_graph: bool =",
                    "self.decode_cudagraph_max_bs: int = (",
                ],
            )?,
            primary: callable(
                "vllm.v1.attention.backends.gdn_attn",
                "GDNAttentionMetadataBuilder.__init__",
                "Decode graph-cache ownership should track vLLM's own decode capture sizing path.",
                "vllm/vllm/v1/attention/backends/gdn_attn.py",
                &[
                    "self.use_full_cuda_graph: bool =",
                    "self.decode_cudagraph_max_bs: int = (",
                ],
            )?,
            auxiliary: vec![callable(
                "vllm.model_executor.warmup.kernel_warmup",
                "kernel_warmup",
                "Kernel warmup is the concrete startup seam that materializes decode capture state.",
                "vllm/vllm/model_executor/warmup/kernel_warmup.py",
                &[
                    "def kernel_warmup(worker: \"Worker\"):",
                    "create_mixed_batch=True",
                ],
            )?],
        },
        other => {
            return Err(VllmIntegrationError::MissingSurface {
                scope: other.to_owned(),
            });
        }
    };

    Ok(apply_survey_context(integrated, survey))
}

fn replay_root_for_surface(
    outcome: &PlanningOutcome,
    surface: &VllmIntegrationSurface,
) -> VllmReplayRoot {
    let manifest_paths = manifest_paths_for_surface(surface);
    let replay_boundary = if surface.scope_kind == IntegrationScopeKind::CompileRegion {
        format!("compile-region:{}", surface.scope_name)
    } else {
        format!("cache-surface:{}", surface.scope_name)
    };
    VllmReplayRoot {
        id: format!("replay-root:{}", surface.id),
        root_kind: if surface.scope_kind == IntegrationScopeKind::CompileRegion {
            VllmReplayRootKind::CompileRegion
        } else {
            VllmReplayRootKind::CacheSurface
        },
        surface_id: surface.id.clone(),
        scope_name: surface.scope_name.clone(),
        root_key: outcome.plan.structural_identity.plan_identity.clone(),
        cache_namespace: surface.cache_namespace.clone(),
        warmup_scope: surface.warmup_scope.clone(),
        replay_boundary,
        manifest_paths,
    }
}

fn manifest_paths_for_surface(surface: &VllmIntegrationSurface) -> Vec<String> {
    let mut manifest_paths = Vec::new();
    if surface.cache_namespace.is_some() {
        manifest_paths.push("graph_artifact_store.json".to_owned());
        manifest_paths.push("compile_replay_manifest.json".to_owned());
    }
    if surface.warmup_scope.is_some() {
        manifest_paths.push("warmup_materialization_manifest.json".to_owned());
    }
    if matches!(surface.cache_namespace.as_deref(), Some("cuda-graph-cache")) {
        manifest_paths.push("cudagraph_capture_manifest.json".to_owned());
    }
    if surface.id == "cache-surface:flashinfer-autotune-cache"
        || surface.scope_name.contains("flashinfer")
        || surface.id == "compile-region:kv_cache_update"
    {
        manifest_paths.push("autotune_cache_manifest.json".to_owned());
    }
    manifest_paths.sort();
    manifest_paths.dedup();
    manifest_paths
}

fn relevant_cache_surfaces<'a>(
    requirements: &[ArtifactRequirement],
    survey: &'a AdapterSurvey,
) -> Vec<&'a CacheOwnershipSurface> {
    let scopes = requirements
        .iter()
        .map(|requirement| requirement.scope.as_str())
        .collect::<BTreeSet<_>>();
    survey
        .cache_ownership_surfaces
        .iter()
        .filter(|surface| {
            surface
                .artifact_scopes
                .iter()
                .any(|scope| scopes.contains(scope.as_str()))
        })
        .collect()
}

fn apply_survey_context(
    mut surface: VllmIntegrationSurface,
    survey: &AdapterSurvey,
) -> VllmIntegrationSurface {
    surface
        .preserved_inputs
        .retain(|name| survey.config_inputs.iter().any(|input| &input.name == name));
    surface.preserved_abstractions.retain(|name| {
        survey
            .preserved_abstractions
            .iter()
            .any(|entry| &entry.name == name)
    });
    surface
}

fn callable(
    module: &str,
    callable: &str,
    summary: &str,
    file: &'static str,
    anchor_patterns: &[&'static str],
) -> Result<VllmCallableTarget, AdapterError> {
    let evidence = evidence(file, summary, anchor_patterns)?;
    Ok(VllmCallableTarget {
        module: module.to_owned(),
        callable: callable.to_owned(),
        summary: summary.to_owned(),
        evidence,
    })
}

fn isolation(
    disposition: VllmIsolationDisposition,
    subset_build_valid: bool,
    direct_entrypoint_invocable: bool,
    required_context: &[&str],
    blockers: &[&str],
    summary: &str,
    file: &'static str,
    anchor_patterns: &[&'static str],
) -> Result<VllmIsolationContract, AdapterError> {
    Ok(VllmIsolationContract {
        disposition,
        subset_build_valid,
        direct_entrypoint_invocable,
        required_context: required_context
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        blockers: blockers.iter().map(|value| (*value).to_owned()).collect(),
        evidence: evidence(file, summary, anchor_patterns)?,
    })
}

fn evidence(
    file: &'static str,
    summary: &str,
    anchor_patterns: &[&'static str],
) -> Result<SourceEvidence, AdapterError> {
    let mut index = SourceIndex::default();
    let anchors = anchor_patterns
        .iter()
        .map(|pattern| index.anchor(file, pattern))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SourceEvidence {
        summary: summary.to_owned(),
        anchors,
    })
}

#[derive(Default)]
struct SourceIndex;

impl SourceIndex {
    fn anchor(
        &mut self,
        file: &'static str,
        pattern: &'static str,
    ) -> Result<SourceAnchor, AdapterError> {
        let path = vllm::root().join(file.strip_prefix("vllm/").unwrap_or(file));
        let content =
            std::fs::read_to_string(&path).map_err(|source| AdapterError::ReadSource {
                file: path.display().to_string(),
                source,
            })?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Planner, PlannerHostSnapshot};
    use sock_core::{
        AcceleratorVendor, BackendFamily, BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer,
        CoveragePlane, EngineSource, ExecutionTopology, FailureMode, GuaranteeLevel,
        GuaranteeTarget, ModelRef, OperatingSystem, RawRequest, RequestedEnvironment, ShapePoint,
        ShapePolicy, ShapeRange, TargetEngine, WarmupPolicy,
    };

    fn host() -> PlannerHostSnapshot {
        PlannerHostSnapshot {
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm90".to_owned()],
            cuda_version: "12.4".to_owned(),
            driver_version: "550.54".to_owned(),
            python_abi: "cp311".to_owned(),
            libc_abi: "glibc-2.35".to_owned(),
            flashinfer_prebuilt_available: true,
            device_count: 1,
        }
    }

    fn request() -> RawRequest {
        RawRequest {
            engine: TargetEngine::Vllm,
            model: ModelRef {
                repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
                revision: "main".to_owned(),
            },
            engine_source: EngineSource {
                kind: "vendored".to_owned(),
                revision: vllm::revision().to_owned(),
            },
            environment: RequestedEnvironment {
                operating_system: OperatingSystem::Linux,
                accelerator_vendor: AcceleratorVendor::Nvidia,
                gpu_arches: vec!["sm90".to_owned()],
                cuda_version: "12.4".to_owned(),
                driver_version: "550.54".to_owned(),
                python_abi: "cp311".to_owned(),
                libc_abi: "glibc-2.35".to_owned(),
            },
            topology: ExecutionTopology {
                tensor_parallelism: 2,
                pipeline_parallelism: 1,
                replicas: 1,
            },
            backend_policy: BackendPolicy {
                preferred_families: vec![
                    BackendFamily::FlashInfer,
                    BackendFamily::Triton,
                    BackendFamily::CudaGraphs,
                ],
                packaging_strategy: sock_core::PackagingStrategy::PreferPrebuiltThenAot,
                runtime_jit_policy: sock_core::RuntimeJitPolicy {
                    disposition: sock_core::RuntimeJitDisposition::Forbidden,
                    max_residual_node_count: 0,
                },
                correctness_target: GuaranteeTarget {
                    level: GuaranteeLevel::ShapeBoundedAot,
                    failure_mode: FailureMode::FailClosed,
                },
                performance_target: GuaranteeTarget {
                    level: GuaranteeLevel::WarmupBounded,
                    failure_mode: FailureMode::FailClosed,
                },
            },
            shape_policy: ShapePolicy {
                correctness_range: ShapeRange {
                    min_batch_size: 1,
                    max_batch_size: 8,
                    min_sequence_length: 1,
                    max_sequence_length: 4096,
                },
                performance_range: ShapeRange {
                    min_batch_size: 1,
                    max_batch_size: 4,
                    min_sequence_length: 1,
                    max_sequence_length: 2048,
                },
                hot_shapes: vec![ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: CoveragePlane::Performance,
                }],
                cuda_graph_shapes: vec![ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: CoveragePlane::CudaGraph,
                }],
            },
            cache_policy: CachePolicy {
                namespace: "prod".to_owned(),
                allow_cross_machine_reuse: false,
            },
            warmup_policy: WarmupPolicy {
                max_warmup_steps: 6,
                verify_cuda_graph_capture: true,
            },
            optimization_policy: sock_core::OptimizationPolicy {
                level: sock_core::OptimizationLevel::O2,
            },
            layered_config: vec![ConfigLayer {
                name: "project".to_owned(),
                precedence: 1,
                entries: vec![ConfigEntry {
                    key: "tensor_parallel_size".to_owned(),
                    value: "2".to_owned(),
                }],
            }],
        }
    }

    #[test]
    fn integration_document_resolves_real_vllm_surfaces() {
        let planner = Planner::new(host());
        let outcome = planner.resolve(request()).expect("plan");
        let doc = build_vllm_integration_document(&outcome).expect("integration doc");

        assert!(!doc.surfaces.is_empty());
        assert!(
            doc.surfaces
                .iter()
                .any(|surface| surface.id == "compile-region:prefill_attention")
        );
        assert!(
            doc.surfaces
                .iter()
                .any(|surface| surface.id == "cache-surface:flashinfer-autotune-cache")
        );
        assert!(
            doc.replay_roots
                .iter()
                .any(|root| root.surface_id == "compile-region:prefill_attention")
        );
        assert!(
            doc.replay_roots
                .iter()
                .all(|root| root.root_key == doc.plan_identity)
        );

        for surface in &doc.surfaces {
            assert!(!surface.primary.evidence.anchors.is_empty());
            for anchor in surface.primary.evidence.anchors.iter().chain(
                surface
                    .auxiliary
                    .iter()
                    .flat_map(|call| call.evidence.anchors.iter()),
            ) {
                let path =
                    vllm::root().join(anchor.file.strip_prefix("vllm/").unwrap_or(&anchor.file));
                let content = std::fs::read_to_string(path).expect("source file should exist");
                assert!(anchor.line >= 1);
                assert!(anchor.line <= content.lines().count());
            }
        }
    }

    #[test]
    fn integration_document_marks_non_isolatable_runtime_bound_surfaces() {
        let planner = Planner::new(host());
        let outcome = planner.resolve(request()).expect("plan");
        let doc = build_vllm_integration_document(&outcome).expect("integration doc");

        let decode = doc
            .surfaces
            .iter()
            .find(|surface| surface.id == "compile-region:decode_attention")
            .expect("decode surface");
        assert_eq!(
            decode.isolation.disposition,
            VllmIsolationDisposition::NonIsolatable
        );
        assert!(!decode.isolation.subset_build_valid);
        assert!(
            decode
                .isolation
                .blockers
                .iter()
                .any(|blocker| blocker.contains("mixed-batch"))
        );
        assert!(!decode.isolation.evidence.anchors.is_empty());

        let prefill = doc
            .surfaces
            .iter()
            .find(|surface| surface.id == "compile-region:prefill_attention")
            .expect("prefill surface");
        assert_eq!(
            prefill.isolation.disposition,
            VllmIsolationDisposition::ContextBound
        );
        assert!(prefill.isolation.subset_build_valid);
        assert!(
            prefill
                .isolation
                .required_context
                .iter()
                .any(|context| context == "Worker context")
        );
        let prefill_root = doc
            .replay_roots
            .iter()
            .find(|root| root.surface_id == "compile-region:prefill_attention")
            .expect("prefill replay root");
        assert_eq!(prefill_root.root_kind, VllmReplayRootKind::CompileRegion);
        assert!(
            prefill_root
                .manifest_paths
                .iter()
                .any(|path| path == "compile_replay_manifest.json")
        );
        assert!(
            prefill_root
                .manifest_paths
                .iter()
                .any(|path| path == "warmup_materialization_manifest.json")
        );
    }
}
