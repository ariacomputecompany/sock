use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use sock_core::{
    CanonicalError, SchemaVersion, VllmCallStrategy, VllmContextKind, VllmEntrypoint,
    VllmEntrypointDocument, VllmIntegrationDocument,
};
use thiserror::Error;

use crate::PlanningOutcome;

#[derive(Debug, Error)]
pub enum VllmEntrypointError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical error: {0}")]
    Canonical(#[from] CanonicalError),
    #[error("integration surface {surface_id} is not invocable")]
    NonInvocableSurface { surface_id: String },
}

pub fn build_vllm_entrypoint_document(
    outcome: &PlanningOutcome,
    integration: &VllmIntegrationDocument,
) -> Result<VllmEntrypointDocument, VllmEntrypointError> {
    let mut entrypoints = integration
        .surfaces
        .iter()
        .filter(|surface| surface.scope_kind == sock_core::IntegrationScopeKind::CompileRegion)
        .map(|surface| match surface.scope_name.as_str() {
            "transformer_block_body" => Ok(VllmEntrypoint {
                id: format!("entrypoint:{}", surface.scope_name),
                surface_id: surface.id.clone(),
                scope_name: surface.scope_name.clone(),
                context_kind: VllmContextKind::PiecewiseBackend,
                call_strategy: VllmCallStrategy::ContextMethod,
                callable: surface.primary.clone(),
                args: BTreeMap::new(),
                required_env: vec![
                    "VLLM_DISABLE_COMPILE_CACHE".to_owned(),
                    "VLLM_COMPILE_CACHE_SAVE_FORMAT".to_owned(),
                ],
                preserved_inputs: surface.preserved_inputs.clone(),
                preserved_abstractions: surface.preserved_abstractions.clone(),
                summary: "Invoke the vendored piecewise backend to compile all selected transformer-block ranges.".to_owned(),
                manifest_path: manifest_path(&surface.scope_name),
                wrapper_path: wrapper_path(&surface.scope_name),
            }),
            "prefill_attention" => Ok(VllmEntrypoint {
                id: format!("entrypoint:{}", surface.scope_name),
                surface_id: surface.id.clone(),
                scope_name: surface.scope_name.clone(),
                context_kind: VllmContextKind::Worker,
                call_strategy: VllmCallStrategy::ModuleFunctionWithContext,
                callable: surface.primary.clone(),
                args: BTreeMap::new(),
                required_env: Vec::new(),
                preserved_inputs: surface.preserved_inputs.clone(),
                preserved_abstractions: surface.preserved_abstractions.clone(),
                summary: "Run vendored sparse-MLA Triton prefill warmup on a worker context.".to_owned(),
                manifest_path: manifest_path(&surface.scope_name),
                wrapper_path: wrapper_path(&surface.scope_name),
            }),
            "decode_attention" => Ok(VllmEntrypoint {
                id: format!("entrypoint:{}", surface.scope_name),
                surface_id: surface.id.clone(),
                scope_name: surface.scope_name.clone(),
                context_kind: VllmContextKind::Worker,
                call_strategy: VllmCallStrategy::ModuleFunctionWithContext,
                callable: surface.primary.clone(),
                args: BTreeMap::new(),
                required_env: Vec::new(),
                preserved_inputs: surface.preserved_inputs.clone(),
                preserved_abstractions: surface.preserved_abstractions.clone(),
                summary: "Run vendored kernel warmup for decode-owned CUDA-graph materialization on a worker context.".to_owned(),
                manifest_path: manifest_path(&surface.scope_name),
                wrapper_path: wrapper_path(&surface.scope_name),
            }),
            "kv_cache_update" => Ok(VllmEntrypoint {
                id: format!("entrypoint:{}", surface.scope_name),
                surface_id: surface.id.clone(),
                scope_name: surface.scope_name.clone(),
                context_kind: VllmContextKind::Worker,
                call_strategy: VllmCallStrategy::ModuleFunctionWithContext,
                callable: surface.primary.clone(),
                args: BTreeMap::new(),
                required_env: vec![
                    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR".to_owned(),
                    "VLLM_HAS_FLASHINFER_CUBIN".to_owned(),
                ],
                preserved_inputs: surface.preserved_inputs.clone(),
                preserved_abstractions: surface.preserved_abstractions.clone(),
                summary: "Run vendored FlashInfer sparse-MLA warmup and autotune on a worker context.".to_owned(),
                manifest_path: manifest_path(&surface.scope_name),
                wrapper_path: wrapper_path(&surface.scope_name),
            }),
            "moe_specialty_path" => Ok(VllmEntrypoint {
                id: format!("entrypoint:{}", surface.scope_name),
                surface_id: surface.id.clone(),
                scope_name: surface.scope_name.clone(),
                context_kind: VllmContextKind::None,
                call_strategy: VllmCallStrategy::ModuleFunction,
                callable: surface.primary.clone(),
                args: BTreeMap::new(),
                required_env: Vec::new(),
                preserved_inputs: surface.preserved_inputs.clone(),
                preserved_abstractions: surface.preserved_abstractions.clone(),
                summary: "Apply vendored Inductor fallback patching for MoE specialty surfaces.".to_owned(),
                manifest_path: manifest_path(&surface.scope_name),
                wrapper_path: wrapper_path(&surface.scope_name),
            }),
            other => Err(VllmEntrypointError::NonInvocableSurface {
                surface_id: other.to_owned(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    entrypoints.sort_by(|left, right| left.id.cmp(&right.id));

    Ok(VllmEntrypointDocument {
        schema_version: SchemaVersion::current(),
        plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
        engine_root: integration.engine_root.clone(),
        engine_revision: integration.engine_revision.clone(),
        entrypoints,
    })
}

pub fn emit_vllm_entrypoints(
    out_dir: &Path,
    document: &VllmEntrypointDocument,
) -> Result<(), VllmEntrypointError> {
    let root = out_dir.join("vllm-entrypoints");
    let surfaces = root.join("surfaces");
    fs::create_dir_all(&surfaces)?;
    fs::write(
        out_dir.join("vllm_entrypoints.json"),
        sock_core::canonical_json(document)?.as_bytes(),
    )?;
    fs::write(
        root.join("invoke_vllm_surface.py"),
        dispatcher_script().as_bytes(),
    )?;

    for entrypoint in &document.entrypoints {
        fs::write(
            out_dir.join(&entrypoint.manifest_path),
            sock_core::canonical_json(entrypoint)?.as_bytes(),
        )?;
        fs::write(
            out_dir.join(&entrypoint.wrapper_path),
            wrapper_script(&entrypoint.manifest_path).as_bytes(),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let wrapper_path = out_dir.join(&entrypoint.wrapper_path);
            let mut permissions = fs::metadata(&wrapper_path)?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(wrapper_path, permissions)?;
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dispatcher = root.join("invoke_vllm_surface.py");
        let mut permissions = fs::metadata(&dispatcher)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(dispatcher, permissions)?;
    }

    Ok(())
}

fn manifest_path(scope: &str) -> String {
    format!("vllm-entrypoints/surfaces/{}.json", slug(scope))
}

fn wrapper_path(scope: &str) -> String {
    format!("vllm-entrypoints/{}.sh", slug(scope))
}

fn slug(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn wrapper_script(manifest_path: &str) -> String {
    format!(
        "#!/usr/bin/env sh\nset -eu\nDIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nexec \"${{PYTHON:-python3}}\" \"$DIR/invoke_vllm_surface.py\" --manifest \"$DIR/../{}\" \"$@\"\n",
        manifest_path
    )
}

fn dispatcher_script() -> &'static str {
    r#"#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from pathlib import Path


def _load_symbol(module_name: str, attr_path: str):
    module = importlib.import_module(module_name)
    value = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("context factory must be module:callable")
    module_name, callable_name = spec.split(":", 1)
    return _load_symbol(module_name, callable_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-factory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    document = json.loads(manifest_path.read_text())
    engine_root = document["engine_root"]
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)

    target = _load_symbol(document["callable"]["module"], document["callable"]["callable"])
    kwargs = document.get("args", {})
    if args.dry_run:
        print(json.dumps({
            "entrypoint": document["id"],
            "surface_id": document["surface_id"],
            "context_kind": document["context_kind"],
            "call_strategy": document["call_strategy"],
            "required_env": document.get("required_env", []),
        }, sort_keys=True))
        return 0

    context_kind = document["context_kind"]
    strategy = document["call_strategy"]
    context = None
    if context_kind != "none":
        if not args.context_factory:
            raise SystemExit("--context-factory is required for this surface")
        context = _load_factory(args.context_factory)(document)

    if strategy == "module_function":
        target(**kwargs)
    elif strategy == "module_function_with_context":
        target(context, **kwargs)
    elif strategy == "context_method":
        method = getattr(context, document["callable"]["callable"].split(".")[-1])
        method(**kwargs)
    else:
        raise SystemExit(f"unsupported strategy: {strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Planner, PlannerHostSnapshot, build_vllm_integration_document, vllm};
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
            layered_config: vec![
                ConfigLayer {
                    name: "env".to_owned(),
                    precedence: 0,
                    entries: vec![ConfigEntry {
                        key: "VLLM_USE_V1".to_owned(),
                        value: "1".to_owned(),
                    }],
                },
                ConfigLayer {
                    name: "project".to_owned(),
                    precedence: 1,
                    entries: vec![ConfigEntry {
                        key: "tensor_parallel_size".to_owned(),
                        value: "2".to_owned(),
                    }],
                },
            ],
        }
    }

    #[test]
    fn entrypoint_document_tracks_invocable_vllm_surfaces() {
        let planner = Planner::new(host());
        let outcome = planner.resolve(request()).expect("plan");
        let integration = build_vllm_integration_document(&outcome).expect("integration");
        let entrypoints =
            build_vllm_entrypoint_document(&outcome, &integration).expect("entrypoints");

        assert!(
            entrypoints
                .entrypoints
                .iter()
                .any(|entrypoint| entrypoint.scope_name == "prefill_attention")
        );
        assert!(
            entrypoints
                .entrypoints
                .iter()
                .any(|entrypoint| entrypoint.scope_name == "decode_attention")
        );
        assert!(
            entrypoints
                .entrypoints
                .iter()
                .all(|entrypoint| entrypoint.wrapper_path.starts_with("vllm-entrypoints/"))
        );
    }
}
