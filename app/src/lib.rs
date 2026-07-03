use sock_core::{
    AcceleratorVendor, BackendFamily, BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer,
    CoveragePlane, EngineSource, ExecutionTopology, FailureMode, GuaranteeLevel, GuaranteeTarget,
    ModelRef, OperatingSystem, RawRequest, RequestedEnvironment, ShapePoint, ShapePolicy,
    ShapeRange, TargetEngine, WarmupPolicy,
};
use sock_engine::{PlannerHostSnapshot, vllm};

#[must_use]
pub fn default_host_snapshot() -> PlannerHostSnapshot {
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

#[must_use]
pub fn default_request() -> RawRequest {
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
            require_prebuilt_artifacts: true,
            allow_runtime_jit: false,
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
            hot_shapes: vec![
                ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: CoveragePlane::Performance,
                },
                ShapePoint {
                    batch_size: 4,
                    sequence_length: 2048,
                    plane: CoveragePlane::Performance,
                },
            ],
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
