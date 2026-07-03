use serde::{Deserialize, Serialize};

use crate::backend::{PackagingStrategy, RuntimeJitPolicy};
use crate::canonical::{CanonicalError, CanonicalHash, canonical_hash};
use crate::{
    AcceleratorVendor, BackendFamily, CoveragePlane, FailureMode, GuaranteeLevel, OperatingSystem,
    TargetEngine,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConfigLayer {
    pub name: String,
    pub precedence: u8,
    pub entries: Vec<ConfigEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConfigEntry {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelRef {
    pub repository: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EngineSource {
    pub kind: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestedEnvironment {
    pub operating_system: OperatingSystem,
    pub accelerator_vendor: AcceleratorVendor,
    pub gpu_arches: Vec<String>,
    pub cuda_version: String,
    pub driver_version: String,
    pub python_abi: String,
    pub libc_abi: String,
}

impl RequestedEnvironment {
    pub fn canonicalize(&mut self) {
        self.gpu_arches.sort();
        self.gpu_arches.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExecutionTopology {
    pub tensor_parallelism: u16,
    pub pipeline_parallelism: u16,
    pub replicas: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GuaranteeTarget {
    pub level: GuaranteeLevel,
    pub failure_mode: FailureMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPolicy {
    pub preferred_families: Vec<BackendFamily>,
    pub packaging_strategy: PackagingStrategy,
    pub runtime_jit_policy: RuntimeJitPolicy,
    pub correctness_target: GuaranteeTarget,
    pub performance_target: GuaranteeTarget,
}

impl BackendPolicy {
    pub fn canonicalize(&mut self) {
        self.preferred_families.sort();
        self.preferred_families.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapeRange {
    pub min_batch_size: u32,
    pub max_batch_size: u32,
    pub min_sequence_length: u32,
    pub max_sequence_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapePoint {
    pub batch_size: u32,
    pub sequence_length: u32,
    pub plane: CoveragePlane,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapePolicy {
    pub correctness_range: ShapeRange,
    pub performance_range: ShapeRange,
    pub hot_shapes: Vec<ShapePoint>,
    pub cuda_graph_shapes: Vec<ShapePoint>,
}

impl ShapePolicy {
    pub fn canonicalize(&mut self) {
        self.hot_shapes.sort();
        self.hot_shapes.dedup();
        self.cuda_graph_shapes.sort();
        self.cuda_graph_shapes.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CachePolicy {
    pub namespace: String,
    pub allow_cross_machine_reuse: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarmupPolicy {
    pub max_warmup_steps: u32,
    pub verify_cuda_graph_capture: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawRequest {
    pub engine: TargetEngine,
    pub model: ModelRef,
    pub engine_source: EngineSource,
    pub environment: RequestedEnvironment,
    pub topology: ExecutionTopology,
    pub backend_policy: BackendPolicy,
    pub shape_policy: ShapePolicy,
    pub cache_policy: CachePolicy,
    pub warmup_policy: WarmupPolicy,
    pub layered_config: Vec<ConfigLayer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedRequest {
    pub engine: TargetEngine,
    pub model: ModelRef,
    pub engine_source: EngineSource,
    pub environment: RequestedEnvironment,
    pub topology: ExecutionTopology,
    pub backend_policy: BackendPolicy,
    pub shape_policy: ShapePolicy,
    pub cache_policy: CachePolicy,
    pub warmup_policy: WarmupPolicy,
    pub layered_config: Vec<ConfigLayer>,
    pub identity: CanonicalHash,
}

impl RawRequest {
    pub fn normalize(mut self) -> Result<NormalizedRequest, CanonicalError> {
        self.environment.canonicalize();
        self.backend_policy.canonicalize();
        self.shape_policy.canonicalize();
        self.layered_config
            .sort_by_key(|layer| (layer.precedence, layer.name.clone()));
        for layer in &mut self.layered_config {
            layer.entries.sort();
            layer.entries.dedup();
        }

        let body = NormalizedRequestBody {
            engine: self.engine,
            model: self.model,
            engine_source: self.engine_source,
            environment: self.environment,
            topology: self.topology,
            backend_policy: self.backend_policy,
            shape_policy: self.shape_policy,
            cache_policy: self.cache_policy,
            warmup_policy: self.warmup_policy,
            layered_config: self.layered_config,
        };
        let identity = canonical_hash(&body)?;

        Ok(NormalizedRequest {
            engine: body.engine,
            model: body.model,
            engine_source: body.engine_source,
            environment: body.environment,
            topology: body.topology,
            backend_policy: body.backend_policy,
            shape_policy: body.shape_policy,
            cache_policy: body.cache_policy,
            warmup_policy: body.warmup_policy,
            layered_config: body.layered_config,
            identity,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct NormalizedRequestBody {
    engine: TargetEngine,
    model: ModelRef,
    engine_source: EngineSource,
    environment: RequestedEnvironment,
    topology: ExecutionTopology,
    backend_policy: BackendPolicy,
    shape_policy: ShapePolicy,
    cache_policy: CachePolicy,
    warmup_policy: WarmupPolicy,
    layered_config: Vec<ConfigLayer>,
}
