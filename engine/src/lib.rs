mod executor;
mod planner;
mod scope;
pub mod vllm;
mod vllm_adapter;
mod vllm_entrypoint;
mod vllm_integration;

pub use executor::{MaterializationError, MaterializationExecutor, StorageRoots};
pub use planner::{PlanError, Planner, PlannerHostSnapshot, PlanningOutcome};
pub use scope::{BuildReadiness, BuildScope, BuildTopologyScope};
pub use vllm_entrypoint::{
    VllmEntrypointError, build_vllm_entrypoint_document, emit_vllm_entrypoints,
};
pub use vllm_integration::{
    VllmIntegrationError, build_vllm_integration_document, validate_scoped_vllm_subset,
};
