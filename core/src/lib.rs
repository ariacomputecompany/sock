pub mod accelerator;
pub mod adapter;
pub mod backend;
pub mod bundle;
pub mod canonical;
pub mod diagnostics;
pub mod entrypoint;
pub mod governance;
pub mod identity;
pub mod integration;
pub mod kv_layout;
pub mod materialization;
pub mod measurement;
pub mod model;
pub mod operator;
pub mod optimization;
pub mod replay;
pub mod request;
pub mod rewrite;
pub mod runtime;
pub mod soc;
pub mod verification;

pub use accelerator::*;
pub use adapter::*;
pub use backend::*;
pub use bundle::{ArtifactManifestDocument, ReplayBundle, ReplayBundleError, ReplayBundleMetadata};
pub use canonical::{
    CanonicalError, CanonicalHash, canonical_hash, canonical_json, parse_canonical_json,
};
pub use diagnostics::{
    DiagnosticCategory, DiagnosticConfidence, DiagnosticEvidence, DiagnosticsDocument,
    RewriteTraceDocument, SchemaVersion, StructuredDiagnostic,
};
pub use entrypoint::*;
pub use identity::*;
pub use integration::*;
pub use kv_layout::*;
pub use materialization::*;
pub use measurement::*;
pub use model::*;
pub use operator::{
    render_backend_decision, render_diagnostics, render_explain, render_optimization_explain,
    render_plan_summary, render_replay_bundle_explain, render_soc_explain,
    render_structural_identity, render_verification_report,
};
pub use optimization::*;
pub use replay::*;
pub use request::*;
pub use rewrite::*;
pub use runtime::*;
pub use soc::*;
pub use verification::*;
