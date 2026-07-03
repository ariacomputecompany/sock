use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
}

impl SchemaVersion {
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    #[must_use]
    pub const fn current() -> Self {
        Self::CURRENT
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilityClass {
    Stable,
    Experimental,
    DebugOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityClass {
    StableWithinMajor,
    BreakingOnMajor,
    NoCompatibilityGuarantee,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactCompatibilityPolicy {
    pub schema_version: SchemaVersion,
    pub compatibility: CompatibilityClass,
    pub cache_directory_pattern: String,
    pub migration_rule: String,
}

impl ArtifactCompatibilityPolicy {
    #[must_use]
    pub fn v1() -> Self {
        Self {
            schema_version: SchemaVersion::current(),
            compatibility: CompatibilityClass::StableWithinMajor,
            cache_directory_pattern: "artifacts/v{major}".to_owned(),
            migration_rule: "Major schema changes require a new cache root; minor schema changes remain forward-readable and reuse the existing cache root.".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicApiSurface {
    Stable,
    Experimental,
    DebugOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectMilestone {
    V0,
    V1,
    PostV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MilestoneDefinition {
    pub milestone: ProjectMilestone,
    pub summary: String,
    pub mandatory_optimizations: Vec<String>,
    pub exit_criteria: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MilestoneCatalog {
    pub definitions: Vec<MilestoneDefinition>,
}

impl MilestoneCatalog {
    #[must_use]
    pub fn production_defaults() -> Self {
        Self {
            definitions: vec![
                MilestoneDefinition {
                    milestone: ProjectMilestone::V0,
                    summary: "Measurement plus substrate for deterministic vLLM build planning."
                        .to_owned(),
                    mandatory_optimizations: vec![
                        "startup phase attribution".to_owned(),
                        "runtime-jit witness collection".to_owned(),
                        "multi-layer structural identity".to_owned(),
                    ],
                    exit_criteria: vec![
                        "Deterministic BuildPlan emission for supported vLLM requests.".to_owned(),
                        "Attributed startup timings persist as first-class artifacts.".to_owned(),
                        "At least one real verification trace can be replayed and checked."
                            .to_owned(),
                    ],
                },
                MilestoneDefinition {
                    milestone: ProjectMilestone::V1,
                    summary: "Deep vLLM closure quality for NVIDIA on Linux.".to_owned(),
                    mandatory_optimizations: vec![
                        "shape-envelope planning".to_owned(),
                        "regional compilation discovery".to_owned(),
                        "hybrid prebuilt/aot/jit artifact planning".to_owned(),
                        "leader/follower distributed materialization".to_owned(),
                    ],
                    exit_criteria: vec![
                        "ArtifactClosure is verifiable and replayable.".to_owned(),
                        "GuaranteeEnvelope is backed by evidence rather than claims.".to_owned(),
                        "Repeated equivalent builds materially reuse artifacts.".to_owned(),
                    ],
                },
                MilestoneDefinition {
                    milestone: ProjectMilestone::PostV1,
                    summary: "Broaden engine coverage only after vLLM closure is proven."
                        .to_owned(),
                    mandatory_optimizations: vec!["new engine adapters".to_owned()],
                    exit_criteria: vec![
                        "New adapters preserve the canonical schema rather than dilute it."
                            .to_owned(),
                    ],
                },
            ],
        }
    }
}
