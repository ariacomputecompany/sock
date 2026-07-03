use serde::{Deserialize, Serialize};

use crate::{CanonicalHash, ResolvedBuildPlan, SchemaVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

impl OptimizationLevel {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::O0 => "O0",
            Self::O1 => "O1",
            Self::O2 => "O2",
            Self::O3 => "O3",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationPolicy {
    pub level: OptimizationLevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationPhaseBudget {
    pub phase_name: String,
    pub startup_budget_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationArtifactBudget {
    pub max_artifact_count: u32,
    pub max_rank_local_artifacts: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationAction {
    pub category: String,
    pub effect: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationEnvelope {
    pub level: OptimizationLevel,
    pub profile_name: String,
    pub compile_secondary_backends: bool,
    pub include_performance_warmup: bool,
    pub include_cuda_graphs: bool,
    pub enable_autotune: bool,
    pub startup_budget_ms: u64,
    pub max_warmup_steps: u32,
    pub artifact_budget: OptimizationArtifactBudget,
    pub phase_budgets: Vec<OptimizationPhaseBudget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationExplainDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub request_identity: CanonicalHash,
    pub optimization_identity: CanonicalHash,
    pub level: OptimizationLevel,
    pub profile_name: String,
    pub startup_budget_ms: u64,
    pub max_warmup_steps: u32,
    pub artifact_budget: OptimizationArtifactBudget,
    pub phase_budgets: Vec<OptimizationPhaseBudget>,
    pub compile_actions: Vec<OptimizationAction>,
    pub warmup_actions: Vec<OptimizationAction>,
    pub autotune_actions: Vec<OptimizationAction>,
    pub graph_actions: Vec<OptimizationAction>,
}

impl OptimizationEnvelope {
    #[must_use]
    pub fn profile_name(level: OptimizationLevel) -> &'static str {
        match level {
            OptimizationLevel::O0 => "minimal_dev",
            OptimizationLevel::O1 => "selective_compile",
            OptimizationLevel::O2 => "balanced_prod",
            OptimizationLevel::O3 => "max_materialized_prod",
        }
    }

    #[must_use]
    pub fn from_level(level: OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::O0 => Self {
                level,
                profile_name: Self::profile_name(level).to_owned(),
                compile_secondary_backends: false,
                include_performance_warmup: false,
                include_cuda_graphs: false,
                enable_autotune: false,
                startup_budget_ms: 3_500,
                max_warmup_steps: 16,
                artifact_budget: OptimizationArtifactBudget {
                    max_artifact_count: 12,
                    max_rank_local_artifacts: 2,
                },
                phase_budgets: vec![
                    OptimizationPhaseBudget {
                        phase_name: "compile".to_owned(),
                        startup_budget_ms: 2_800,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "link_assemble".to_owned(),
                        startup_budget_ms: 300,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "warmup_materialize".to_owned(),
                        startup_budget_ms: 400,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "verify".to_owned(),
                        startup_budget_ms: 0,
                    },
                ],
            },
            OptimizationLevel::O1 => Self {
                level,
                profile_name: Self::profile_name(level).to_owned(),
                compile_secondary_backends: false,
                include_performance_warmup: true,
                include_cuda_graphs: false,
                enable_autotune: true,
                startup_budget_ms: 4_500,
                max_warmup_steps: 8,
                artifact_budget: OptimizationArtifactBudget {
                    max_artifact_count: 6,
                    max_rank_local_artifacts: 0,
                },
                phase_budgets: vec![
                    OptimizationPhaseBudget {
                        phase_name: "compile".to_owned(),
                        startup_budget_ms: 3_200,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "link_assemble".to_owned(),
                        startup_budget_ms: 400,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "warmup_materialize".to_owned(),
                        startup_budget_ms: 900,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "verify".to_owned(),
                        startup_budget_ms: 0,
                    },
                ],
            },
            OptimizationLevel::O2 => Self {
                level,
                profile_name: Self::profile_name(level).to_owned(),
                compile_secondary_backends: true,
                include_performance_warmup: true,
                include_cuda_graphs: true,
                enable_autotune: true,
                startup_budget_ms: 8_500,
                max_warmup_steps: 32,
                artifact_budget: OptimizationArtifactBudget {
                    max_artifact_count: 12,
                    max_rank_local_artifacts: 4,
                },
                phase_budgets: vec![
                    OptimizationPhaseBudget {
                        phase_name: "compile".to_owned(),
                        startup_budget_ms: 5_500,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "link_assemble".to_owned(),
                        startup_budget_ms: 700,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "warmup_materialize".to_owned(),
                        startup_budget_ms: 2_300,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "verify".to_owned(),
                        startup_budget_ms: 0,
                    },
                ],
            },
            OptimizationLevel::O3 => Self {
                level,
                profile_name: Self::profile_name(level).to_owned(),
                compile_secondary_backends: true,
                include_performance_warmup: true,
                include_cuda_graphs: true,
                enable_autotune: true,
                startup_budget_ms: 12_000,
                max_warmup_steps: 48,
                artifact_budget: OptimizationArtifactBudget {
                    max_artifact_count: 16,
                    max_rank_local_artifacts: 6,
                },
                phase_budgets: vec![
                    OptimizationPhaseBudget {
                        phase_name: "compile".to_owned(),
                        startup_budget_ms: 7_500,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "link_assemble".to_owned(),
                        startup_budget_ms: 1_000,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "warmup_materialize".to_owned(),
                        startup_budget_ms: 3_500,
                    },
                    OptimizationPhaseBudget {
                        phase_name: "verify".to_owned(),
                        startup_budget_ms: 0,
                    },
                ],
            },
        }
    }
}

impl OptimizationExplainDocument {
    #[must_use]
    pub fn from_plan(plan: &ResolvedBuildPlan) -> Self {
        let envelope = &plan.optimization_envelope;
        let compile_actions = vec![
            OptimizationAction {
                category: "compile".to_owned(),
                effect: format!(
                    "secondary_backends={}",
                    if envelope.compile_secondary_backends {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ),
                reason: format!(
                    "{} keeps compile closure {} secondary backend families",
                    envelope.level.as_str(),
                    if envelope.compile_secondary_backends {
                        "open to"
                    } else {
                        "closed to"
                    }
                ),
            },
            OptimizationAction {
                category: "compile".to_owned(),
                effect: format!("compile_regions={}", plan.compile_regions.len()),
                reason: "compile identity is bound to explicit region selection and request policy"
                    .to_owned(),
            },
        ];
        let warmup_actions = vec![
            OptimizationAction {
                category: "warmup".to_owned(),
                effect: format!(
                    "performance_warmup={}",
                    if envelope.include_performance_warmup {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ),
                reason: format!(
                    "{} maps to a max_warmup_steps budget of {}",
                    envelope.level.as_str(),
                    envelope.max_warmup_steps
                ),
            },
            OptimizationAction {
                category: "warmup".to_owned(),
                effect: format!("warmup_obligations={}", plan.warmup_obligations.len()),
                reason: "warmup is emitted as explicit obligations instead of hidden startup work"
                    .to_owned(),
            },
        ];
        let autotune_actions = vec![OptimizationAction {
            category: "autotune".to_owned(),
            effect: format!(
                "autotune={}",
                if envelope.enable_autotune {
                    "enabled"
                } else {
                    "disabled"
                }
            ),
            reason:
                "autotune follows the optimization envelope instead of ambient backend defaults"
                    .to_owned(),
        }];
        let graph_actions = vec![OptimizationAction {
            category: "graph".to_owned(),
            effect: format!(
                "cuda_graphs={}",
                if envelope.include_cuda_graphs {
                    "enabled"
                } else {
                    "disabled"
                }
            ),
            reason: "graph capture artifacts are materialized only when the chosen optimization level budgets them"
                .to_owned(),
        }];

        Self {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            request_identity: plan.normalized_request.identity.clone(),
            optimization_identity: plan.structural_identity.optimization_identity.clone(),
            level: envelope.level,
            profile_name: envelope.profile_name.clone(),
            startup_budget_ms: envelope.startup_budget_ms,
            max_warmup_steps: envelope.max_warmup_steps,
            artifact_budget: envelope.artifact_budget.clone(),
            phase_budgets: envelope.phase_budgets.clone(),
            compile_actions,
            warmup_actions,
            autotune_actions,
            graph_actions,
        }
    }
}
