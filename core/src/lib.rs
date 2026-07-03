use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetEngine {
    Vllm,
}

impl TargetEngine {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeEnvelope {
    pub name: String,
}

impl ShapeEnvelope {
    #[must_use]
    pub fn bounded(name: &str) -> Self {
        Self {
            name: name.to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildPlan {
    pub engine: TargetEngine,
    pub envelope: ShapeEnvelope,
}
