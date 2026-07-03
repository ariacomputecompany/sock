use serde::{Deserialize, Serialize};

use crate::{CoveragePlane, QueueKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueueDiscipline {
    Serial,
    ParallelPerRank,
    LeaderThenBroadcast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ServePhase {
    PreServeBlocking,
    EarlyServeReady,
    DeferredPerformance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FanoutStrategy {
    None,
    BroadcastFromLeader,
    RebuildPerRank,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeExecutionContract {
    pub queue: QueueKind,
    pub discipline: QueueDiscipline,
    pub serve_phase: ServePhase,
    pub fanout_strategy: FanoutStrategy,
    pub dependency_barrier: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarmupContradiction {
    pub trigger: String,
    pub invalidates_node: String,
    pub next_action: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarmupCoverageProof {
    pub proof_id: String,
    pub node_name: String,
    pub region_name: String,
    pub plane: CoveragePlane,
    pub required_artifacts: Vec<String>,
    pub rank_scope: Vec<u16>,
    pub blocking: bool,
    pub expected_states: Vec<String>,
    pub contradiction_triggers: Vec<WarmupContradiction>,
    pub serve_phase: ServePhase,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WaveExecutionContract {
    pub wave_name: String,
    pub queue: QueueKind,
    pub discipline: QueueDiscipline,
    pub serve_phase: ServePhase,
    pub fulfills: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuntimeRoi {
    pub artifact_scope: String,
    pub compile_ms: u64,
    pub transfer_ms: u64,
    pub rebuild_ms: u64,
    pub preferred_strategy: FanoutStrategy,
}
