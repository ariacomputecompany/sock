use anyhow::Result;
use sock_core::{BuildPlan, ShapeEnvelope, TargetEngine};
use sock_engine::vllm;

fn main() -> Result<()> {
    let plan = BuildPlan {
        engine: TargetEngine::Vllm,
        envelope: ShapeEnvelope::bounded("bootstrap"),
    };

    println!("sock workspace initialized");
    println!("plan engine: {}", plan.engine.as_str());
    println!("shape envelope: {}", plan.envelope.name);
    println!("vendored vLLM root: {}", vllm::root().display());
    println!("vendored vLLM revision: {}", vllm::revision());

    Ok(())
}
