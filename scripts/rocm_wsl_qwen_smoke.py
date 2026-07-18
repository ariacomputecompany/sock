from __future__ import annotations

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.platforms.rocm import rocm_custom_paged_attention_rejection_reasons

try:
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - optional runtime aid
    AutoConfig = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small ROCm/WSL smoke test against vendored vLLM."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id to load for the smoke run.",
    )
    parser.add_argument(
        "--prompt",
        default="Write one short sentence saying the ROCm smoke test passed.",
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model context length for the smoke run.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="Target GPU memory utilization for the vLLM engine.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of untimed warmup generations to run before measuring throughput.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a compact machine-readable summary after the smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paged_attention_summary: dict[str, object] | None = None

    if AutoConfig is not None:
        cfg = AutoConfig.from_pretrained(args.model)
        hidden_size = getattr(cfg, "hidden_size", None)
        num_attention_heads = getattr(cfg, "num_attention_heads", None)
        num_key_value_heads = getattr(cfg, "num_key_value_heads", None)
        sliding_window = getattr(cfg, "sliding_window", None)
        if hidden_size and num_attention_heads and num_key_value_heads:
            head_size = hidden_size // num_attention_heads
            gqa_ratio = num_attention_heads // num_key_value_heads
            reasons = rocm_custom_paged_attention_rejection_reasons(
                qtype=getattr(cfg, "dtype", getattr(cfg, "torch_dtype", None)),
                head_size=head_size,
                block_size=16,
                gqa_ratio=gqa_ratio,
                max_seq_len=min(args.max_model_len, getattr(cfg, "max_position_embeddings", args.max_model_len)),
                sliding_window=0 if sliding_window is None else sliding_window,
                kv_cache_dtype="auto",
            )
            paged_attention_summary = {
                "model": args.model,
                "head_size": head_size,
                "gqa_ratio": gqa_ratio,
                "sliding_window": sliding_window,
                "eligible": not reasons,
                "reasons": list(reasons),
            }
            print("paged_attention_check", paged_attention_summary, flush=True)

    print("starting_llm", flush=True)
    start = time.perf_counter()
    llm = LLM(
        model=args.model,
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    init_s = time.perf_counter() - start
    print(f"engine_ready init_s={init_s:.2f}", flush=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    for warmup_idx in range(args.warmup_iters):
        llm.generate([args.prompt], sampling_params)
        print(f"warmup_complete iteration={warmup_idx + 1}", flush=True)

    gen_start = time.perf_counter()
    output = llm.generate([args.prompt], sampling_params)[0]
    gen_s = time.perf_counter() - gen_start
    completion = output.outputs[0]
    generated_tokens = len(getattr(completion, "token_ids", []) or [])
    tok_per_s = generated_tokens / gen_s if gen_s > 0 else 0.0
    print(
        f"generation_ready gen_s={gen_s:.2f} generated_tokens={generated_tokens} tok_s={tok_per_s:.2f}",
        flush=True,
    )
    print("OUTPUT_START", flush=True)
    print(completion.text, flush=True)
    print("OUTPUT_END", flush=True)
    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "prompt": args.prompt,
                    "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "warmup_iters": args.warmup_iters,
                    "vllm_use_v2_model_runner": os.environ.get(
                        "VLLM_USE_V2_MODEL_RUNNER"
                    ),
                    "wsl_pin_memory_env": os.environ.get(
                        "VLLM_WSL2_ENABLE_PIN_MEMORY"
                    ),
                    "engine_init_s": round(init_s, 4),
                    "generation_s": round(gen_s, 4),
                    "generated_tokens": generated_tokens,
                    "tok_per_s": round(tok_per_s, 4),
                    "output_text": completion.text,
                    "paged_attention": paged_attention_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
