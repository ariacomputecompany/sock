from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def repeated_topic(topic: str, repeats: int) -> str:
    seed = (
        f"{topic}. Include concrete mechanisms, edge cases, implementation "
        "details, tradeoffs, and a concise conclusion."
    )
    return " ".join(seed for _ in range(repeats))


@dataclass(frozen=True)
class BenchCase:
    name: str
    prompt: str
    max_tokens: int
    temperature: float = 0.2


DEFAULT_CASES = [
    BenchCase(
        name="tiny_fact_64",
        prompt="In two paragraphs, explain why the sky appears blue.",
        max_tokens=64,
    ),
    BenchCase(
        name="short_codegen_128",
        prompt=(
            "Write a compact Python function that computes a rolling mean for a "
            "list of floats, then explain the complexity."
        ),
        max_tokens=128,
    ),
    BenchCase(
        name="medium_architecture_256",
        prompt=(
            "Design a production-safe job queue for GPU inference requests. "
            "Cover scheduling, retries, cancellation, backpressure, metrics, "
            "and failure isolation."
        ),
        max_tokens=256,
    ),
    BenchCase(
        name="long_cosmology_512",
        prompt=(
            "Explain how the universe came into being. Give a clear, rigorous, "
            "long-form explanation that covers the Big Bang, inflation, early "
            "particle formation, nucleosynthesis, recombination, stars, "
            "galaxies, and what is still unknown."
        ),
        max_tokens=512,
    ),
    BenchCase(
        name="long_context_summary_256",
        prompt=(
            repeated_topic(
                "We are evaluating an AMD ROCm inference stack against upstream "
                "vLLM on a WSL machine",
                18,
            )
            + " Summarize the operational risks and benchmark methodology."
        ),
        max_tokens=256,
    ),
    BenchCase(
        name="extended_generation_768",
        prompt=(
            "Write a technical essay on how to build a deterministic, "
            "least-dependency LLM serving runtime for heterogeneous GPU fleets. "
            "Discuss AMD, NVIDIA, build isolation, wheel selection, runtime "
            "feature detection, endpoint compatibility, and benchmark design."
        ),
        max_tokens=768,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-case benchmark suite against an OpenAI-compatible endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--concurrency-levels",
        default="1,2,4",
        help="Comma-separated request concurrency levels per case.",
    )
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in DEFAULT_CASES],
        help="Run only the named case. Can be repeated.",
    )
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, round((pct / 100) * (len(sorted_values) - 1))))
    return sorted_values[idx]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p90": round(percentile(values, 90), 4),
    }


def run_one(
    *,
    url: str,
    model: str,
    case: BenchCase,
    timeout_s: int,
    run_index: int,
    request_index: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": case.prompt,
        "max_tokens": case.max_tokens,
        "temperature": case.temperature,
    }
    started = time.perf_counter()
    response = post_json(url, payload, timeout_s)
    elapsed_s = time.perf_counter() - started
    choice = response.get("choices", [{}])[0]
    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    return {
        "run_index": run_index,
        "request_index": request_index,
        "elapsed_s": round(elapsed_s, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tok_per_s": round(completion_tokens / elapsed_s, 4)
        if elapsed_s > 0
        else 0.0,
        "total_tok_per_s": round(total_tokens / elapsed_s, 4) if elapsed_s > 0 else 0.0,
        "finish_reason": choice.get("finish_reason"),
        "response_text": choice.get("text", ""),
        "raw_response": response,
    }


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    requests = batch["requests"]
    wall_s = batch["wall_s"]
    completion_tokens = sum(request["completion_tokens"] for request in requests)
    total_tokens = sum(request["total_tokens"] for request in requests)
    return {
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "aggregate_completion_tok_per_s": round(completion_tokens / wall_s, 4)
        if wall_s > 0
        else 0.0,
        "aggregate_total_tok_per_s": round(total_tokens / wall_s, 4)
        if wall_s > 0
        else 0.0,
        "per_request_completion_tok_per_s": summarize(
            [request["completion_tok_per_s"] for request in requests]
        ),
        "per_request_elapsed_s": summarize([request["elapsed_s"] for request in requests]),
    }


def run_batch(
    *,
    url: str,
    model: str,
    case: BenchCase,
    timeout_s: int,
    run_index: int,
    concurrency: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                url=url,
                model=model,
                case=case,
                timeout_s=timeout_s,
                run_index=run_index,
                request_index=request_index + 1,
            )
            for request_index in range(concurrency)
        ]
        requests = [future.result() for future in futures]
    wall_s = round(time.perf_counter() - started, 4)
    batch = {
        "run_index": run_index,
        "concurrency": concurrency,
        "wall_s": wall_s,
        "requests": requests,
    }
    batch["summary"] = summarize_batch(batch)
    return batch


def summarize_case(case_result: dict[str, Any]) -> dict[str, Any]:
    by_concurrency: dict[str, Any] = {}
    for concurrency, batches in case_result["batches_by_concurrency"].items():
        batch_summaries = [batch["summary"] for batch in batches]
        by_concurrency[str(concurrency)] = {
            "aggregate_completion_tok_per_s": summarize(
                [summary["aggregate_completion_tok_per_s"] for summary in batch_summaries]
            ),
            "aggregate_total_tok_per_s": summarize(
                [summary["aggregate_total_tok_per_s"] for summary in batch_summaries]
            ),
            "wall_s": summarize([summary["wall_s"] for summary in batch_summaries]),
            "completion_tokens": summarize(
                [summary["completion_tokens"] for summary in batch_summaries]
            ),
            "total_tokens": summarize([summary["total_tokens"] for summary in batch_summaries]),
        }
    return by_concurrency


def main() -> int:
    args = parse_args()
    concurrency_levels = [
        int(item.strip()) for item in args.concurrency_levels.split(",") if item.strip()
    ]
    if not concurrency_levels or any(level < 1 for level in concurrency_levels):
        raise ValueError("--concurrency-levels must contain positive integers")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")

    selected_names = set(args.case or [case.name for case in DEFAULT_CASES])
    cases = [case for case in DEFAULT_CASES if case.name in selected_names]
    url = args.base_url.rstrip("/") + "/v1/completions"

    started = time.perf_counter()
    case_results = []
    for case in cases:
        warmups = [
            run_batch(
                url=url,
                model=args.model,
                case=case,
                timeout_s=args.timeout_s,
                run_index=index + 1,
                concurrency=1,
            )
            for index in range(args.warmup_runs)
        ]
        batches_by_concurrency: dict[int, list[dict[str, Any]]] = {}
        for concurrency in concurrency_levels:
            batches_by_concurrency[concurrency] = [
                run_batch(
                    url=url,
                    model=args.model,
                    case=case,
                    timeout_s=args.timeout_s,
                    run_index=index + 1,
                    concurrency=concurrency,
                )
                for index in range(args.runs)
            ]
        case_result = {
            "name": case.name,
            "prompt": case.prompt,
            "max_tokens": case.max_tokens,
            "temperature": case.temperature,
            "warmups": warmups,
            "batches_by_concurrency": batches_by_concurrency,
        }
        case_result["summary"] = summarize_case(case_result)
        case_results.append(case_result)

    elapsed_s = round(time.perf_counter() - started, 4)
    result = {
        "ok": True,
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "concurrency_levels": concurrency_levels,
        "elapsed_s": elapsed_s,
        "cases": case_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    compact = {
        "ok": True,
        "label": args.label,
        "model": args.model,
        "output_path": str(args.out),
        "elapsed_s": elapsed_s,
        "cases": {
            case["name"]: case["summary"] for case in case_results
        },
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
