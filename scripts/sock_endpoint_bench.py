from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Explain how the universe came into being. Give a clear, rigorous, "
    "long-form explanation that covers the Big Bang, inflation, early particle "
    "formation, nucleosynthesis, recombination, stars, galaxies, and what is "
    "still unknown."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a live OpenAI-compatible completion endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--out", type=Path, default=Path("tmp/endpoint-bench.json"))
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


def run_completion(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_s: int,
    measured: bool,
    index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(url, payload, timeout_s)
    elapsed_s = time.perf_counter() - started
    choice = response.get("choices", [{}])[0]
    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    return {
        "index": index,
        "measured": measured,
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


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    url = args.base_url.rstrip("/") + "/v1/completions"

    warmups = [
        run_completion(
            url=url,
            payload=payload,
            timeout_s=args.timeout_s,
            measured=False,
            index=index + 1,
        )
        for index in range(args.warmup_runs)
    ]
    runs = [
        run_completion(
            url=url,
            payload=payload,
            timeout_s=args.timeout_s,
            measured=True,
            index=index + 1,
        )
        for index in range(args.runs)
    ]

    completion_tps = [run["completion_tok_per_s"] for run in runs]
    total_tps = [run["total_tok_per_s"] for run in runs]
    elapsed = [run["elapsed_s"] for run in runs]
    result = {
        "ok": True,
        "base_url": args.base_url,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmup_runs": len(warmups),
        "measured_runs": len(runs),
        "summary": {
            "elapsed_s": summarize(elapsed),
            "completion_tok_per_s": summarize(completion_tps),
            "total_tok_per_s": summarize(total_tps),
            "completion_tokens": summarize([run["completion_tokens"] for run in runs]),
            "total_tokens": summarize([run["total_tokens"] for run in runs]),
        },
        "warmups": warmups,
        "runs": runs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(args.out),
                "model": args.model,
                "measured_runs": len(runs),
                "summary": result["summary"],
                "response_preview": runs[-1]["response_text"][:500],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
