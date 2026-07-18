from __future__ import annotations

import argparse
import json
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
        description="Run a measured inference request against a live sock endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--out", type=Path, default=Path("tmp/endpoint-inference.json"))
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the full response payload to stdout instead of a compact summary.",
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
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    return json.loads(body)


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    url = args.base_url.rstrip("/") + "/v1/completions"

    started = time.perf_counter()
    response = post_json(url, payload, args.timeout_s)
    elapsed_s = time.perf_counter() - started

    choice = response.get("choices", [{}])[0]
    text = choice.get("text", "")
    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    completion_tok_per_s = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
    total_tok_per_s = total_tokens / elapsed_s if elapsed_s > 0 else 0.0

    result = {
        "ok": True,
        "base_url": args.base_url,
        "model": args.model,
        "prompt": args.prompt,
        "elapsed_s": round(elapsed_s, 4),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "completion_tok_per_s": round(completion_tok_per_s, 4),
        "total_tok_per_s": round(total_tok_per_s, 4),
        "finish_reason": choice.get("finish_reason"),
        "response_text": text,
        "raw_response": response,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    stdout_result = result
    if not args.print_response:
        stdout_result = {
            "ok": True,
            "base_url": args.base_url,
            "model": args.model,
            "output_path": str(args.out),
            "elapsed_s": result["elapsed_s"],
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "completion_tok_per_s": result["completion_tok_per_s"],
            "total_tok_per_s": result["total_tok_per_s"],
            "finish_reason": choice.get("finish_reason"),
            "response_preview": text[:500],
        }
    print(json.dumps(stdout_result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
