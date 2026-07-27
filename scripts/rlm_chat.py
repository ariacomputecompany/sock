#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


DEFAULT_SYSTEM_PROMPT = """You are a production coding agent in chat mode.
Help the user design and build software end to end. You can reason about these
RLM tool intents when useful: read_file(path), bash(command), scratchpad(note),
repo_index(repo), write_code(path, content), run_tests(command),
aegis_search(query), memory_save(fact), deploy_plan(service, steps, rollback),
approval_pause(risk, reason). For normal user chat, answer directly in clear
prose. Do not emit dataset markers, metadata tags, raw harness labels, or JSON
unless the user explicitly asks for JSON or a tool action."""

DEFAULT_STOPS = [
    "<|repo_name|>",
    "<|file_path|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the RLM LoRA SOCK endpoint")
    parser.add_argument("prompt", nargs="*", help="Prompt text. Omit for stdin.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--model", default="qwen5b-grown-v3-step1316-rlm")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or sys.stdin.read().strip()
    if not prompt:
        print("rlm_chat.py: prompt is required", file=sys.stderr)
        return 64
    payload = {
        "model": args.model,
        "stream": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": DEFAULT_STOPS,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
            if chunk:
                print(chunk, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
