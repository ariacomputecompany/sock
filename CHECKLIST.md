# sock checklist

This is the active sober implementation checklist for the remaining Lane A work.
It is intentionally one-by-one, production-oriented, and limited to the highest-confidence build-safe path.

Scope:

- production `sock`
- modular engine architecture
- deep V1 focus on vendored `vLLM`
- V1 deployment target: NVIDIA on Linux
- north star: deterministic partial builds, fast cold start, great operator DX

Working rule:

- execute in order unless a later item becomes a strict prerequisite
- remove completed items instead of annotating them
- treat any runtime-adjacent drift as an escalation out of Lane A
