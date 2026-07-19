# sock Endpoint Pressure Benchmark

This is a standalone TMH pressure harness result. It does not modify or import TMH production runtime code.

- label: `tmh-accounting-runtime-kv`
- model: `qwen3-30b`
- endpoint: `http://127.0.0.1:8000`
- profile: `smoke`
- generated_at: `2026-07-19T08:04:07Z`
- elapsed_s: `359.5556`

## Token Preflight

| case | prompt tokens | original max new | effective max new | max model len |
| --- | ---: | ---: | ---: | ---: |
| `early_anchor_long_tail` | 466 | 96 | 96 | 2048 |
| `middle_anchor_detour` | 360 | 128 | 128 | 2048 |
| `late_anchor_control` | 241 | 96 | 96 | 2048 |
| `decoy_collision` | 280 | 128 | 128 | 2048 |
| `routing_table` | 278 | 128 | 128 | 2048 |
| `structured_records` | 290 | 160 | 160 | 2048 |
| `instruction_persistence` | 477 | 128 | 128 | 2048 |
| `multi_hop_bridge` | 252 | 160 | 160 | 2048 |
| `payload_dense` | 463 | 192 | 192 | 2048 |
| `long_generation_systems` | 169 | 384 | 384 | 2048 |

## Throughput

| case | category | concurrency | completion tok/s mean | wall s mean | contains target mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `early_anchor_long_tail` | `anchor_recall` | 1 | 28.2977 | 3.3925 | 100.0 |
| `early_anchor_long_tail` | `anchor_recall` | 2 | 34.3415 | 5.5909 | 100.0 |
| `early_anchor_long_tail` | `anchor_recall` | 4 | 66.8861 | 5.7411 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 1 | 27.8843 | 4.5904 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 2 | 32.6735 | 7.8351 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 4 | 62.404 | 8.2046 | 100.0 |
| `late_anchor_control` | `late_control` | 1 | 28.5213 | 3.3659 | 100.0 |
| `late_anchor_control` | `late_control` | 2 | 35.378 | 5.4271 | 100.0 |
| `late_anchor_control` | `late_control` | 4 | 66.6782 | 5.759 | 100.0 |
| `decoy_collision` | `confusable_recall` | 1 | 28.595 | 4.4763 | 100.0 |
| `decoy_collision` | `confusable_recall` | 2 | 32.8521 | 7.7925 | 100.0 |
| `decoy_collision` | `confusable_recall` | 4 | 60.3297 | 8.4867 | 100.0 |
| `routing_table` | `structured_lookup` | 1 | 28.2119 | 4.5371 | 100.0 |
| `routing_table` | `structured_lookup` | 2 | 32.3228 | 7.9201 | 100.0 |
| `routing_table` | `structured_lookup` | 4 | 60.3446 | 8.4846 | 100.0 |
| `structured_records` | `structured_lookup` | 1 | 27.9222 | 5.7302 | 0.0 |
| `structured_records` | `structured_lookup` | 2 | 35.4276 | 9.0325 | 0.0 |
| `structured_records` | `structured_lookup` | 4 | 57.8997 | 11.0536 | 25.0 |
| `instruction_persistence` | `instruction_retention` | 1 | 27.1146 | 4.7207 | 100.0 |
| `instruction_persistence` | `instruction_retention` | 2 | 30.145 | 8.4923 | 100.0 |
| `instruction_persistence` | `instruction_retention` | 4 | 58.2931 | 8.7832 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 1 | 27.5108 | 5.8159 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 2 | 32.7004 | 9.7858 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 4 | 61.8447 | 10.3485 | 100.0 |
| `payload_dense` | `dense_noise` | 1 | 26.8089 | 7.1618 | 100.0 |
| `payload_dense` | `dense_noise` | 2 | 33.3241 | 11.5232 | 100.0 |
| `payload_dense` | `dense_noise` | 4 | 64.0203 | 11.9962 | 100.0 |
| `long_generation_systems` | `long_generation` | 1 | 28.2867 | 13.5753 | 0.0 |
| `long_generation_systems` | `long_generation` | 2 | 38.2188 | 20.0948 | 0.0 |
| `long_generation_systems` | `long_generation` | 4 | 57.6116 | 26.6613 | 0.0 |

## Streaming TTFT Probes

| case | ttft s | elapsed s | completion tok/s |
| --- | ---: | ---: | ---: |
| `early_anchor_long_tail` | 0.0929 | 3.3896 | 28.322 |
| `middle_anchor_detour` | 0.1291 | 4.4387 | 28.8372 |
| `late_anchor_control` | 0.057 | 3.3144 | 28.9648 |
| `decoy_collision` | 0.1276 | 4.5183 | 28.3295 |
| `routing_table` | 0.1154 | 4.686 | 27.3156 |
| `structured_records` | 0.0877 | 4.62 | 27.7055 |
| `instruction_persistence` | 0.1511 | 4.8666 | 26.3017 |
| `multi_hop_bridge` | 0.1534 | 4.5268 | 28.2762 |
| `payload_dense` | 0.1554 | 4.6634 | 27.4475 |
| `long_generation_systems` | 0.1322 | 4.483 | 28.5524 |
