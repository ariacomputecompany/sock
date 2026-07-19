# sock Endpoint Pressure Benchmark

This is a standalone TMH pressure harness result. It does not modify or import TMH production runtime code.

- label: `regular-runtime-kv`
- model: `qwen3-30b`
- endpoint: `http://127.0.0.1:8000`
- profile: `smoke`
- generated_at: `2026-07-19T07:57:02Z`
- elapsed_s: `349.9299`

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
| `early_anchor_long_tail` | `anchor_recall` | 1 | 28.4993 | 3.3685 | 100.0 |
| `early_anchor_long_tail` | `anchor_recall` | 2 | 34.9931 | 5.4868 | 100.0 |
| `early_anchor_long_tail` | `anchor_recall` | 4 | 69.2004 | 5.5491 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 1 | 28.8639 | 4.4346 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 2 | 32.6119 | 7.8499 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 4 | 64.0745 | 7.9907 | 100.0 |
| `late_anchor_control` | `late_control` | 1 | 28.9436 | 3.3168 | 100.0 |
| `late_anchor_control` | `late_control` | 2 | 34.7606 | 5.5235 | 100.0 |
| `late_anchor_control` | `late_control` | 4 | 70.1075 | 5.4773 | 100.0 |
| `decoy_collision` | `confusable_recall` | 1 | 27.8467 | 4.5966 | 100.0 |
| `decoy_collision` | `confusable_recall` | 2 | 33.2282 | 7.7043 | 100.0 |
| `decoy_collision` | `confusable_recall` | 4 | 67.1387 | 7.626 | 100.0 |
| `routing_table` | `structured_lookup` | 1 | 29.1971 | 4.384 | 100.0 |
| `routing_table` | `structured_lookup` | 2 | 42.5581 | 6.0153 | 100.0 |
| `routing_table` | `structured_lookup` | 4 | 60.993 | 8.3944 | 100.0 |
| `structured_records` | `structured_lookup` | 1 | 29.0244 | 5.5126 | 0.0 |
| `structured_records` | `structured_lookup` | 2 | 34.6245 | 9.242 | 0.0 |
| `structured_records` | `structured_lookup` | 4 | 61.4174 | 10.4205 | 50.0 |
| `instruction_persistence` | `instruction_retention` | 1 | 27.897 | 4.5883 | 100.0 |
| `instruction_persistence` | `instruction_retention` | 2 | 31.1784 | 8.2108 | 100.0 |
| `instruction_persistence` | `instruction_retention` | 4 | 60.4636 | 8.4679 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 1 | 29.1939 | 5.4806 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 2 | 43.245 | 7.3997 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 4 | 66.0461 | 9.6902 | 100.0 |
| `payload_dense` | `dense_noise` | 1 | 27.8762 | 6.8876 | 100.0 |
| `payload_dense` | `dense_noise` | 2 | 35.5964 | 10.7876 | 100.0 |
| `payload_dense` | `dense_noise` | 4 | 68.6677 | 11.1843 | 100.0 |
| `long_generation_systems` | `long_generation` | 1 | 28.2075 | 13.6134 | 0.0 |
| `long_generation_systems` | `long_generation` | 2 | 32.1416 | 23.8943 | 0.0 |
| `long_generation_systems` | `long_generation` | 4 | 60.9703 | 25.1926 | 0.0 |

## Streaming TTFT Probes

| case | ttft s | elapsed s | completion tok/s |
| --- | ---: | ---: | ---: |
| `early_anchor_long_tail` | 0.0825 | 3.3946 | 28.2804 |
| `middle_anchor_detour` | 0.1328 | 4.6583 | 27.4781 |
| `late_anchor_control` | 0.0563 | 3.2312 | 29.7101 |
| `decoy_collision` | 0.1189 | 4.6585 | 27.4769 |
| `routing_table` | 0.1185 | 4.4543 | 28.7365 |
| `structured_records` | 0.0857 | 4.517 | 28.3376 |
| `instruction_persistence` | 0.1508 | 4.5862 | 27.9095 |
| `multi_hop_bridge` | 0.151 | 4.5803 | 27.9456 |
| `payload_dense` | 0.1487 | 4.3307 | 29.5564 |
| `long_generation_systems` | 0.1342 | 4.5303 | 28.254 |
