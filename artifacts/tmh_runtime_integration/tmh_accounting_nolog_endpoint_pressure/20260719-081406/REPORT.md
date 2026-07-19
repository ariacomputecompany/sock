# sock Endpoint Pressure Benchmark

This is a standalone TMH pressure harness result. It does not modify or import TMH production runtime code.

- label: `tmh-accounting-runtime-kv-nolog`
- model: `qwen3-30b`
- endpoint: `http://127.0.0.1:8000`
- profile: `smoke`
- generated_at: `2026-07-19T08:14:06Z`
- elapsed_s: `354.5221`

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
| `early_anchor_long_tail` | `anchor_recall` | 1 | 29.1545 | 3.2928 | 100.0 |
| `early_anchor_long_tail` | `anchor_recall` | 2 | 34.3335 | 5.5922 | 100.0 |
| `early_anchor_long_tail` | `anchor_recall` | 4 | 69.6081 | 5.5166 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 1 | 28.037 | 4.5654 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 2 | 32.7098 | 7.8264 | 100.0 |
| `middle_anchor_detour` | `anchor_recall` | 4 | 64.0216 | 7.9973 | 100.0 |
| `late_anchor_control` | `late_control` | 1 | 29.0241 | 3.3076 | 100.0 |
| `late_anchor_control` | `late_control` | 2 | 34.4419 | 5.5746 | 100.0 |
| `late_anchor_control` | `late_control` | 4 | 69.4106 | 5.5323 | 100.0 |
| `decoy_collision` | `confusable_recall` | 1 | 27.9922 | 4.5727 | 100.0 |
| `decoy_collision` | `confusable_recall` | 2 | 33.0459 | 7.7468 | 100.0 |
| `decoy_collision` | `confusable_recall` | 4 | 65.6049 | 7.8043 | 100.0 |
| `routing_table` | `structured_lookup` | 1 | 28.1833 | 4.5417 | 100.0 |
| `routing_table` | `structured_lookup` | 2 | 42.7872 | 5.9831 | 100.0 |
| `routing_table` | `structured_lookup` | 4 | 64.3111 | 7.9613 | 100.0 |
| `structured_records` | `structured_lookup` | 1 | 28.6271 | 5.5891 | 0.0 |
| `structured_records` | `structured_lookup` | 2 | 34.4356 | 9.2927 | 0.0 |
| `structured_records` | `structured_lookup` | 4 | 61.4339 | 10.4177 | 25.0 |
| `instruction_persistence` | `instruction_retention` | 1 | 26.091 | 4.9059 | 100.0 |
| `instruction_persistence` | `instruction_retention` | 2 | 30.91 | 8.2821 | 100.0 |
| `instruction_persistence` | `instruction_retention` | 4 | 58.6356 | 8.7319 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 1 | 28.6323 | 5.5881 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 2 | 32.4972 | 9.847 | 100.0 |
| `multi_hop_bridge` | `multi_hop` | 4 | 64.0852 | 9.9867 | 100.0 |
| `payload_dense` | `dense_noise` | 1 | 27.2117 | 7.0558 | 100.0 |
| `payload_dense` | `dense_noise` | 2 | 33.1297 | 11.5908 | 100.0 |
| `payload_dense` | `dense_noise` | 4 | 63.1865 | 12.1545 | 100.0 |
| `long_generation_systems` | `long_generation` | 1 | 27.4243 | 14.0022 | 0.0 |
| `long_generation_systems` | `long_generation` | 2 | 39.3362 | 19.524 | 0.0 |
| `long_generation_systems` | `long_generation` | 4 | 58.3296 | 26.3331 | 0.0 |

## Streaming TTFT Probes

| case | ttft s | elapsed s | completion tok/s |
| --- | ---: | ---: | ---: |
| `early_anchor_long_tail` | 0.0967 | 3.5049 | 27.3906 |
| `middle_anchor_detour` | 0.1345 | 4.6149 | 27.7361 |
| `late_anchor_control` | 0.0705 | 3.4658 | 27.6994 |
| `decoy_collision` | 0.1329 | 4.7918 | 26.7124 |
| `routing_table` | 0.1192 | 4.6109 | 27.7604 |
| `structured_records` | 0.0871 | 4.4928 | 28.4898 |
| `instruction_persistence` | 0.1467 | 4.6427 | 27.5704 |
| `multi_hop_bridge` | 0.1428 | 4.5771 | 27.9653 |
| `payload_dense` | 0.1597 | 4.7111 | 27.1699 |
| `long_generation_systems` | 0.1335 | 4.6662 | 27.4315 |
