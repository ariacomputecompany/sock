# TMH Runtime Integration Bench

Matched endpoint smoke pressure runs on Qwen/Qwen3-30B-A3B-GPTQ-Int4 via `sock serve` on ROCm WSL.

## Runs

| mode | artifact | wall s | TTFT mean s | target retention mean | c1 mean tok/s | c2 mean tok/s | c4 mean tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `regular` | `artifacts/tmh_runtime_integration/regular_endpoint_pressure/20260719-075702/result.json` | 349.9299 | 0.1179 | 81.667% | 28.555 | 35.4938 | 64.9079 |
| `tmh_accounting_log` | `artifacts/tmh_runtime_integration/tmh_accounting_endpoint_pressure/20260719-080407/result.json` | 359.5556 | 0.1202 | 80.833% | 27.9153 | 33.7384 | 61.6312 |
| `tmh_accounting_nolog` | `artifacts/tmh_runtime_integration/tmh_accounting_nolog_endpoint_pressure/20260719-081406/result.json` | 354.5221 | 0.1224 | 80.833% | 28.0377 | 34.7627 | 63.8627 |

## Deltas Vs Regular

| mode | wall delta | TTFT mean delta | c1 tok/s delta | c2 tok/s delta | c4 tok/s delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tmh_accounting_log` | 2.751% | 1.951% | -2.24% | -4.946% | -5.048% |
| `tmh_accounting_nolog` | 1.312% | 3.817% | -1.812% | -2.06% | -1.61% |

## Allocator-Path TMH Pressure

- TMH allocation log lines: `14016`
- old-KV pressure rows: `14016`
- old/warm reduction floor vs same-hot uniform-int8 old KV: `16.667%`
- old/warm reduction mean: `16.667%`
- total effective reduction floor vs same-hot uniform-int8 total KV: `7.407%`
- total effective reduction mean: `9.102%`

## Readout

- TMH is now wired into the live vLLM allocator path behind `--tmh-kv-policy accounting` and records pressure during real endpoint traffic.
- `--tmh-kv-policy physical` fails closed until mixed-fidelity warm-page tensors and attention kernels exist, so these results do not claim physical TMH speedup yet.
- No-log accounting overhead is small in wall clock for this smoke suite; allocation logging is intentionally slower and should remain a diagnostic mode.
