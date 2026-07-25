# KV Layouts

sock treats KV layout as a first-class runtime axis, separate from attention
backend selection.

## Public CLI

Use the canonical `sock` command:

```bash
sock serve <model> --kv-layout standard
sock serve <model> --kv-layout tmh
```

`standard` selects regular vLLM paged KV. `tmh` selects the Tiered Memory
Hierarchy physical KV path on supported accelerator/runtime combinations.

## Production Contract

The layout contract has three separate concerns:

- KV layout backend: storage identity, page lifecycle, residency policy, and
  physical-storage capability.
- Attention backend: compute implementation such as FlashInfer or Triton.
- Compatibility resolver: device/runtime/backend validation that fails closed
  before serving.

The current production support matrix is:

| Layout | Public value | Runtime mode | CUDA | ROCm | Physical storage |
| --- | --- | --- | --- | --- | --- |
| Standard paged KV | `standard` | standard | yes | yes | yes |
| TMH fidelity paged KV | `tmh` | physical | yes | yes | yes, runtime-gated |

Physical TMH must not silently store standard KV while reporting TMH. If the
detected runtime cannot support the physical TMH backend, startup must fail
before inference.
