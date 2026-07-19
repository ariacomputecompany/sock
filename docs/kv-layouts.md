# KV Layouts

sock treats KV layout as a first-class runtime axis, separate from attention
backend selection.

## Public CLI

Use the canonical `sock` command:

```bash
sock serve <model> --kv-layout standard
sock serve <model> --kv-layout tmh
```

`standard` selects regular vLLM paged KV. `tmh` selects the Transformer Memory
Hierarchy fidelity-paged layout contract and maps to allocator-path accounting
until physical mixed-fidelity storage is implemented.

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
| TMH fidelity paged KV | `tmh` | accounting | yes | yes | no, fail-closed |

Physical TMH must not silently store standard KV while reporting TMH. Until the
mixed-fidelity warm-page tensors and layout-aware attention kernels exist,
physical mode is rejected before inference startup.
