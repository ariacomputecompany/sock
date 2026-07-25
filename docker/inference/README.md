# SOCK Inference Container

This package builds a ROCm container that runs:

- `sock serve` as an OpenAI-compatible vLLM server on port `8000`
- `linx` as a public-link proxy on port `18110`

The default model is `Qwen/Qwen3-30B-A3B-GPTQ-Int4` with the production TMH layout:

```text
--kv-layout tmh
--tmh-hot-budget-pct 25
--attention-backend TRITON_ATTN
--gpu-memory-utilization 0.90
--max-num-batched-tokens 8192
--max-num-seqs 32
```

## Prerequisites

On the GPU host:

- Quilt runtime for local execution, or Docker Engine with Compose plugin
- ROCm host driver/runtime for the GPU
- access to `/dev/kfd` and `/dev/dri`
- the Strix Halo `gfx1151` torch wheel

For GMK `gfx1151`, copy the working torch wheel into the Docker build context:

```bash
mkdir -p docker/inference/wheelhouse
cp /home/deepsaint/wheelhouse/pytorch-gfx1151/torch-2.11.0+gfx1151-cp312-cp312-linux_x86_64.whl \
  docker/inference/wheelhouse/
```

The wheel is intentionally not committed to git.

## Configure

```bash
cp docker/inference/.env.example docker/inference/.env.inference
docker/inference/generate-api-key.sh
```

Put the generated key into `SOCK_API_KEY` if you want OpenAI-compatible bearer auth:

```text
SOCK_API_KEY=sk-sock-...
```

For multiple ad hoc keys, use comma-separated `SOCK_API_KEYS` instead:

```text
SOCK_API_KEYS=sk-sock-alice,sk-sock-bob
```

Set `LINX_PUBLIC_BASE_URL` to the public URL after DNS/domain setup, for example:

```text
LINX_PUBLIC_BASE_URL=https://api.example.com
```

Until a domain is attached, leave it as `http://<host>:18110`.

## Build And Run

Prepare `.env.inference` and keys without starting a container:

```bash
docker/inference/run-rocm.sh --prepare-only
```

Quilt path, preferred when a Quilt-compatible image/rootfs archive is already available:

```bash
QUILT_CLI=/home/deepsaint/work/quilt-oss/quilt-core/target/release/cli \
QUILT_IMAGE_PATH=/path/to/sock-inference-rocm-gfx1151.tar.gz \
docker/inference/run-quilt.sh
```

The Quilt path intentionally runs with host-leaning defaults and no explicit CPU
or memory cap so ROCm and the vLLM scheduler can drive the GPU as hard as the
model profile allows.

Docker/Compose fallback:

```bash
docker/inference/run-rocm.sh
```

Manual path:

```bash
docker compose -f docker/inference/compose.rocm.yml --env-file docker/inference/.env.inference build
docker compose -f docker/inference/compose.rocm.yml --env-file docker/inference/.env.inference up -d
docker logs -f sock-inference
```

The container prints:

- local SOCK base URL: `http://127.0.0.1:8000/v1`
- linx public URL: from `LINX_PUBLIC_BASE_URL`
- whether `SOCK_API_KEY` is enabled

## Smoke Test

Direct local vLLM/SOCK path:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${SOCK_API_KEY}"
```

Through linx, use the `public_url` printed in the logs. If `LINX_AUTH_MODE=public`,
the OpenAI-compatible API key is the main access control. For the default public
mode, the URL ends in `/linx/<service_id>/`, so append OpenAI paths after it:

```bash
curl -s "${LINX_PUBLIC_URL}v1/models" \
  -H "Authorization: Bearer ${SOCK_API_KEY}"
```

If `LINX_AUTH_MODE=service_token`, keep the `linx_token` query string printed in
the public URL. This is useful for short-lived links, but bearer auth is easier
for standard OpenAI SDK clients.

## OpenAI-Compatible Client

Use the linx URL as the base URL and pass the SOCK API key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.example.com/linx/<service_id>/v1",
    api_key="sk-sock-...",
)

print(client.models.list())
```

## Notes

- This image packages the current SOCK repository and clones/builds `linx`.
- Model weights are cached through the mounted Hugging Face cache volume.
- Runtime data and the linx SQLite DB live under `/data`.
- Throughput defaults are tuned for the live TMH production profile: `8192`
  context, `8192` batched tokens, `32` max sequences, and `0.90` GPU memory
  utilization. For frontier capacity tests, set `SOCK_MAX_MODEL_LEN=32768`.
- The GMK host currently needs either a prepared Quilt image/rootfs archive or
  Docker installed before this can be built/run there.
