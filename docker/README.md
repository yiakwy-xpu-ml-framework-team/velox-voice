# VeloxVoice ASR — Docker

One-command build & serve of the SGLang-Omni-compatible ASR server.

## Quick start

```bash
# working GPU defaults to 0
bash docker/run.sh

# the current H800 test box uses GPU 5
VELOX_GPU=5 bash docker/run.sh
```

`run.sh` builds the image, starts the container, waits for `/health`, and prints
the endpoint. Equivalent raw commands:

```bash
VELOX_GPU=5 docker compose -f docker/docker-compose.yml up --build -d
docker compose -f docker/docker-compose.yml logs -f
docker compose -f docker/docker-compose.yml down
```

Service endpoints (host port = `VELOX_PORT`, default 8000):

| route | purpose |
|---|---|
| `GET /health` | readiness probe (used by the healthcheck) |
| `GET /docs` | Swagger UI |
| `GET /client` | browser test client |
| `POST /v1/audio/transcriptions` | multipart transcription |
| `WS  /v1/audio/ws` | streaming / pooled partials |

## GPU selection

The container is given **all** host GPUs (`count: all`); the working device is
chosen with `VELOX_GPU` (a physical PCI index), which the entrypoint exports as
`CUDA_VISIBLE_DEVICES` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` — the same convention
as `scripts/run_server_background.sh`.

- default `VELOX_GPU=0`
- the current test box: `VELOX_GPU=5`

## Configuration (env)

| variable | default | meaning |
|---|---|---|
| `VELOX_GPU` | `0` | host GPU index → `CUDA_VISIBLE_DEVICES` |
| `VELOX_PORT` | `8000` | host + container HTTP port |
| `VELOX_MODEL_DIR` | `/models/asr_model` | model path inside the container |
| `MODEL_HOST_DIR` | `../data/models/asr_model` | host path bind-mounted read-only |
| `VELOX_HOST` | `0.0.0.0` | bind address |
| `VELOXVOICE_KERNEL_CACHE` | `/var/cache/veloxvoice` | nvcc/TVM-FFI JIT cache (named volume) |
| `BASE_IMAGE` | `nvidia/cuda:13.0.0-devel-ubuntu24.04` | CUDA base (build arg) |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cu130` | torch wheel index (build arg) |

Copy `docker/.env.example` to `docker/.env` to persist local choices (`run.sh`
sources it). `.env` is git-ignored.

## Notes

- **Multi-replica / scaled** (N inference processes behind one entry port, API +
  webpage + WS all on :8000): see [`multi_instances/`](multi_instances/README.md)
  — `VELOX_GPU=5 REPLICAS=4 bash docker/multi_instances/run.sh`, then
  `bash docker/multi_instances/run.sh scale 6`.

- **CUDA devel base is required**: VeloxVoice compiles its flash-float JIT
  kernels at first run (TVM-FFI `load_inline` → nvcc). The `devel` image carries
  the toolkit; the JIT cache lands in the `veloxvoice-kernel-cache` volume so
  later starts are fast. First start can take a few minutes (model load + JIT).
- The model bundle is **never** baked in — it is bind-mounted read-only.
  Bump the compose healthcheck `start_period` if your storage is slow.
- `shm_size: 8gb` is set for the CUDA graphs / multiprocessing paths.
- CUDA 12.8 fallback (compatible to 580+ driver and cu13 API) :
  `BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04 TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash docker/run.sh`

## Troubleshooting

- **`count: all` unsupported** (Compose < 2.30): replace the `deploy:` block with
  the classic `docker run --gpus all -e VELOX_GPU=5 -p 8000:8000 ...`, or set
  `count: 1` and `VELOX_GPU=0` (single visible GPU becomes index 0).
- **Image/wheel pull fails** (offline registry): override `BASE_IMAGE` /
  `TORCH_INDEX`, or pre-pull the CUDA base and torch wheels into your mirror.
- **`model dir ... missing train.yaml`**: the model wasn't mounted — check
  `MODEL_HOST_DIR` / `docker/.env` and that the host path is the ASR bundle root.
- **First request is slow**: JIT kernels compile on first use; subsequent starts
  reuse the `veloxvoice-kernel-cache` volume.

