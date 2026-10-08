# VeloxVoice ASR — dynamically scaled replicas behind one port

Scales the ASR service to N independent inference processes (each ~4-5 GiB of
GPU memory) fronted by a round-robin load balancer on the single entry port
**8000**. The API, the WebSocket streaming lane **and the test webpage** all go
through the same port, so nothing changes for clients:

```
:8000  lb (HTTP + WS round-robin)  ─┬─ asr replica 1
                                    ├─ asr replica 2   (share GPU = VELOX_GPU)
                                    └─ asr replica N
```

| client URL | what |
|---|---|
| `http://localhost:8000/v1/audio/transcriptions` | API |
| `http://localhost:8000/client` | browser test webpage (proxied unchanged) |
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/health` | aggregate readiness (`replicas`, `healthy`) |
| `ws://localhost:8000/v1/audio/ws` | streaming lane |

## One-command build & serve

```bash
# 2 replicas on GPU 0 (default)
bash docker/multi_instances/run.sh

# H800 test box: 4 replicas on GPU 5
VELOX_GPU=5 REPLICAS=4 bash docker/multi_instances/run.sh
```

Then, at any time, scale dynamically — the balancer re-resolves the replica
set every few seconds, no restart:

```bash
bash docker/multi_instances/run.sh scale 6
```

Stop / logs:

```bash
bash docker/multi_instances/run.sh stop
bash docker/multi_instances/run.sh logs
```

> **Stopping gotcha:** this stack is Compose project **`veloxvoice-multi`**, so a
> bare `docker compose down` run from `docker/` targets the *single-instance*
> project (`veloxvoice`) and does nothing here. Stop it correctly with the
> `run.sh stop` above, or from any directory:
> `docker compose -p veloxvoice-multi down`.
> `run.sh stop` also force-removes any orphaned containers by project label as a
> fallback.

## Verify (stateful two-step flow)

One command reproduces exactly what the webpage does — upload, then transcribe
by `audio_id` — and fails loudly if the balancer misroutes it:

```bash
bash docker/multi_instances/run.sh verify
# GET /client -> 200
# upload id=<...>
# PASS: streamed to transcript.text.done
```

> **After any change to the source or to `asr_lb.py`, rebuild** — the LB script
> is baked into the image (`COPY . /app`). `run.sh up` always rebuilds
> (`--build`) and recreates the containers; a plain `scale` does not.
> `bash docker/multi_instances/run.sh logs | grep -E 'remember|pin'` shows the
> affinity decisions.

## Why replicas (not in-process workers)

The `Velox` engine holds one global lock, so a single process serializes
inference. Throughput scales by running several processes and load-balancing —
this is exactly the 1-GPU / 3-worker topology benchmarked in
`benchmark/BENCHMARK_GUIDE.md` (~42 req/s on a 99.6 s clip; GPU-compute bound at
that point, so more replicas mainly cut queue latency).

Replicas **share one GPU** (`VELOX_GPU`): budget ~4-5 GiB per replica, so an
80 GB card fits ~14. For production put each replica on its own GPU by running
one `docker run` per GPU against the same `docker/Dockerfile`.

## How the balancer works

`asr_lb.py` resolves the Compose service name `asr:8000` to **all** replica IPs
via the embedded DNS, refreshing every `--ttl` seconds (default 5), and routes:

- HTTP bodies are buffered and forwarded; responses are streamed back (SSE and
  `stream=true` pass through unchanged).
- A dead replica is skipped (retry the next) — a rolling `scale` is seamless.
- `/health` is an aggregate probe over all replicas (`{"replicas": N,
  "healthy": M}`).
- `/client`, `/docs`, `/openapi.json` and `WS /v1/audio/ws` are proxied as-is.

### Session affinity (why the webpage needs it)

The ASR service is **stateful for the webpage's two-step flow**:

1. `POST /v1/audio/uploads` stores the file **in that process's memory** and
   returns `{id}`;
2. `POST /v1/audio/transcriptions` with `audio_id=<id>` looks it up **in the
   same process**.

Plain round-robin sends (2) to another replica → **HTTP 404 → the browser shows
"transcribing failed"**. The balancer prevents that two ways:

- **Sticky cookie** `velox_lb=<replica>`: set on the first response; the browser
  sends it on every later request, pinning that client to one replica.
- **audio_id affinity**: the balancer reads the `{id}` out of each upload reply
  and remembers `audio_id → replica`, so even cookie-less API clients (curl)
  using the two-step flow route back correctly.

Caveats: affinity lives in the balancer's memory (TTL 600 s, capped at 10k ids),
and a replica restart drops its uploaded ids — the client should upload again.
Single-call API use (`file=@...` directly on `/v1/audio/transcriptions`) is
stateless and needs neither mechanism.

## Logs & status

This project is named **`veloxvoice-multi`**, so logs must be asked for with its
own compose file (a bare `docker compose logs` from `docker/` targets the
single-instance project `veloxvoice` and shows nothing):

```bash
bash docker/multi_instances/run.sh logs          # all services, timestamped
bash docker/multi_instances/run.sh logs asr      # every replica
bash docker/multi_instances/run.sh logs asr-1    # one replica (asr-1..asr-N)
bash docker/multi_instances/run.sh logs lb       # the balancer only
# equivalently:
docker compose -f docker/multi_instances/docker-compose.yml logs -f --timestamps
docker logs -f veloxvoice-multi-asr-1
docker logs -f veloxvoice-multi-lb-1

bash docker/multi_instances/run.sh status        # container list + aggregate /health
```

Any other compose subcommand, correct file + project handled for you:

```bash
bash docker/multi_instances/run.sh compose logs -f --timestamps
bash docker/multi_instances/run.sh compose exec asr-1 bash
bash docker/multi_instances/run.sh compose config
```

> Watch the syntax: `docker compose -f <file> logs -f` (the `-f` *before* the
> subcommand selects the file; a `-f` *after* `logs` means "follow"). Mixing them
> — e.g. `docker compose logs -f multi_instances/docker-compose.yml ...` — makes
> compose ignore the file and print nothing.

**Warmup:** right after `up`, the aggregate `/health` shows
`{"replicas": 4, "healthy": 2, ...}` while the slower replicas finish JIT
compilation + model load (first start can take several minutes). Watch it climb
with `run.sh status`, and follow a straggler with `run.sh logs asr-3`. The
balancer retries the next replica when one refuses a connection, so requests
keep working during warmup — and a sticky client whose replica is down falls
back to another replica instead of looping on the dead one.

- Replica server output (uvicorn + JIT compile) is on **container stdout** —
  captured by the lines above.
- Per-request artifacts and the server's own log file live on disk under
  `${ROOT}/logs` (bind-mounted to `/app/logs` in every replica), e.g.
  `logs/user/output/veloxvoice-server.log` and `logs/user_data/`.
- The balancer logs every proxied request (aiohttp access log) plus a startup
  line `[lb] listening ...`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `VELOX_GPU` | `0` | host GPU index shared by all replicas (→ `CUDA_VISIBLE_DEVICES`) |
| `VELOX_PORT` | `8000` | single entry port (API + webpage + WS) |
| `REPLICAS` | `2` | initial replica count (rescale with `run.sh scale N`) |
| `MODEL_HOST_DIR` | `../../data/models/asr_model` | host path of the ASR bundle (ro) |
| `BASE_IMAGE` / `TORCH_INDEX` | CUDA 13 devel / cu130 | build args (see `docker/README.md`) |

Copy `.env.example` → `.env` to persist local choices.

## Notes

- First start compiles the JIT kernels (nvcc) and loads the model — allow a few
  minutes; the shared `veloxvoice-kernel-cache` volume makes later starts fast.
- `lb` reuses the same built image as `asr` (it already ships Python + aiohttp).
- The balancer is benchmark/production-lite grade; swap in nginx/Envoy for TLS,
  retries with backoff, or sticky sessions if you need them.
