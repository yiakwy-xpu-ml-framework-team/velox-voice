# VeloxVoice ASR Concurrency / avg RTF Benchmark Guide

--

## 0. Interfaces to measure

VeloxVoice is an ASR inference engine, supports **WeNet-style Conformer + CTC** (!= WeNet) with modern JIT kernels.

The API service exposes two interfaces:

| Lane                | Endpoint                        | Who uses it               | Timing source        |
|---------------------|---------------------------------|---------------------------|----------------------|
| curl                | `POST /v1/audio/transcriptions` | OpenAI-compatible clients | response JSON fields |
| WebSocket streaming | `WS /v1/audio/ws`               | browser                   |       payload fields |
| WebRTC              | `WS /v1/audio/ws`               | app                       |       payload fields |

The three timing numbers every report must contain:

1. **Server processing time / server RTF** — time inside the engine
   (`elapsed_seconds` in the response; `RTF = elapsed_seconds / audio_seconds`).
2. **User end-to-end time / e2e RTF** — client wall clock / audio duration
   (upload + decode + queue + server processing + download).
3. **Upload + decode residual** — `e2e − server processing`. When testing on the
   server itself, real network upload ≈ 0 and this residual is dominated by ffmpeg
   decode + queueing + the LB copy.

Golden Standard of VeloxVoice, batch 1 RTF reference : `0.0003` in Hopper.

---

## 1. Prerequisites

```bash
cd $PROJECT_ROOT # e.g. PROJECT_ROOT=VELOX_HOME=VeloxVoice-dev

export PYTHONPATH=`pwd`

PYTHON=/root/venv-sgl/bin/python                              # has requests + aiohttp + matplotlib + python-pptx
nvidia-smi -i 5 --query-gpu=memory.free --format=csv,noheader # need > 20 GiB free
```

Test audio (already in the repo):

| File | Duration | Use |
|------------------------------------------|------------------------|---------------------------------------------------------|
| `test_data/speaker/meeting-test.wav`     | 99.6 s, 48 kHz         | standard audio workload (diaraition support is pending) |
| `test_data/testset/aishell4/L_R004*.wav` | ~3.8 s avg             | short-clip                                              |

---

## 2. Architecture

```
                ┌────────────── single entry port 8000 ───────────────────────┐
clients ──────► │  benchmark/asr_lb.py   (round-robin HTTP + WS)          ... │
                └───────┬──────────────────┬──────────────────┬───────────────┘
                        ▼                  ▼                  ▼
                 worker :8100        worker :8101        worker :8102     ...
                 (GPU 5, ~4 GB)      (GPU 5, ~4 GB)      (GPU 5, ~4 GB)
```

* Each worker process holds one `Velox` engine guarded by a **global
  `threading.Lock`** (`veloxvoice/api.py`);
* The LB is a minimal aiohttp round-robin proxy (HTTP multipart + WebSocket).
  It is benchmark-grade, not hardened for the public internet.
* API note: the plain-JSON transcription response now also carries
  `audio_seconds`, `elapsed_seconds`, `rtf`.

---

## 3. Launch / reset the topology (idempotent)

```bash
# start 3 workers on GPU 5 (safe to re-run; starting an already-running port is fine)
bash benchmark/asr_ctl.sh start 8100
bash benchmark/asr_ctl.sh start 8101
bash benchmark/asr_ctl.sh start 8102

# wait until all three answer /health
for p in 8100 8101 8102; do
  until curl -sf http://127.0.0.1:$p/health | grep -q healthy; do sleep 3; done
done

# LB on port 8000 (the single entry). If 8000 is occupied by a bare worker, free it:
# bash benchmark/asr_ctl.sh stopall            # kills ALL veloxvoice servers

# ...then re-start the 3 workers as above, then:
setsid nohup $PYTHON benchmark/asr_lb.py --port 8000 --workers 8100 8101 8102 \
    > /tmp/asr_lb_8000.log 2>&1 < /dev/null &
curl -s http://127.0.0.1:8000/health | jq # must return healthy
```

Sanity checks before any benchmark:

```bash
# expect 200 (webpage intact)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/client

# expect:

# {
#   "text": "小孩儿他是不可能拿力那么大推",
#   "audio_seconds": 2.259125,
#   "elapsed_seconds": 0.12391576590016484,
#   "rtf": 0.054851221557091726,
#   "usage": {
#     "type": "duration",
#     "seconds": 3
#   }
# }
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
     -F file=@test_data/testset/aishell4/L_R004S03C01_interval_251.wav | jq .
```

---

## 4. Run the benchmarks

### 4.1 Standard audio workload (100 secs) concurrency bench via CURL

```bash
$PYTHON benchmark/asr_concurrency_bench.py \
    --server http://127.0.0.1:8000 \
    --audio test_data/speaker/meeting-test.wav \
    --levels 1 2 4 8 16 32 64 96 --gpu 5
```

Max tput 41.927 req/s (out of expectation!), rtf 0.0106 (recession) at concurrency of 96.

Mean tput 26.547 req/s, with avg RTF 0.0049

### 4.2 Short-clip

```bash
$PYTHON benchmark/asr_concurrency_bench.py --server http://127.0.0.1:8000 \
    --audio 'test_data/testset/aishell4/L_R004*.wav' --levels 1 4 16 64 128
```

Max tpu 321 req/s, rtf 0.2144

Mean tput 192.46 req/s, with avg RTF 0.08

### 4.3 WS / WebRTC streaming lane — test SEPARATELY, safe increments

```bash
$PYTHON
python benchmark/asr_ws_concurrency_bench.py --server http://127.0.0.1:8000 \
    --audio test_data/speaker/meeting-test.wav --levels 1 2 4 8 16 32
```

Max tput 12.351 sessions/s, rtf 0.0966

Mean tput 9.394 sessions/s, with avg RTF 0.0252

Safe-mode semantics: each level spins up N independent ws sessions; if any
session fails the sweep **stops** rather than pushing a broken service harder.

Each level's JSON line carries `server_process_s_mean` (server RTF),
`e2e_s_mean` (user e2e), `e2e_rtf_mean`, `upload_decode_residual_s_mean`.

### 4.4 Real-time multi-channel（32）soak

```bash
$PYTHON
python benchmark/asr_soak_realtime.py --server http://127.0.0.1:8000 \
    --audio test_data/speaker/meeting-test.wav --levels 8 16 32 --duration 120
```

N independent ws sessions stream raw 16 kHz mono PCM paced at 1x realtime.
. Real-time safety stays bounded.

NOTE :

- the script pre-converts the source with ffmpeg : in `s16le` mode the server
interprets `-ar/-ac` as the raw input layout, feeding anything else inflates
duration by the sample-rate ratio

- Real-time safety = per-channel backlog (audio sent − audio processed)

Reference (H800, 3 workers):
32/32 channels, 120 s soak, backlog 0.0 s, engine RTF 0.013, 13.5 GiB GPU.

| channels | ok | failed | wall_s | backlog_mean | backlog_max | rtf_mean | safe | gpu_mem | gpu_util |
|---------:|---:|-------:|-------:|-------------:|------------:|---------:|------|--------:|---------:|
| 8        | 8  | 0      | 120.2  | 0            | 0           | 0.00455  | true | 15633   | 0        |
| 16       | 16 | 0      | 120.3  | 0            | 0           | 0.00753  | true | 15633   | 22       |
| 32       | 32 | 0      | 120.5  | 0            | 0           | 0.01337  | true | 15633   | 86       |

### 4.5 All in one (workers + LB + all sweeps + artifacts)

```bash
# writes suite/final_report.md + raw JSONL for every method
bash benchmark/run_asr_bench_suite.sh --out /tmp/asr_suite_$(date +%s)
```

---

## 5. Harnessing Results (H800, GPU 5, measured 2026-10-06)

Reference the table the author produced, use these as the acceptance
envelope when verifying. Your (the reproducer) absolute numbers may
shift ±10 % depending on co-tenants on the GPU; the SHAPE must match.

### 5.1 Single worker, 99.6 s audio

| c | rps | server proc | server RTF | e2e | e2e RTF | GPU util peak |
|---|-----|-------------|------------|------|---------|----------------|
| 1 | 4.8 | 33 ms | 0.00033 | 0.19 s | 0.0021 | low |
| 8 | 19.7 | 58 ms | 0.00058 | 0.34 s | 0.0035 | ~46 % |
| 32 | 34.6 | 148 ms | 0.00149 | 0.60 s | 0.0058 | ~67 % |

Plateau ≈ 30-36 rps: the per-worker lock serializes transcribe; extra clients only queue.

### 5.2 Three workers behind the LB, 99.6 s audio

| c | rps | e2e RTF | GPU util peak | GPU mem (3 workers) |
|---|-----|---------|----------------|----------------------|
| 32 | 33.8 | 0.0057 | ~90 % | 11.4 GiB |
| 64 | 37.9 | 0.0125 | ~95 % | 12.5 GiB |
| 96 | **42.0 (4184× realtime)** | 0.0115 | ~94 % | 13.8 GiB |

Interpretation: **the shared GPU saturates** (util 94-95 %); adding workers past
this point only shortens queues, it cannot add throughput. GPU memory cost is
~3.9-5.8 GiB per worker (~13.8 GiB for 3).

### 5.3 WS streaming lane, 99.6 s audio (3 workers)

| sessions | server proc | server RTF | e2e | e2e RTF | sessions/s |
|---|-------------|------------|------|---------|------------|
| 1 | 38 ms | 0.0004 | 0.27 s | 0.0027 | 3.6 |
| 8 | 132 ms | 0.0013 | 0.50 s | 0.0050 | 14.1 |

### 5.4 Short clips (3.84 s avg), single worker

c=1 → 13.4 rps (p50 75 ms); plateau ~40-43 rps at c≥64; GPU util ≤ 23 %
(overhead-dominated: decode/HTTP/lock per tiny request).

---

## 6. What to check when writing the verification report

1. `failed: 0` on every level you report. A level with failures is NOT a result.
2. `server_rtf_mean` at c=1 for the 99.6 s clip should be ≈ 0.0003-0.0006.
   If you see > 0.002, another process is sharing the GPU — check `nvidia-smi`.
3. The 3-worker curve must be ≥ the single-worker curve at every level (LB adds
   a copy but never more than ~1 ms). If it is lower, a worker died — check
   `/tmp/asr_worker_810x.log`.
4. Webpage intactness: `GET /client` must be 200 and a ws smoke session must
   produce a `final` message (see 4.3; one session = level 1).
5. GPU memory: quote `nvidia-smi --query-compute-apps=pid,used_memory` per worker
   plus the free memory; "fits N more workers" claims must come from free/avg.
6. Distinguish **throughput saturation** (rps stops growing) from
   **latency collapse** (p50 keeps growing): report both.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| LB returns 502 "Cannot connect" | worker died or not started | `bash benchmark/asr_ctl.sh start 8100`; check worker log |
| `elapsed_seconds` missing in response | worker predates the API change | restart workers (§3) |
| RTF looks 10× worse | GPU shared with training jobs | `nvidia-smi` → pick a quiet GPU, pass `--gpu` |
| ws bench hangs at connect | LB not running | check `/tmp/asr_lb_8000.log`; restart §3 |
| `pkill -f asr_lb` kills your own shell | pattern matches your cmdline | use `benchmark/asr_ctl.sh` or `pgrep -f veloxvoice[.]server` |
| uvicorn workers die under very high c | ffmpeg decode thread pool + uploads | keep c ≤ 96 on one GPU for the 99.6 s clip |

## 8. Cleanup

```bash
bash benchmark/asr_ctl.sh stopall        # stops all workers
# stop the LB (careful: match exactly, not -f with a self-matching pattern)
kill $(pgrep -f 'asr_lb[.]py --port 8000') 2>/dev/null || true
```
