#!/usr/bin/env bash
# One-command reproduction of the VeloxVoice ASR GPU performance suite.
#
# Boots 3 workers + the LB on port 8000, runs the HTTP long/short sweeps and
# the WS streaming sweep, samples GPU memory, and writes raw JSONL plus a
# summary into --out.
#
#   bash benchmark/run_asr_bench_suite.sh [--out DIR] [--gpu N] [--workers "8100 8101 8102"]
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/venv-sgl/bin/python}"

GPU=5; WORKERS="8100 8101 8102"; PORT=8000

TMPDIR=/tmp/veloxvoice_live/

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
OUT="${OUT:-$ROOT/benchmark/results/suite_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

echo "[suite] starting workers: $WORKERS (gpu $GPU)"
for p in $WORKERS; do
  CUDA_VISIBLE_DEVICES="$GPU" setsid nohup /usr/bin/python3.12 -m veloxvoice.server \
      --model-dir "$ROOT/data/models/asr_model" --host 0.0.0.0 --port "$p" \
      > "$TMPDIR/asr_worker_$p.log" 2>&1 < /dev/null &
done
for p in $WORKERS; do
  for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$p/health" 2>/dev/null | grep -q healthy && break
    sleep 3
  done
  curl -sf "http://127.0.0.1:$p/health" | grep -q healthy \
    || { echo "worker $p failed to start; see $TMPDIR/asr_worker_$p.log" >&2; exit 1; }
done

echo "[suite] starting LB on $PORT -> $WORKERS"
setsid nohup "$PYTHON" "$ROOT/benchmark/asr_lb.py" --port "$PORT" $WORKERS \
    > "$OUT/lb.log" 2>&1 < /dev/null &
for i in $(seq 1 20); do
  curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q healthy && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" | grep -q healthy || { echo "LB failed" >&2; exit 1; }

LONG=test_data/speaker/meeting-test.wav
SHORT_GLOB='test_data/testset/aishell4/L_R004*.wav'

echo "[suite] HTTP long-audio sweep"
"$PYTHON" "$ROOT/benchmark/asr_concurrency_bench.py" \
  --server "http://127.0.0.1:$PORT" --audio "$LONG" \
  --levels 1 2 4 8 16 32 64 96 --gpu "$GPU" | tee "$OUT/http_long.jsonl"

echo "[suite] HTTP short-clip sweep"
"$PYTHON" "$ROOT/benchmark/asr_concurrency_bench.py" \
  --server "http://127.0.0.1:$PORT" --audio "$SHORT_GLOB" \
  --levels 1 4 16 64 128 --gpu "$GPU" | tee "$OUT/http_short.jsonl"

echo "[suite] WS streaming sweep (safe increments)"
"$PYTHON" "$ROOT/benchmark/asr_ws_concurrency_bench.py" \
  --server "http://127.0.0.1:$PORT" --audio "$LONG" --levels 1 2 4 8 | tee "$OUT/ws.jsonl"

echo "[suite] real-time multi-channel soak (safe increments 8 -> 32)"
"$PYTHON" "$ROOT/benchmark/asr_soak_realtime.py" \
  --server "http://127.0.0.1:$PORT" --audio "$LONG" \
  --levels 8 16 32 --duration "${SOAK_SECONDS:-120}" --gpu "$GPU" | tee "$OUT/soak.jsonl"

{
  echo "nvidia-smi -i $GPU:"
  nvidia-smi -i "$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
  echo "per-process:"
  nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader
} | tee "$OUT/gpu.txt"

{
  echo "# Suite summary ($(date -u +%FT%TZ))"
  echo "- lanes: http_long / http_short / ws (raw JSONL in $OUT)"
  echo "- single entry port $PORT -> workers: $WORKERS (gpu $GPU)"
  echo "- verify webpage: curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/client  (expect 200)"
  echo "- peak values:"
  tail -1 "$OUT/http_long.jsonl"
  tail -1 "$OUT/http_short.jsonl"
  tail -1 "$OUT/ws.jsonl"
  tail -1 "$OUT/soak.jsonl"
} | tee "$OUT/final_report.md"

echo "[suite] done -> $OUT"
