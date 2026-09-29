#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="${VELOX_MODEL_DIR:-$ROOT/data/models/asr_model}"

HOST="${VELOX_HOST:-0.0.0.0}"
PORT="${VELOX_PORT:-8000}"

GPU="${VELOX_GPU:-5}"

OUTPUT_DIR="$ROOT/logs/user/output"

LOG_FILE="$OUTPUT_DIR/veloxvoice-server.log"
PID_FILE="$OUTPUT_DIR/veloxvoice-server.pid"

mkdir -p "$OUTPUT_DIR"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Stopping existing VeloxVoice server pid=$old_pid" | tee -a "$LOG_FILE"
        kill "$old_pid"
        for _ in $(seq 1 30); do
            kill -0 "$old_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    rm -f "$PID_FILE"
fi

cd "$ROOT"
printf '\n[%s] Starting VeloxVoice server on GPU %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$GPU" >> "$LOG_FILE"

nohup env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTHONUNBUFFERED=1 \
    stdbuf -oL -eL python -m veloxvoice.server \
        --model-dir "$MODEL_DIR" \
        --host "$HOST" \
        --port "$PORT" \
    >> "$LOG_FILE" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_FILE"
disown "$pid" 2>/dev/null || true

printf '%s\n' "$pid" | tee -a "$LOG_FILE" >/dev/null
echo "VeloxVoice server started: pid=$pid, gpu=$GPU, log=$LOG_FILE"
