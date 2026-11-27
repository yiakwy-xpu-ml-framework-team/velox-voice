#!/usr/bin/env bash
# run_music_server.sh — launch VeloxVoice with BOTH ASR + Music lanes.
#
#   MODEL_DIR=/path/to/asr_model \
#   MUSIC_MODEL_DIR=/path/to/SheetSage2 \
#   BASE_MODEL_DIR=/path/to/MERT-v2-FullSong \
#   bash scripts/run_music_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/data/models/asr_model}"
MUSIC_MODEL_DIR="${MUSIC_MODEL_DIR:-/raid/yiakwy/SheetSage2}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/raid/yiakwy/MERT-v2-FullSong}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU="${GPU:-5}"
VENV_PY="${VENV_PY:-/root/venv-music/bin/python}"

OUTPUT_DIR="$ROOT/logs/user/output"
LOG_FILE="$OUTPUT_DIR/veloxvoice-server.log"
PID_FILE="$OUTPUT_DIR/veloxvoice-server.pid"
mkdir -p "$OUTPUT_DIR"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Stopping existing server pid=$old_pid"
        kill "$old_pid"; sleep 3
    fi
    rm -f "$PID_FILE"
fi

cd "$ROOT"
printf '\n[%s] Starting VeloxVoice (ASR + Music) on GPU %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$GPU" >> "$LOG_FILE"

nohup env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HOME="/root/.cache/huggingface" \
    MODEL_DIR="$MODEL_DIR" \
    MUSIC_MODEL_DIR="$MUSIC_MODEL_DIR" \
    BASE_MODEL_DIR="$BASE_MODEL_DIR" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$ROOT" \
    stdbuf -oL -eL "$VENV_PY" -m uvicorn \
        launch_music_server:app \
        --host "$HOST" --port "$PORT" \
    >> "$LOG_FILE" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_FILE"
disown "$pid" 2>/dev/null || true
echo "VeloxVoice server started: pid=$pid, gpu=$GPU, log=$LOG_FILE"
echo "  ASR model:   $MODEL_DIR"
echo "  Music model: $MUSIC_MODEL_DIR (+ $BASE_MODEL_DIR)"
echo "  Web UI:      http://localhost:$PORT/client"
