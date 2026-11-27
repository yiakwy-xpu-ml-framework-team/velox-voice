#!/usr/bin/env bash
# run_music_server.sh — launch VeloxVoice with the SheetSage2 music lane.
#
# MODEL_DIR is optional: leave it empty for a music-only server (no WeNet),
# or point it at an ASR checkpoint to serve both lanes.
#
#   MUSIC_MODEL_DIR=/path/to/SheetSage2 \
#   BASE_MODEL_DIR=/path/to/MERT-v2-FullSong \
#   VENV_PY=/home/yiakwy/venvs/sglang-h3/bin/python \
#   bash scripts/run_music_server.sh
#
# Set VENV_PY to the environment that already has torch/SheetSage2/demucs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-}"
MUSIC_MODEL_DIR="${MUSIC_MODEL_DIR:-/home/yiakwy/workspace/Github/models/SheetSage2}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/yiakwy/workspace/Github/models/MERT-v2-FullSong}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
VENV_PY="${VENV_PY:-python3}"
VELOX_DEMUCS_MODEL="${VELOX_DEMUCS_MODEL:-htdemucs_ft}"
VELOX_MUSIC_OUTPUT_DIR="${VELOX_MUSIC_OUTPUT_DIR:-$ROOT/output/music_sessions}"

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
printf '\n[%s] Starting VeloxVoice (ASR=%s, Music=%s) on GPU %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${MODEL_DIR:-disabled}" \
    "$MUSIC_MODEL_DIR" "$GPU" >> "$LOG_FILE"

nohup env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    MODEL_DIR="$MODEL_DIR" \
    MUSIC_MODEL_DIR="$MUSIC_MODEL_DIR" \
    BASE_MODEL_DIR="$BASE_MODEL_DIR" \
    VELOX_DEMUCS_MODEL="$VELOX_DEMUCS_MODEL" \
    VELOX_MUSIC_OUTPUT_DIR="$VELOX_MUSIC_OUTPUT_DIR" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$ROOT" \
    stdbuf -oL -eL "$VENV_PY" -m uvicorn \
        launch_music_server:app \
        --host "$HOST" --port "$PORT" \
    >> "$LOG_FILE" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_FILE"
disown "$pid" 2>/dev/null || true
host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "VeloxVoice server started: pid=$pid, gpu=$GPU, log=$LOG_FILE"
echo "  ASR model:   ${MODEL_DIR:-disabled}"
echo "  Music model: $MUSIC_MODEL_DIR (+ $BASE_MODEL_DIR)"
echo "  Demucs:      $VELOX_DEMUCS_MODEL (tools/music/separate_vocals.py)"
echo "  Sessions:    $VELOX_MUSIC_OUTPUT_DIR"
echo "  Local UI:    http://localhost:$PORT/client"
if [[ -n "$host_ip" ]]; then
    echo "  LAN UI:      http://$host_ip:$PORT/client"
    echo "  Music UI:    http://$host_ip:$PORT/client?tab=music"
fi
echo "  Note: localhost only works on this host; remote browsers need the LAN"
echo "        URL above or: ssh -N -L $PORT:127.0.0.1:$PORT <user>@<host>"
