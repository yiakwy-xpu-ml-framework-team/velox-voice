#!/usr/bin/env bash
# Container entrypoint for the VeloxVoice ASR server.
#
# Maps the working GPU (VELOX_GPU, host physical index) onto
# CUDA_VISIBLE_DEVICES and launches the OpenAI-compatible server. All knobs are
# environment variables so docker-compose / `docker run -e` can drive them.
#
#   VELOX_GPU         0 (default; use 5 for the current H800 test box)
#   VELOX_MODEL_DIR   /models/asr_model
#   VELOX_HOST        0.0.0.0
#   VELOX_PORT        8000
#
# Extra CMD args are appended to the server command line.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${VELOX_GPU:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

VELOX_MODEL_DIR="${VELOX_MODEL_DIR:-/models/asr_model}"
VELOX_HOST="${VELOX_HOST:-0.0.0.0}"
VELOX_PORT="${VELOX_PORT:-8000}"
VELOXVOICE_KERNEL_CACHE="${VELOXVOICE_KERNEL_CACHE:-/var/cache/veloxvoice}"

mkdir -p "$VELOXVOICE_KERNEL_CACHE" /app/logs/user/output /app/logs/user_data

echo "[veloxvoice] gpu(CUDA_VISIBLE_DEVICES)=${CUDA_VISIBLE_DEVICES} " \
     "model=${VELOX_MODEL_DIR} bind=${VELOX_HOST}:${VELOX_PORT} " \
     "kernel_cache=${VELOXVOICE_KERNEL_CACHE}"

if [[ ! -f "${VELOX_MODEL_DIR}/train.yaml" ]]; then
    echo "[veloxvoice] ERROR: model dir '${VELOX_MODEL_DIR}' does not look like an" >&2
    echo "  ASR bundle (missing train.yaml). Mount the model, e.g.:" >&2
    echo "    -v /path/to/asr_model:/models/asr_model:ro" >&2
    exit 1
fi

exec python -m veloxvoice.server \
    --model-dir "$VELOX_MODEL_DIR" \
    --host "$VELOX_HOST" \
    --port "$VELOX_PORT" \
    "$@"
