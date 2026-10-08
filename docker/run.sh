# VeloxVoice ASR — one-command build & serve.
#
#   bash docker/run.sh              # working GPU defaults to 0
#   VELOX_GPU=5 bash docker/run.sh  # current H800 test box uses GPU 5
#   bash docker/run.sh logs         # follow server logs
#   bash docker/run.sh stop         # stop and remove the container
#
# GPU selection: the container is given ALL host GPUs and picks the physical
# index through CUDA_VISIBLE_DEVICES=VELOX_GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID),
# mirroring scripts/run_server_background.sh.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DOCKER_DIR/.." && pwd)"
COMPOSE=(docker compose -f "$DOCKER_DIR/docker-compose.yml")

# Optional docker/.env (VELOX_GPU=..., VELOX_PORT=..., MODEL_HOST_DIR=...).
if [[ -f "$DOCKER_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$DOCKER_DIR/.env"
    set +a
fi

export VELOX_GPU="${VELOX_GPU:-0}"
export VELOX_PORT="${VELOX_PORT:-8000}"

cmd="${1:-up}"
cd "$ROOT"

case "$cmd" in
  up|start)
    echo "[run] building + starting VeloxVoice ASR on GPU ${VELOX_GPU} (port ${VELOX_PORT})"
    "${COMPOSE[@]}" up --build -d
    echo -n "[run] waiting for /health "
    for _ in $(seq 1 90); do
        if curl -fsS "http://127.0.0.1:${VELOX_PORT}/health" >/dev/null 2>&1; then
            echo "ok"
            curl -s "http://127.0.0.1:${VELOX_PORT}/health"; echo
            echo "[run] service ready: http://127.0.0.1:${VELOX_PORT} (docs: /docs, client: /client)"
            exit 0
        fi
        echo -n "."
        sleep 5
    done
    echo
    echo "[run] health check timed out — inspect logs:" >&2
    echo "      bash docker/run.sh logs" >&2
    exit 1
    ;;
  logs)
    "${COMPOSE[@]}" logs -f --tail=200
    ;;
  stop|down)
    "${COMPOSE[@]}" down
    ;;
  build)
    "${COMPOSE[@]}" build
    ;;
  *)
    echo "usage: bash docker/run.sh [up|logs|stop|build]" >&2
    exit 2
    ;;
esac
