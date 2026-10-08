#!/usr/bin/env bash
# One-command build + start of dynamically scaled VeloxVoice replicas + LB.
#
#   bash docker/multi_instances/run.sh                 # 2 replicas, GPU 0
#   VELOX_GPU=5 REPLICAS=4 bash docker/multi_instances/run.sh
#   bash docker/multi_instances/run.sh scale 6         # rescale at runtime
#   bash docker/multi_instances/run.sh verify          # end-to-end two-step upload->transcribe check
#   bash docker/multi_instances/run.sh logs|stop|build
#
# Entry port 8000 serves everything: the API, the test webpage (/client) and
# the WS streaming lane, all proxied to the current replica set.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
COMPOSE=(docker compose -f "$HERE/docker-compose.yml")

if [[ -f "$HERE/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$HERE/.env"
    set +a
fi

export VELOX_GPU="${VELOX_GPU:-0}"
export VELOX_PORT="${VELOX_PORT:-8000}"
REPLICAS="${REPLICAS:-2}"

wait_healthy() {
    local port="$1" want="$2"
    echo -n "[multi] waiting for ${want} healthy replica(s) "
    for _ in $(seq 1 240); do
        body="$(curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
        healthy="$(printf '%s' "$body" | grep -o '"healthy": *[0-9]*' | grep -o '[0-9]*' || true)"
        replicas="$(printf '%s' "$body" | grep -o '"replicas": *[0-9]*' | grep -o '[0-9]*' || true)"
        if [[ -n "$healthy" && -n "$replicas" && "$healthy" -ge "$want" ]]; then
            echo " ok (${healthy}/${replicas})"
            echo "[multi] $body"
            return 0
        fi
        printf '.'
        sleep 5
    done
    echo
    echo "[multi] timed out waiting for ${want} healthy replica(s) - still warming up?" >&2
    echo "        inspect:  bash docker/multi_instances/run.sh logs asr" >&2
    return 1
}

cmd="${1:-up}"
cd "$ROOT"

case "$cmd" in
  up|start)
    echo "[multi] GPU=${VELOX_GPU} replicas=${REPLICAS} entry=:${VELOX_PORT}"
    "${COMPOSE[@]}" up --build -d --scale "asr=${REPLICAS}"
    wait_healthy "$VELOX_PORT" "$REPLICAS"
    echo "[multi] API     : http://localhost:${VELOX_PORT}/v1/audio/transcriptions"
    echo "[multi] webpage : http://localhost:${VELOX_PORT}/client"
    echo "[multi] docs    : http://localhost:${VELOX_PORT}/docs"
    echo "[multi] logs    : bash docker/multi_instances/run.sh logs"
    echo "                  (or: docker compose -f docker/multi_instances/docker-compose.yml logs -f)"
    echo "[multi] on-disk : ${ROOT}/logs (mounted at /app/logs in every replica)"
    ;;
  scale)
    n="${2:?usage: run.sh scale <N>}"
    echo "[multi] scaling asr replicas -> ${n}"
    "${COMPOSE[@]}" up -d --scale "asr=${n}" --no-recreate
    wait_healthy "$VELOX_PORT" "$n"
    ;;
  logs)
    # run.sh logs            -> every service of THIS project (asr + lb)
    # run.sh logs lb         -> balancer only
    # run.sh logs asr        -> all replicas
    # run.sh logs asr-1      -> a single replica
    target="${2:-all}"
    case "$target" in
      all)
        "${COMPOSE[@]}" logs -f --tail=200 --timestamps
        ;;
      lb)
        docker logs -f --tail=200 --timestamps veloxvoice-multi-lb-1
        ;;
      asr)
        "${COMPOSE[@]}" logs -f --tail=200 --timestamps asr
        ;;
      asr-[0-9]*)
        docker logs -f --tail=200 --timestamps "veloxvoice-multi-${target}"
        ;;
      *)
        echo "unknown log target: $target (use: all | asr | lb | asr-1..N)" >&2
        exit 2
        ;;
    esac
    ;;
  status|ps)
    "${COMPOSE[@]}" ps
    curl -s "http://127.0.0.1:${VELOX_PORT}/health"; echo
    ;;
  verify)
    # End-to-end check of the webpage's STATEFUL two-step flow through the LB:
    # upload -> audio_id -> transcribe must hit the SAME replica (no 404).
    base="http://127.0.0.1:${VELOX_PORT}"
    wav="${WAV:-$ROOT/test_data/testset/aishell4/L_R004S03C01_interval_251.wav}"
    [[ -f "$wav" ]] || { echo "[verify] sample wav not found: $wav (set WAV=...)" >&2; exit 1; }

    echo "[verify] GET /client"
    code="$(curl -s -o /dev/null -w '%{http_code}' "$base/client")"
    echo "         /client -> $code"

    echo "[verify] POST /v1/audio/uploads"
    url_upload="$base/v1/audio/uploads"
    up_resp="$(curl -s -F "file=@${wav}" "$url_upload")"
    audio_id="$(printf '%s' "$up_resp" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null)"
    if [[ -z "$audio_id" ]]; then
        echo "         FAIL: upload returned no id: ${up_resp:0:200}" >&2
        exit 1
    fi
    echo "         upload id=${audio_id}"

    echo "[verify] POST /v1/audio/transcriptions (audio_id, stream=true)"
    url_tr="$base/v1/audio/transcriptions"
    out="$(curl -sN -F "audio_id=${audio_id}" -F "stream=true" "$url_tr")"
    if printf '%s' "$out" | grep -q "transcript.text.done"; then
        echo "         PASS: streamed to transcript.text.done"
        echo "[verify] OK - the two-step flow survives load balancing"
    elif printf '%s' "$out" | grep -qiE "unknown audio_id|error"; then
        echo "         FAIL: ${out:0:300}" >&2
        echo "[verify] the LB did not route the transcription to the upload replica" >&2
        exit 1
    else
        echo "         WARN: no done marker, response head: $(printf '%s' "$out" | head -c 200)" >&2
        exit 1
    fi
    ;;
  stop|down)
    echo "[multi] stopping project 'veloxvoice-multi'"
    "${COMPOSE[@]}" down --remove-orphans || true
    # Fallback: force-remove anything left carrying this project's label, in
    # case compose metadata is stale (e.g. a container orphaned by a crash).
    leftover="$(docker ps -aq \
        --filter label=com.docker.compose.project=veloxvoice-multi 2>/dev/null || true)"
    if [[ -n "$leftover" ]]; then
        echo "[multi] force-removing leftovers: $(echo "$leftover" | tr '\n' ' ')"
        echo "$leftover" | xargs -r docker rm -f
    fi
    docker network rm veloxvoice-multi_default >/dev/null 2>&1 || true
    echo "[multi] stopped"
    ;;
  build)
    "${COMPOSE[@]}" build
    ;;
  compose|dc)
    # passthrough: run.sh compose <any docker compose args>
    # e.g. run.sh compose logs -f --timestamps
    #      run.sh compose exec asr-1 bash
    #      run.sh compose config
    shift
    "${COMPOSE[@]}" "$@"
    ;;
  *)
    echo "usage: bash docker/multi_instances/run.sh [up|scale N|logs [all|asr|lb|asr-N]|status|verify|stop|build|compose ...]" >&2
    echo "note: this project is 'veloxvoice-multi'; a bare 'docker compose down'" >&2
    echo "      in docker/ targets the OTHER project ('veloxvoice'). Stop from" >&2
    echo "      anywhere with: docker compose -p veloxvoice-multi down" >&2
    exit 2
    ;;
esac
