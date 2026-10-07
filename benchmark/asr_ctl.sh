#!/usr/bin/env bash
# Start/stop helpers for VeloxVoice ASR workers.
#
#   benchmark/asr_ctl.sh start <port>   # one worker on the current GPU (CUDA_VISIBLE_DEVICES)
#   benchmark/asr_ctl.sh stopall        # stop every veloxvoice server process
#
# NOTE: the patterns here deliberately avoid `pkill -f <pattern>` from an
# interactive shell whose own cmdline contains the pattern (it would kill the
# shell). The [.] trick plus script-file execution keeps self-matches away.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/veloxvoice_live}"
mkdir -p "$LOG_DIR"

case "${1:-}" in
  start)
    port="${2:?usage: asr_ctl.sh start <port>}"
    cd "$ROOT"
    setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" nohup \
        /usr/bin/python3.12 -m veloxvoice.server \
        --model-dir "$ROOT/data/models/asr_model" --host 0.0.0.0 --port "$port" \
        > "$LOG_DIR/asr_worker_$port.log" 2>&1 < /dev/null &
    echo "started worker :$port (log $LOG_DIR/asr_worker_$port.log)"
    ;;
  stopall)
    for pid in $(pgrep -f "veloxvoice[.]server"); do
        kill "$pid" 2>/dev/null && echo "stopped $pid"
    done
    ;;
  *)
    echo "usage: asr_ctl.sh start <port> | stopall" >&2
    exit 2
    ;;
esac
