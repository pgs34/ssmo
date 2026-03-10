#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

run_gpu0() {
  bash scripts/distributed/run_time_series_block.sh 0 "electricity" "transformer:dlinear"
  bash scripts/distributed/run_time_series_block.sh 0 "weather" "transformer:transformer"
}

run_gpu1() {
  bash scripts/distributed/run_time_series_block.sh 1 "electricity" "transformer:transformer"
  bash scripts/distributed/run_time_series_block.sh 1 "weather" "transformer:dlinear"
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"
