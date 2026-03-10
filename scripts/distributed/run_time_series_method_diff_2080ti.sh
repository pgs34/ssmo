#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

run_gpu0() {
  bash scripts/distributed/run_time_series_block.sh 0 "etth1" "dlinear:dlinear"
}

run_gpu1() {
  echo "[DIST] GPU1 is reserved for smoke runs or failed-run retries."
  echo "[DIST] Example:"
  echo "CUDA_VISIBLE_DEVICES=1 METHODS='studygroup' DATASETS='etth1' MODEL_PAIRS='dlinear:dlinear' EPOCHS='2' OUTPUT_DIR='results/time_series_method_diff_retry' bash scripts/simple/run_simple_time_series.sh"
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"
