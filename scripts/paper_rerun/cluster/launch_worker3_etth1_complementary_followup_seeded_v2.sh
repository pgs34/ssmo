#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_complementary_followup_seeded_v2.lock"
flock -n 9 || {
  echo "[worker3_etth1_complementary_followup_seeded_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_complementary_followup_seeded_v2/worker3}"
mkdir -p "$LOG_DIR"

CASE_SPECS="${CASE_SPECS:-trmix_biasm25_k17_lr6e5_l12_sp4e4_h96:0.00006:0.12:0.0004:96:0.00:-2.5:2:10:12:12:40:0.20:17 trmix_biasm28_k13_lr8e5_l12_sp4e4_h96:0.00008:0.12:0.0004:96:0.00:-2.8:2:10:12:12:40:0.20:13 trmix_biasm30_k13_lr1e4_l15_sp5e4_h64:0.00010:0.15:0.0005:64:0.00:-3.0:2:8:8:15:45:0.25:13}"

run_logged_job \
  "worker3/etth1_complementary_followup_seeded_v2" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_3090TI:-64}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}" \
    EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-12}" \
    EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}" \
    OUTPUT_ROOT="results/time_series_etth1_complementary_followup_seeded_v2/worker3" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="$CASE_SPECS" \
    bash scripts/paper_rerun/run_time_series_etth1_complementary_v4.sh

echo "[worker3_etth1_complementary_followup_seeded_v2] job finished"
