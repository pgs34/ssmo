#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_dense_regime_horizon_seeded_v3.lock"
flock -n 9 || {
  echo "[worker3_etth1_dense_regime_horizon_seeded_v3] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_dense_regime_horizon_seeded_v3/worker3}"
mkdir -p "$LOG_DIR"

CASE_SPECS="${CASE_SPECS:-dhr_wide_reg45_ad30_ws5_exp9_raw:0.00005:0.10:0.00015:128:0.00:-1.8:0.30:2:8:8:10:32:0.20:9:0.20:0.45:0.04:0.45:0.30:0.90:5:9:0.00:1.00:raw:0.0000 dhr_tail75_reg85_ad55_ws3_exp5_res:0.00007:0.18:0.00040:96:0.00:-3.0:0.40:2:12:14:15:50:0.25:21:0.75:0.85:0.15:0.25:0.55:0.45:3:5:0.50:1.25:residual:0.0000 dhr_midtail_reg65_ad38_ws7_exp11_delta:0.00009:0.12:0.00030:128:0.00:-2.4:0.32:2:10:10:12:42:0.20:17:0.35:0.65:0.09:0.38:0.38:0.75:7:11:0.20:1.15:delta:0.0000 dhr_full_reg55_ad42_ws1_exp1_raw:0.00008:0.08:0.00015:64:0.00:-1.6:0.28:1:6:6:8:28:0.20:13:0.00:0.55:0.03:0.50:0.42:0.60:1:1:0.00:0.90:raw:0.0000}"

run_logged_job \
  "worker3/etth1_dense_regime_horizon_seeded_v3" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_3090TI:-64}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-18}" \
    EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-15}" \
    EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}" \
    OUTPUT_ROOT="results/time_series_etth1_dense_regime_horizon_seeded_v3/worker3" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="$CASE_SPECS" \
    bash scripts/paper_rerun/run_time_series_etth1_dense_regime_horizon_seeded_v3.sh

echo "[worker3_etth1_dense_regime_horizon_seeded_v3] job finished"
