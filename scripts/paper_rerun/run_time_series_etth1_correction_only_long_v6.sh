#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

export LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_correction_only_long_v6}"
mkdir -p "$LOG_DIR"

export OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_correction_only_long_v6}"
export SEEDS="${SEEDS:-0 1 2}"
export EPOCHS="${EPOCHS:-90}"
export BATCH_SIZE="${BATCH_SIZE:-768}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-24}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-60}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.00005}"
export CASE_SPECS="${CASE_SPECS:-corrlong_tail35_q50_lr6e4:0.0006:0.06:0.00005:96:0.00:-1.2:1:10:0.35:0.50:1.0:9 corrlong_tail45_q55_lr8e4:0.0008:0.08:0.00010:96:0.00:-1.6:1:10:0.45:0.55:1.2:13 corrlong_tail55_q60_lr1e3:0.0010:0.10:0.00010:64:0.00:-2.0:1:8:0.55:0.60:1.5:17}"
export REFRESH_TOP_LEVEL="${REFRESH_TOP_LEVEL:-0}"

echo "[time_series_etth1_correction_only_long_v6] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_correction_only_long_v6] gpu=${GPU:-0} seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[time_series_etth1_correction_only_long_v6] early_stop patience=$EARLY_STOP_PATIENCE min_epochs=$EARLY_STOP_MIN_EPOCHS"

bash "$SCRIPT_DIR/run_time_series_etth1_correction_only_v5.sh"

if [[ "$REFRESH_TOP_LEVEL" == "1" ]]; then
  echo "[time_series_etth1_correction_only_long_v6] refreshing top-level plots"
  bash scripts/paper_rerun/refresh_top_level_best_plots.sh
fi

echo "[time_series_etth1_correction_only_long_v6] done"

