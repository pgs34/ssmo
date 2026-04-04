#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

POOL_ROOT="${POOL_ROOT:-results/time_series_etth1_best_ckpt_pool_v1/time_series/etth1}"
SEEDS="${SEEDS:-0 1 2}"
TRANSFORMER_SOURCE_ROOT="${TRANSFORMER_SOURCE_ROOT:-results/time_series_etth1_early_stop_v1/time_series/etth1}"
DLINEAR_SOURCE_ROOT="${DLINEAR_SOURCE_ROOT:-results/instruction_matrix_v1/time_series/time_series/etth1}"
mkdir -p "$POOL_ROOT"

for seed in $SEEDS; do
  transformer_dir="$POOL_ROOT/transformer_independent_huber_seed${seed}"
  dlinear_dir="$POOL_ROOT/dlinear_independent_huber_seed${seed}"
  mkdir -p "$transformer_dir" "$dlinear_dir"

  ln -sfn \
    "$ROOT_DIR/$TRANSFORMER_SOURCE_ROOT/transformer_independent_huber_seed${seed}/best_model.pt" \
    "$transformer_dir/best_model.pt"

  ln -sfn \
    "$ROOT_DIR/$DLINEAR_SOURCE_ROOT/dlinear_independent_mse_seed${seed}/best_model.pt" \
    "$dlinear_dir/best_model.pt"
done

echo "[prepare_etth1_best_ckpt_pool_v1] pool_root=$POOL_ROOT seeds=$SEEDS"
