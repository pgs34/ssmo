#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REG_LOSSES=(mse mae huber)
LAMBDAS=(0.5 1.0 2.0)
MARGINS=(0.0 0.05)
WARMUPS=(3 5)
SWEEP_OUTPUT_DIR="results/sweep/operator/hyper_parameter"

echo "[CONFIG] task=operator_sweep"
echo "[CONFIG] reg_losses=${REG_LOSSES[*]}"
echo "[CONFIG] lambdas=${LAMBDAS[*]} margins=${MARGINS[*]} warmups=${WARMUPS[*]}"
echo "[CONFIG] methods=dml studygroup datasets=darcy navier_stokes burgers"
echo "[CONFIG] models=fno deeponet gnot"
echo "[CONFIG] seeds=0 1 2 3 4 epochs=180 batch_size=16 device=cuda"
echo "[CONFIG] output_dir=$SWEEP_OUTPUT_DIR"

for reg_loss in "${REG_LOSSES[@]}"; do
  for lambda_imitation in "${LAMBDAS[@]}"; do
    for margin in "${MARGINS[@]}"; do
      for warmup in "${WARMUPS[@]}"; do
        echo "[TUNE] operator loss=$reg_loss lambda=$lambda_imitation margin=$margin warmup=$warmup"
        REGRESSION_IMITATION_LOSS="$reg_loss" \
        LAMBDA_IMITATION="$lambda_imitation" \
        MARGIN="$margin" \
        WARMUP_STUDYGROUP="$warmup" \
        METHODS="dml studygroup" \
        DATASETS="darcy navier_stokes burgers" \
        MODELS="fno deeponet gnot" \
        SEEDS="0 1 2 3 4" \
        EPOCHS="180" \
        BATCH_SIZE="16" \
        DEVICE="cuda" \
        OUTPUT_DIR="$SWEEP_OUTPUT_DIR" \
        MAX_TRAIN_BATCHES="" \
        MAX_VAL_BATCHES="" \
        bash scripts/simple/run_simple_operator.sh
      done
    done
  done
done
