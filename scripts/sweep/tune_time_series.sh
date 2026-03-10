#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REG_LOSSES=(mse mae huber)
LAMBDAS=(0.5 1.0 2.0)
MARGINS=(0.0 0.05)
PRED_LENS=(24 48)
WARMUPS=(2 3)
SWEEP_OUTPUT_DIR="results/sweep/time_series/hyper_parameter"

echo "[CONFIG] task=time_series_sweep"
echo "[CONFIG] reg_losses=${REG_LOSSES[*]}"
echo "[CONFIG] lambdas=${LAMBDAS[*]} margins=${MARGINS[*]} pred_lens=${PRED_LENS[*]} warmups=${WARMUPS[*]}"
echo "[CONFIG] methods=dml studygroup datasets=etth1 electricity weather"
echo "[CONFIG] model_pairs=transformer:transformer dlinear:dlinear transformer:dlinear"
echo "[CONFIG] seeds=0 1 2 3 4 epochs=80 batch_size=64 device=cuda"
echo "[CONFIG] output_dir=$SWEEP_OUTPUT_DIR"

for reg_loss in "${REG_LOSSES[@]}"; do
  for lambda_imitation in "${LAMBDAS[@]}"; do
    for margin in "${MARGINS[@]}"; do
      for pred_len in "${PRED_LENS[@]}"; do
        for warmup in "${WARMUPS[@]}"; do
          echo "[TUNE] time_series loss=$reg_loss lambda=$lambda_imitation margin=$margin pred_len=$pred_len warmup=$warmup"
          REGRESSION_IMITATION_LOSSES="$reg_loss" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          WARMUP_STUDYGROUP="$warmup" \
          PRED_LENS="$pred_len" \
          METHODS="dml studygroup" \
          DATASETS="etth1 electricity weather" \
          MODEL_PAIRS="transformer:transformer dlinear:dlinear transformer:dlinear" \
          SEEDS="0 1 2 3 4" \
          EPOCHS="80" \
          BATCH_SIZE="64" \
          DEVICE="cuda" \
          OUTPUT_DIR="$SWEEP_OUTPUT_DIR" \
          RUN_VISUALIZATION="0" \
          bash scripts/simple/run_simple_time_series.sh
        done
      done
    done
  done
done

if [[ -d "$SWEEP_OUTPUT_DIR/time_series" ]]; then
  python -m src.utils.visualization \
    --input-dir "$SWEEP_OUTPUT_DIR/time_series" \
    --output-dir "$SWEEP_OUTPUT_DIR"
fi
