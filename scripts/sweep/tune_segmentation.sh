#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SEG_LOSSES=(kl js mse_logits)
LAMBDAS=(0.5 1.0 2.0)
MARGINS=(0.0 0.05)
WARMUPS=(3 5)
SWEEP_OUTPUT_DIR="results/sweep/segmentation/hyper_parameter"

echo "[CONFIG] task=segmentation_sweep"
echo "[CONFIG] seg_losses=${SEG_LOSSES[*]}"
echo "[CONFIG] lambdas=${LAMBDAS[*]} margins=${MARGINS[*]} warmups=${WARMUPS[*]}"
echo "[CONFIG] methods=dml studygroup models=unet deeplabv3_resnet50"
echo "[CONFIG] include_cityscapes=1 seeds=0 1 2"
echo "[CONFIG] epochs=100 batch_size=8 device=cuda"
echo "[CONFIG] output_dir=$SWEEP_OUTPUT_DIR"

for seg_loss in "${SEG_LOSSES[@]}"; do
  for lambda_imitation in "${LAMBDAS[@]}"; do
    for margin in "${MARGINS[@]}"; do
      for warmup in "${WARMUPS[@]}"; do
        echo "[TUNE] segmentation loss=$seg_loss lambda=$lambda_imitation margin=$margin warmup=$warmup"
        SEGMENTATION_IMITATION_LOSS="$seg_loss" \
        LAMBDA_IMITATION="$lambda_imitation" \
        MARGIN="$margin" \
        WARMUP_STUDYGROUP="$warmup" \
        METHODS="dml studygroup" \
        MODELS="unet deeplabv3_resnet50" \
        INCLUDE_CITYSCAPES="1" \
        SEEDS="0 1 2" \
        EPOCHS="100" \
        BATCH_SIZE="8" \
        DEVICE="cuda" \
        OUTPUT_DIR="$SWEEP_OUTPUT_DIR" \
        MAX_TRAIN_BATCHES="" \
        MAX_VAL_BATCHES="" \
        bash scripts/simple/run_simple_segmentation.sh
      done
    done
  done
done
