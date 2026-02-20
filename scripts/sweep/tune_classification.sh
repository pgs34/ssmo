#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CLASS_LOSSES=(kl js mse_logits)
LAMBDAS=(0.5 1.0 2.0)
MARGINS=(0.0 0.1)
WARMUPS=(3 5)
SWEEP_OUTPUT_DIR="results/sweep/classification/hyper_parameter"

echo "[CONFIG] task=classification_sweep"
echo "[CONFIG] class_losses=${CLASS_LOSSES[*]}"
echo "[CONFIG] lambdas=${LAMBDAS[*]} margins=${MARGINS[*]} warmups=${WARMUPS[*]}"
echo "[CONFIG] methods=naive dml studygroup datasets=mnist cifar10 cifar100"
echo "[CONFIG] models=resnet18 vit_b16 simple_cnn"
echo "[CONFIG] seeds=0 1 2 3 4 epochs=120 batch_size=128 device=cuda"
echo "[CONFIG] output_dir=$SWEEP_OUTPUT_DIR"

for loss in "${CLASS_LOSSES[@]}"; do
  for lambda_imitation in "${LAMBDAS[@]}"; do
    for margin in "${MARGINS[@]}"; do
      for warmup in "${WARMUPS[@]}"; do
        echo "[TUNE] classification loss=$loss lambda=$lambda_imitation margin=$margin warmup=$warmup"
        CLASSIFICATION_IMITATION_LOSS="$loss" \
        LAMBDA_IMITATION="$lambda_imitation" \
        MARGIN="$margin" \
        WARMUP_STUDYGROUP="$warmup" \
        METHODS="naive dml studygroup" \
        DATASETS="mnist cifar10 cifar100" \
        MODELS="resnet18 vit_b16 simple_cnn" \
        SEEDS="0 1 2 3 4" \
        EPOCHS="120" \
        BATCH_SIZE="128" \
        DEVICE="cuda" \
        OUTPUT_DIR="$SWEEP_OUTPUT_DIR" \
        MAX_TRAIN_BATCHES="" \
        MAX_VAL_BATCHES="" \
        bash scripts/simple/run_simple_classification.sh
      done
    done
  done
done
