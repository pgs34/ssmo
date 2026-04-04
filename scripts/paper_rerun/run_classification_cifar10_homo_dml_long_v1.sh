#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar10_homo_dml_long_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar10_homo_dml_long_v1}"
SUMMARY_PLOT_ROOT="${SUMMARY_PLOT_ROOT:-results/plots/classification_cifar10_homo_dml_long_v1}"
DOWNLOAD="${DOWNLOAD:-0}"
REFRESH_TOP_LEVEL="${REFRESH_TOP_LEVEL:-0}"

echo "[classification_cifar10_homo_dml_long_v1] output_root=$OUTPUT_ROOT"
echo "[classification_cifar10_homo_dml_long_v1] summary_plot_root=$SUMMARY_PLOT_ROOT"
echo "[classification_cifar10_homo_dml_long_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"

run_locked_job \
  "classification_cifar10_homo_dml_long_v1" \
  "classification_cifar10_homo_dml_long_v1/baseline_dml" \
  "$LOG_DIR/baseline_dml.log" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    OUTPUT_DIR="$OUTPUT_ROOT" \
    DATASETS="cifar10" \
    METHODS="dml" \
    MODEL_PAIRS="resnet18:resnet18" \
    REQUIRE_DISTINCT_PEER="0" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DOWNLOAD="$DOWNLOAD" \
    CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}" \
    DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-4.0}" \
    LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.02}" \
    MARGIN="${MARGIN:-0.05}" \
    SSML_TOPK_RATIO="0.0" \
    WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}" \
    IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}" \
    IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-85}" \
    IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.05}" \
    bash scripts/paper_rerun/run_core_classification.sh

echo "[classification_cifar10_homo_dml_long_v1] refreshing summary plots"
bash scripts/paper_rerun/refresh_summary_plots.sh "$OUTPUT_ROOT" "$SUMMARY_PLOT_ROOT"

if [[ "$REFRESH_TOP_LEVEL" == "1" ]]; then
  echo "[classification_cifar10_homo_dml_long_v1] refreshing top-level plots"
  bash scripts/paper_rerun/refresh_top_level_best_plots.sh
fi

echo "[classification_cifar10_homo_dml_long_v1] done"

