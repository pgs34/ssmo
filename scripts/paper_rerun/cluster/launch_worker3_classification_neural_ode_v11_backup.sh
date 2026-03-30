#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU:-0}" \
GPU="${CLASSIFICATION_GPU:-0}" \
DATASETS="cifar100" \
MODEL_PAIRS="${CLASSIFICATION_BACKUP_MODEL_PAIRS:-ode_cnn:ode_cnn resnet34_gelu:ode_cnn}" \
OUTPUT_ROOT="${CLASSIFICATION_BACKUP_OUTPUT_ROOT:-results/classification_neural_ode_cifar100_v11_backup}" \
CIFAR10_TOPKS="" \
CIFAR100_TOPKS="${CLASSIFICATION_BACKUP_CIFAR100_TOPKS:-0.3 0.4}" \
BATCH_SIZE="${CLASSIFICATION_BACKUP_BATCH_SIZE:-96}" \
SSML_GATE_SCORE_MODE="${CLASSIFICATION_BACKUP_SSML_GATE_SCORE_MODE:-peer_better_true_prob_gap_weighted}" \
SSML_PEER_TRUE_PROB_THRESHOLD="${CLASSIFICATION_BACKUP_SSML_PEER_TRUE_PROB_THRESHOLD:-0.4}" \
SSML_MARGIN_CIFAR100="${CLASSIFICATION_BACKUP_SSML_MARGIN:-0.03}" \
run_logged_job \
  "worker3/classification_neural_ode_v11_backup_gpu${CLASSIFICATION_GPU:-0}" \
  "$LOG_DIR/classification_neural_ode_v11_backup_gpu${CLASSIFICATION_GPU:-0}.log" \
  bash scripts/paper_rerun/run_classification_neural_ode_v11.sh

echo "[worker3_classification_neural_ode_v11_backup] job finished"
