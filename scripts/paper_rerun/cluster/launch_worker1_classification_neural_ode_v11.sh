#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker1}"
mkdir -p "$LOG_DIR"

run_cifar10() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
  GPU="${CLASSIFICATION_GPU0:-0}" \
  DATASETS="cifar10" \
  MODEL_PAIRS="${CLASSIFICATION_CIFAR10_MODEL_PAIRS:-ode_cnn:ode_cnn ode_cnn:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_CIFAR10_OUTPUT_ROOT:-results/classification_neural_ode_cifar10_v11}" \
  CIFAR10_TOPKS="${CLASSIFICATION_CIFAR10_TOPKS:-0.15 0.2}" \
  CIFAR100_TOPKS="" \
  BATCH_SIZE="${CLASSIFICATION_CIFAR10_BATCH_SIZE:-96}" \
  SSML_GATE_SCORE_MODE="${CLASSIFICATION_CIFAR10_SSML_GATE_SCORE_MODE:-peer_better_true_prob_gap_weighted}" \
  SSML_PEER_TRUE_PROB_THRESHOLD="${CLASSIFICATION_CIFAR10_SSML_PEER_TRUE_PROB_THRESHOLD:-0.3}" \
  run_logged_job \
    "worker1/classification_neural_ode_v11_cifar10_gpu${CLASSIFICATION_GPU0:-0}" \
    "$LOG_DIR/classification_neural_ode_v11_cifar10_gpu${CLASSIFICATION_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_classification_neural_ode_v11.sh
}

run_cifar100() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
  GPU="${CLASSIFICATION_GPU1:-1}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_CIFAR100_MODEL_PAIRS:-ode_cnn:resnet34_gelu resnet34_gelu:ode_cnn}" \
  OUTPUT_ROOT="${CLASSIFICATION_CIFAR100_OUTPUT_ROOT:-results/classification_neural_ode_cifar100_v11_main}" \
  CIFAR10_TOPKS="" \
  CIFAR100_TOPKS="${CLASSIFICATION_CIFAR100_TOPKS:-0.2 0.3}" \
  BATCH_SIZE="${CLASSIFICATION_CIFAR100_BATCH_SIZE:-96}" \
  SSML_GATE_SCORE_MODE="${CLASSIFICATION_CIFAR100_SSML_GATE_SCORE_MODE:-peer_better_true_prob_gap_weighted}" \
  SSML_PEER_TRUE_PROB_THRESHOLD="${CLASSIFICATION_CIFAR100_SSML_PEER_TRUE_PROB_THRESHOLD:-0.45}" \
  SSML_MARGIN_CIFAR100="${CLASSIFICATION_CIFAR100_SSML_MARGIN:-0.05}" \
  run_logged_job \
    "worker1/classification_neural_ode_v11_cifar100_gpu${CLASSIFICATION_GPU1:-1}" \
    "$LOG_DIR/classification_neural_ode_v11_cifar100_gpu${CLASSIFICATION_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_classification_neural_ode_v11.sh
}

run_cifar10 &
PID0=$!
run_cifar100 &
PID1=$!

echo "[worker1_classification_neural_ode_v11] started cifar10 pid=$PID0 gpu=${CLASSIFICATION_GPU0:-0}"
echo "[worker1_classification_neural_ode_v11] started cifar100 pid=$PID1 gpu=${CLASSIFICATION_GPU1:-1}"

wait "$PID0"
wait "$PID1"
echo "[worker1_classification_neural_ode_v11] all jobs finished"
