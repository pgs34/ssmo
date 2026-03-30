#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker1}"
mkdir -p "$LOG_DIR"

if [[ -z "${CLASSIFICATION_V14_MAIN_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V14_MAIN_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V14_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V14_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V14_ALT_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V14_ALT_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V14_ALT_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V14_ALT_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

run_main() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
  GPU="${CLASSIFICATION_GPU0:-0}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_V14_MAIN_MODEL_PAIRS:-ode_cnn:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_V14_MAIN_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v14_main}" \
  CASE_SPECS="${CLASSIFICATION_V14_MAIN_CASE_SPECS:-main_pb4_t30:0.30:0.10:0.020:0.02:0.35:4:6.0 main_pb6_t40:0.40:0.10:0.015:0.02:0.40:6:8.0}" \
  INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V14_MAIN_INIT_CHECKPOINT_TEMPLATE}" \
  PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V14_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE}" \
  LOG_DIR="results/logs/classification_ssml_reweight_v14_main" \
  run_logged_job \
    "worker1/classification_reweight_v14_main_gpu${CLASSIFICATION_GPU0:-0}" \
    "$LOG_DIR/classification_reweight_v14_main_gpu${CLASSIFICATION_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_classification_ssml_reweight_v14.sh
}

run_alt() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
  GPU="${CLASSIFICATION_GPU1:-1}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_V14_ALT_MODEL_PAIRS:-ode_cnn:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_V14_ALT_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v14_alt}" \
  CASE_SPECS="${CLASSIFICATION_V14_ALT_CASE_SPECS:-alt_pb5_t35:0.35:0.08:0.020:0.03:0.40:5:6.0 alt_pb6_t45:0.45:0.08:0.015:0.03:0.45:6:8.0}" \
  INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V14_ALT_INIT_CHECKPOINT_TEMPLATE}" \
  PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V14_ALT_PEER_INIT_CHECKPOINT_TEMPLATE}" \
  LOG_DIR="results/logs/classification_ssml_reweight_v14_alt" \
  run_logged_job \
    "worker1/classification_reweight_v14_alt_gpu${CLASSIFICATION_GPU1:-1}" \
    "$LOG_DIR/classification_reweight_v14_alt_gpu${CLASSIFICATION_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_classification_ssml_reweight_v14.sh
}

run_main &
PID0=$!
run_alt &
PID1=$!

echo "[worker1_classification_ssml_reweight_v14] started main pid=$PID0 gpu=${CLASSIFICATION_GPU0:-0}"
echo "[worker1_classification_ssml_reweight_v14] started alt pid=$PID1 gpu=${CLASSIFICATION_GPU1:-1}"

wait "$PID0"
wait "$PID1"
echo "[worker1_classification_ssml_reweight_v14] all jobs finished"
