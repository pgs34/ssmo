#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker1}"
mkdir -p "$LOG_DIR"

if [[ -z "${CLASSIFICATION_V16_MAIN_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V16_MAIN_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V16_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V16_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V16_ALT_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V16_ALT_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V16_ALT_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V16_ALT_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

run_main() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
  GPU="${CLASSIFICATION_GPU0:-0}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_V16_MAIN_MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_V16_MAIN_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v16_main}" \
  CASE_SPECS="${CLASSIFICATION_V16_MAIN_CASE_SPECS:-uh_pb30_aw5e4:0.30:0.04:0.010:0.030:0.40:4:6.0:0.0005 uh_pb40_aw1e3:0.40:0.03:0.015:0.040:0.45:6:8.0:0.0010}" \
  INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V16_MAIN_INIT_CHECKPOINT_TEMPLATE}" \
  PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V16_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE}" \
  REQUIRE_DISTINCT_PEER="0" \
  LOG_DIR="results/logs/classification_ssml_reweight_v16_main" \
  run_logged_job \
    "worker1/classification_reweight_v16_main_gpu${CLASSIFICATION_GPU0:-0}" \
    "$LOG_DIR/classification_reweight_v16_main_gpu${CLASSIFICATION_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_classification_ssml_reweight_v16.sh
}

run_alt() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
  GPU="${CLASSIFICATION_GPU1:-1}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_V16_ALT_MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_V16_ALT_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v16_alt}" \
  CASE_SPECS="${CLASSIFICATION_V16_ALT_CASE_SPECS:-uh_pb25_aw5e4:0.25:0.05:0.010:0.030:0.35:4:6.0:0.0005 uh_pb35_aw1e3:0.35:0.04:0.012:0.035:0.40:5:8.0:0.0010}" \
  INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V16_ALT_INIT_CHECKPOINT_TEMPLATE}" \
  PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V16_ALT_PEER_INIT_CHECKPOINT_TEMPLATE}" \
  REQUIRE_DISTINCT_PEER="0" \
  LOG_DIR="results/logs/classification_ssml_reweight_v16_alt" \
  run_logged_job \
    "worker1/classification_reweight_v16_alt_gpu${CLASSIFICATION_GPU1:-1}" \
    "$LOG_DIR/classification_reweight_v16_alt_gpu${CLASSIFICATION_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_classification_ssml_reweight_v16.sh
}

run_main &
PID0=$!
run_alt &
PID1=$!

echo "[worker1_classification_ssml_reweight_v16] started main pid=$PID0 gpu=${CLASSIFICATION_GPU0:-0}"
echo "[worker1_classification_ssml_reweight_v16] started alt pid=$PID1 gpu=${CLASSIFICATION_GPU1:-1}"

wait "$PID0"
wait "$PID1"
echo "[worker1_classification_ssml_reweight_v16] all jobs finished"
