#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker1}"
mkdir -p "$LOG_DIR"

if [[ -z "${CLASSIFICATION_V15_MAIN_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V15_MAIN_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V15_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V15_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V15_ALT_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V15_ALT_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V15_ALT_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V15_ALT_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

run_main() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
  GPU="${CLASSIFICATION_GPU0:-0}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_V15_MAIN_MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_V15_MAIN_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v15_main}" \
  CASE_SPECS="${CLASSIFICATION_V15_MAIN_CASE_SPECS:-r34_pb25_aw5e4:0.25:0.05:0.010:0.015:0.30:4:6.0:0.0005 r34_pb35_aw1e3:0.35:0.05:0.015:0.020:0.35:5:8.0:0.0010}" \
  INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V15_MAIN_INIT_CHECKPOINT_TEMPLATE}" \
  PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V15_MAIN_PEER_INIT_CHECKPOINT_TEMPLATE}" \
  REQUIRE_DISTINCT_PEER="0" \
  LOG_DIR="results/logs/classification_ssml_reweight_v15_main" \
  run_logged_job \
    "worker1/classification_reweight_v15_main_gpu${CLASSIFICATION_GPU0:-0}" \
    "$LOG_DIR/classification_reweight_v15_main_gpu${CLASSIFICATION_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_classification_ssml_reweight_v15.sh
}

run_alt() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
  GPU="${CLASSIFICATION_GPU1:-1}" \
  DATASETS="cifar100" \
  MODEL_PAIRS="${CLASSIFICATION_V15_ALT_MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}" \
  OUTPUT_ROOT="${CLASSIFICATION_V15_ALT_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v15_alt}" \
  CASE_SPECS="${CLASSIFICATION_V15_ALT_CASE_SPECS:-r34_pb30_aw5e4:0.30:0.04:0.012:0.015:0.30:4:6.0:0.0005 r34_pb40_aw1e3:0.40:0.04:0.015:0.020:0.35:6:8.0:0.0010}" \
  INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V15_ALT_INIT_CHECKPOINT_TEMPLATE}" \
  PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V15_ALT_PEER_INIT_CHECKPOINT_TEMPLATE}" \
  REQUIRE_DISTINCT_PEER="0" \
  LOG_DIR="results/logs/classification_ssml_reweight_v15_alt" \
  run_logged_job \
    "worker1/classification_reweight_v15_alt_gpu${CLASSIFICATION_GPU1:-1}" \
    "$LOG_DIR/classification_reweight_v15_alt_gpu${CLASSIFICATION_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_classification_ssml_reweight_v15.sh
}

run_main &
PID0=$!
run_alt &
PID1=$!

echo "[worker1_classification_ssml_reweight_v15] started main pid=$PID0 gpu=${CLASSIFICATION_GPU0:-0}"
echo "[worker1_classification_ssml_reweight_v15] started alt pid=$PID1 gpu=${CLASSIFICATION_GPU1:-1}"

wait "$PID0"
wait "$PID1"
echo "[worker1_classification_ssml_reweight_v15] all jobs finished"
