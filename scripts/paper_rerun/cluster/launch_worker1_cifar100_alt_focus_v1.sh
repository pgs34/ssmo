#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker1}"
mkdir -p "$LOG_DIR"

ALT_FOCUS_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_ALT_FOCUS_INIT_CHECKPOINT_TEMPLATE:-results/classification_neural_ode_cifar100_v11_backup/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt}"
ALT_FOCUS_PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_ALT_FOCUS_PEER_INIT_CHECKPOINT_TEMPLATE:-results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt}"

run_gpu0() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
  GPU="${CLASSIFICATION_GPU0:-0}" \
  OUTPUT_ROOT="${CLASSIFICATION_ALT_FOCUS_MAIN_OUTPUT_ROOT:-results/classification_cifar100_alt_focus_v1/main}" \
  CASE_SPECS="${CLASSIFICATION_ALT_FOCUS_MAIN_CASE_SPECS:-conf_pb22_aw5e4:0.22:0.04:0.012:0.020:0.35:4:6.0:0.0005 conf_pb25_aw5e4:0.25:0.04:0.012:0.020:0.35:4:6.0:0.0005 conf_pb28_aw5e4:0.28:0.04:0.012:0.020:0.35:4:6.0:0.0005}" \
  INIT_CHECKPOINT_TEMPLATE="$ALT_FOCUS_INIT_CHECKPOINT_TEMPLATE" \
  PEER_INIT_CHECKPOINT_TEMPLATE="$ALT_FOCUS_PEER_INIT_CHECKPOINT_TEMPLATE" \
  run_logged_job \
    "worker1/classification_cifar100_alt_focus_main_gpu${CLASSIFICATION_GPU0:-0}" \
    "$LOG_DIR/classification_cifar100_alt_focus_main_gpu${CLASSIFICATION_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_classification_cifar100_alt_focus_v1.sh
}

run_gpu1() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
  GPU="${CLASSIFICATION_GPU1:-1}" \
  OUTPUT_ROOT="${CLASSIFICATION_ALT_FOCUS_AUX_OUTPUT_ROOT:-results/classification_cifar100_alt_focus_v1/aux}" \
  CASE_SPECS="${CLASSIFICATION_ALT_FOCUS_AUX_CASE_SPECS:-conf_pb25_thr38:0.25:0.04:0.012:0.020:0.38:4:6.0:0.0005 conf_pb25_thr40:0.25:0.04:0.012:0.020:0.40:4:6.0:0.0005 conf_pb25_aw7e4:0.25:0.04:0.012:0.020:0.35:4:6.0:0.0007}" \
  INIT_CHECKPOINT_TEMPLATE="$ALT_FOCUS_INIT_CHECKPOINT_TEMPLATE" \
  PEER_INIT_CHECKPOINT_TEMPLATE="$ALT_FOCUS_PEER_INIT_CHECKPOINT_TEMPLATE" \
  run_logged_job \
    "worker1/classification_cifar100_alt_focus_aux_gpu${CLASSIFICATION_GPU1:-1}" \
    "$LOG_DIR/classification_cifar100_alt_focus_aux_gpu${CLASSIFICATION_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_classification_cifar100_alt_focus_v1.sh
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

echo "[worker1_cifar100_alt_focus_v1] started gpu0 pid=$PID0 gpu=${CLASSIFICATION_GPU0:-0}"
echo "[worker1_cifar100_alt_focus_v1] started gpu1 pid=$PID1 gpu=${CLASSIFICATION_GPU1:-1}"

wait "$PID0"
wait "$PID1"
echo "[worker1_cifar100_alt_focus_v1] all jobs finished"
