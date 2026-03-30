#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

if [[ -z "${CLASSIFICATION_V14_BACKUP_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V14_BACKUP_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V14_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V14_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU:-0}" \
GPU="${CLASSIFICATION_GPU:-0}" \
DATASETS="cifar100" \
MODEL_PAIRS="${CLASSIFICATION_V14_BACKUP_MODEL_PAIRS:-ode_cnn:resnet34_gelu}" \
OUTPUT_ROOT="${CLASSIFICATION_V14_BACKUP_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v14_backup}" \
CASE_SPECS="${CLASSIFICATION_V14_BACKUP_CASE_SPECS:-bak_pb5_t50:0.50:0.06:0.015:0.03:0.45:5:8.0 bak_pb6_t60:0.60:0.05:0.010:0.03:0.50:6:8.0}" \
INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V14_BACKUP_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V14_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE}" \
LOG_DIR="results/logs/classification_ssml_reweight_v14_backup" \
run_logged_job \
  "worker3/classification_reweight_v14_backup_gpu${CLASSIFICATION_GPU:-0}" \
  "$LOG_DIR/classification_reweight_v14_backup_gpu${CLASSIFICATION_GPU:-0}.log" \
  bash scripts/paper_rerun/run_classification_ssml_reweight_v14.sh

echo "[worker3_classification_ssml_reweight_v14_backup] job finished"
