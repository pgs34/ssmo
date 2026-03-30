#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

if [[ -z "${CLASSIFICATION_V20_BACKUP_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V20_BACKUP_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V20_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V20_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/{dataset}/resnet34_gelu_ssml_{classification_imitation_loss}_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU:-0}" \
GPU="${CLASSIFICATION_GPU:-0}" \
DATASETS="cifar100" \
MODEL_PAIRS="${CLASSIFICATION_V20_BACKUP_MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}" \
OUTPUT_ROOT="${CLASSIFICATION_V20_BACKUP_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v20_backup}" \
INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V20_BACKUP_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V20_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE}" \
CASE_SPECS="${CLASSIFICATION_V20_BACKUP_CASE_SPECS:-safe_pb18_gap12:0.18:0.05:0.010:0.015:0.45:4:6.0:0.0005:0.40:0.12 safe_pb22_gap15:0.22:0.05:0.010:0.020:0.50:4:6.0:0.0005:0.38:0.15}" \
REQUIRE_DISTINCT_PEER="0" \
LOG_DIR="results/logs/classification_ssml_reweight_v20_backup" \
run_logged_job \
  "worker3/classification_reweight_v20_backup_gpu${CLASSIFICATION_GPU:-0}" \
  "$LOG_DIR/classification_reweight_v20_backup_gpu${CLASSIFICATION_GPU:-0}.log" \
  bash scripts/paper_rerun/run_classification_ssml_reweight_v20.sh

echo "[worker3_classification_ssml_reweight_v20_backup] job finished"
