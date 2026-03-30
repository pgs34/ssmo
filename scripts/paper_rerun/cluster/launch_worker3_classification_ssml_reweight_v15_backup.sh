#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

if [[ -z "${CLASSIFICATION_V15_BACKUP_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V15_BACKUP_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${CLASSIFICATION_V15_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  CLASSIFICATION_V15_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU:-0}" \
GPU="${CLASSIFICATION_GPU:-0}" \
DATASETS="cifar100" \
MODEL_PAIRS="${CLASSIFICATION_V15_BACKUP_MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}" \
OUTPUT_ROOT="${CLASSIFICATION_V15_BACKUP_OUTPUT_ROOT:-results/classification_ssml_reweight_cifar100_v15_backup}" \
CASE_SPECS="${CLASSIFICATION_V15_BACKUP_CASE_SPECS:-r34_pb45_aw1e3:0.45:0.03:0.015:0.025:0.40:6:8.0:0.0010 r34_pb55_aw2e3:0.55:0.03:0.020:0.025:0.45:8:8.0:0.0020}" \
INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V15_BACKUP_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${CLASSIFICATION_V15_BACKUP_PEER_INIT_CHECKPOINT_TEMPLATE}" \
REQUIRE_DISTINCT_PEER="0" \
LOG_DIR="results/logs/classification_ssml_reweight_v15_backup" \
run_logged_job \
  "worker3/classification_reweight_v15_backup_gpu${CLASSIFICATION_GPU:-0}" \
  "$LOG_DIR/classification_reweight_v15_backup_gpu${CLASSIFICATION_GPU:-0}.log" \
  bash scripts/paper_rerun/run_classification_ssml_reweight_v15.sh

echo "[worker3_classification_ssml_reweight_v15_backup] job finished"
