#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_alt_focus_v2}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_alt_focus_v2}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_backup/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

CASE_SPECS="${CASE_SPECS:-conf_pb24_thr33_aw4e4:0.24:0.04:0.012:0.020:0.33:4:6.0:0.0004 conf_pb24_thr35_aw4e4:0.24:0.04:0.012:0.020:0.35:4:6.0:0.0004 conf_pb26_thr33_aw4e4:0.26:0.04:0.012:0.020:0.33:4:6.0:0.0004 conf_pb24_l1e2_aw4e4:0.24:0.04:0.010:0.020:0.35:4:6.0:0.0004 conf_pb24_thr35_aw6e4:0.24:0.04:0.012:0.020:0.35:4:6.0:0.0006}"

CUDA_VISIBLE_DEVICES="$GPU" \
GPU="$GPU" \
DATASETS="cifar100" \
MODEL_PAIRS="$MODEL_PAIRS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
CASE_SPECS="$CASE_SPECS" \
SEEDS="$SEEDS" \
REQUIRE_DISTINCT_PEER="0" \
LOG_DIR="$LOG_DIR" \
bash scripts/paper_rerun/run_classification_ssml_reweight_v17.sh
