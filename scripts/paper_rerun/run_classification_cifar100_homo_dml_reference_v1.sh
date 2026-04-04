#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_homo_dml_reference_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_homo_dml_reference_v1}"
DATASETS="${DATASETS:-cifar100}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
DOWNLOAD="${DOWNLOAD:-1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
CASE_SPECS="${CASE_SPECS:-dml_l2e2_t4:0.02:4.0:0.00 dml_l4e2_t4:0.04:4.0:0.00 dml_l2e2_t6:0.02:6.0:0.00 dml_l4e2_t6:0.04:6.0:0.00}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_backup/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "classification_cifar100_homo_dml_reference_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="dml" \
      DATASETS="$DATASETS" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_cifar100_homo_dml_reference_v1] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_homo_dml_reference_v1] gpu=$GPU seeds=$SEEDS"
echo "[classification_cifar100_homo_dml_reference_v1] model_pairs=$MODEL_PAIRS"
echo "[classification_cifar100_homo_dml_reference_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[classification_cifar100_homo_dml_reference_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[classification_cifar100_homo_dml_reference_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lambda temperature margin <<< "$spec"
  run_case \
    "$label" \
    OUTPUT_DIR="$OUTPUT_ROOT/$label" \
    LAMBDA_IMITATION="$lambda" \
    DISTILL_TEMPERATURE="$temperature" \
    MARGIN="$margin" \
    WARMUP_EPOCHS="0" \
    IMITATION_DECAY_START_EPOCH="-1" \
    IMITATION_DECAY_END_EPOCH="-1" \
    IMITATION_DECAY_MIN_SCALE="1.0"
done
