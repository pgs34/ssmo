#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_bestckpt_followup_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-1}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
POOL_ROOT="${POOL_ROOT:-results/classification_cifar100_bestckpt_pool_v1}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-results/classification_cifar100_bestckpt_followup_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DML_CASE_SPECS="${DML_CASE_SPECS:-dml_l2e2_t4:0.02:4.0:0.00 dml_l2e2_t6:0.02:6.0:0.00}"
SSML_CASE_SPECS="${SSML_CASE_SPECS:-bestinit_pb26_thr33_gap0_aw4e4:0.26:0.04:0.012:0.020:0.33:0.00:4:6.0:0.0004 bestinit_pb26_thr33_gap2_aw4e4:0.26:0.04:0.012:0.020:0.33:0.02:4:6.0:0.0004 bestinit_pb30_thr35_gap2_aw6e4:0.30:0.04:0.012:0.020:0.35:0.02:5:6.0:0.0006}"

BEST_CKPT_TEMPLATE="$POOL_ROOT/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"

run_logged() {
  local label="$1"
  shift
  run_logged_job \
    "classification_cifar100_bestckpt_followup_v1/$label" \
    "$LOG_DIR/$label.log" \
    "$@"
}

echo "[classification_cifar100_bestckpt_followup_v1] gpu=$GPU seeds=$SEEDS"
echo "[classification_cifar100_bestckpt_followup_v1] pool_root=$POOL_ROOT"
echo "[classification_cifar100_bestckpt_followup_v1] followup_root=$FOLLOWUP_ROOT"

run_logged \
  "independent_pool" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    METHODS="independent" \
    DATASETS="cifar100" \
    INDEPENDENT_MODELS="resnet34_gelu" \
    MODEL_PAIRS="$MODEL_PAIRS" \
    REQUIRE_DISTINCT_PEER="0" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DOWNLOAD="$DOWNLOAD" \
    OUTPUT_DIR="$POOL_ROOT/classification" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    bash scripts/paper_rerun/run_core_classification.sh

for spec in $DML_CASE_SPECS; do
  IFS=':' read -r label lambda temperature margin <<< "$spec"
  run_logged \
    "$label" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="dml" \
      DATASETS="cifar100" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      OUTPUT_DIR="$FOLLOWUP_ROOT/$label" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      DISTILL_TEMPERATURE="$temperature" \
      LAMBDA_IMITATION="$lambda" \
      MARGIN="$margin" \
      bash scripts/paper_rerun/run_core_classification.sh
done

for spec in $SSML_CASE_SPECS; do
  IFS=':' read -r label topk_ratio alpha lambda margin prob_threshold prob_gap per_class_budget distill_temperature anchor_weight <<< "$spec"
  run_logged \
    "$label" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="cifar100" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      OUTPUT_DIR="$FOLLOWUP_ROOT/$label" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      WARMUP_EPOCHS="3" \
      IMITATION_DECAY_START_EPOCH="10" \
      IMITATION_DECAY_END_EPOCH="40" \
      IMITATION_DECAY_MIN_SCALE="0.25" \
      SSML_TOPK_SCOPE="positive" \
      SSML_SUPERVISED_WEIGHT_MODE="binary" \
      SSML_GATE_SCORE_MODE="useful_hard_sample_confident" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_GUIDANCE_MODE="hybrid" \
      SSML_PEER_CORRECT_ONLY="1" \
      SSML_STUDENT_INCORRECT_ONLY="1" \
      SSML_STUDENT_TRUE_PROB_MAX="0.45" \
      SSML_DISAGREEMENT_ONLY="1" \
      SSML_CLASS_BALANCED_TOPK="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_WORSE_ONLY_UPDATE="1" \
      SSML_TOPK_RATIO="$topk_ratio" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
      LAMBDA_IMITATION="$lambda" \
      MARGIN="$margin" \
      SSML_PEER_TRUE_PROB_THRESHOLD="$prob_threshold" \
      SSML_PEER_STUDENT_PROB_GAP_MIN="$prob_gap" \
      SSML_PER_CLASS_BUDGET="$per_class_budget" \
      DISTILL_TEMPERATURE="$distill_temperature" \
      SSML_ANCHOR_WEIGHT="$anchor_weight" \
      bash scripts/paper_rerun/run_core_classification.sh
done
