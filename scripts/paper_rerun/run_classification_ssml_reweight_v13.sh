#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_ssml_reweight_v13}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-cifar100}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:ode_cnn}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_ssml_reweight_v13}"
DOWNLOAD="${DOWNLOAD:-1}"

CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-4.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-20}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-60}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.0}"
SSML_TOPK_SCOPE="${SSML_TOPK_SCOPE:-positive}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-binary}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_true_prob_gap_weighted}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-none}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-hybrid}"
SSML_PEER_CORRECT_ONLY="${SSML_PEER_CORRECT_ONLY:-1}"
SSML_STUDENT_INCORRECT_ONLY="${SSML_STUDENT_INCORRECT_ONLY:-1}"
SSML_DISAGREEMENT_ONLY="${SSML_DISAGREEMENT_ONLY:-1}"
SSML_CLASS_BALANCED_TOPK="${SSML_CLASS_BALANCED_TOPK:-1}"
CASE_SPECS="${CASE_SPECS:-pb2_t25:0.25:0.15:0.020:0.03:0.55:2 pb3_t35:0.35:0.10:0.015:0.02:0.50:3}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
fi

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "classification_reweight_v13/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="$DATASETS" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      SSML_TOPK_SCOPE="$SSML_TOPK_SCOPE" \
      SSML_SUPERVISED_WEIGHT_MODE="$SSML_SUPERVISED_WEIGHT_MODE" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      SSML_PEER_CORRECT_ONLY="$SSML_PEER_CORRECT_ONLY" \
      SSML_STUDENT_INCORRECT_ONLY="$SSML_STUDENT_INCORRECT_ONLY" \
      SSML_DISAGREEMENT_ONLY="$SSML_DISAGREEMENT_ONLY" \
      SSML_CLASS_BALANCED_TOPK="$SSML_CLASS_BALANCED_TOPK" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_ssml_reweight_v13] output_root=$OUTPUT_ROOT"
echo "[classification_ssml_reweight_v13] gpu=$GPU datasets=$DATASETS seeds=$SEEDS"
echo "[classification_ssml_reweight_v13] model_pairs=$MODEL_PAIRS"
echo "[classification_ssml_reweight_v13] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[classification_ssml_reweight_v13] gate=$SSML_GATE_SCORE_MODE transform=$SSML_SCORE_TRANSFORM guidance=$SSML_GUIDANCE_MODE"
echo "[classification_ssml_reweight_v13] peer_correct_only=$SSML_PEER_CORRECT_ONLY student_incorrect_only=$SSML_STUDENT_INCORRECT_ONLY disagreement_only=$SSML_DISAGREEMENT_ONLY class_balanced_topk=$SSML_CLASS_BALANCED_TOPK"
echo "[classification_ssml_reweight_v13] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[classification_ssml_reweight_v13] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[classification_ssml_reweight_v13] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label topk_ratio alpha lambda_imitation margin prob_threshold per_class_budget <<< "$spec"
  run_job \
    "$label" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    LAMBDA_IMITATION="$lambda_imitation" \
    MARGIN="$margin" \
    SSML_PEER_TRUE_PROB_THRESHOLD="$prob_threshold" \
    SSML_PER_CLASS_BUDGET="$per_class_budget" \
    OUTPUT_DIR="$OUTPUT_ROOT/$label"
done
