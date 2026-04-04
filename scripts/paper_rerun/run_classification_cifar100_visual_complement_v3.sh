#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_visual_complement_v3}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-1}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
POOL_ROOT="${POOL_ROOT:-results/classification_cifar100_visual_complement_v3_pool}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_visual_complement_v3}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
SSML_AUG_CONSISTENCY_WEIGHT="${SSML_AUG_CONSISTENCY_WEIGHT:-0.75}"
SSML_AUG_CONSISTENCY_SHIFT="${SSML_AUG_CONSISTENCY_SHIFT:-2}"
SSML_AUG_CONSISTENCY_FLIP_PROB="${SSML_AUG_CONSISTENCY_FLIP_PROB:-0.50}"
SSML_AUG_CONSISTENCY_NOISE_STD="${SSML_AUG_CONSISTENCY_NOISE_STD:-0.03}"
RUN_INDEPENDENT_POOL="${RUN_INDEPENDENT_POOL:-1}"
CASE_SPECS="${CASE_SPECS:-viscomp35_thr45_gap2_spmax80_aw2e4:0.35:0.02:0.012:0.000:0.45:0.02:10:6.0:0.0002:0.80 viscomp45_thr40_gap2_spmax85_aw2e4:0.45:0.02:0.012:0.000:0.40:0.02:12:6.0:0.0002:0.85 viscomp35_thr50_gap3_spmax80_aw4e4:0.35:0.02:0.015:0.000:0.50:0.03:10:8.0:0.0004:0.80}"

BEST_CKPT_TEMPLATE="${BEST_CKPT_TEMPLATE:-}"
if [[ -z "$BEST_CKPT_TEMPLATE" ]]; then
  BEST_CKPT_TEMPLATE="$POOL_ROOT/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
fi

run_logged() {
  local label="$1"
  shift
  run_logged_job \
    "classification_cifar100_visual_complement_v3/$label" \
    "$LOG_DIR/$label.log" \
    "$@"
}

echo "[classification_cifar100_visual_complement_v3] gpu=$GPU seeds=$SEEDS"
echo "[classification_cifar100_visual_complement_v3] pool_root=$POOL_ROOT"
echo "[classification_cifar100_visual_complement_v3] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_visual_complement_v3] case_specs=$CASE_SPECS"
echo "[classification_cifar100_visual_complement_v3] aug_consistency_w=$SSML_AUG_CONSISTENCY_WEIGHT shift=$SSML_AUG_CONSISTENCY_SHIFT flip=$SSML_AUG_CONSISTENCY_FLIP_PROB noise=$SSML_AUG_CONSISTENCY_NOISE_STD"
echo "[classification_cifar100_visual_complement_v3] run_independent_pool=$RUN_INDEPENDENT_POOL"
echo "[classification_cifar100_visual_complement_v3] best_ckpt_template=$BEST_CKPT_TEMPLATE"

if [[ "$RUN_INDEPENDENT_POOL" == "1" ]]; then
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
fi

for spec in $CASE_SPECS; do
  IFS=':' read -r label topk_ratio alpha lambda margin prob_threshold prob_gap per_class_budget distill_temperature anchor_weight student_true_prob_max <<< "$spec"
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
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      WARMUP_EPOCHS="1" \
      IMITATION_DECAY_START_EPOCH="15" \
      IMITATION_DECAY_END_EPOCH="45" \
      IMITATION_DECAY_MIN_SCALE="0.35" \
      SSML_TOPK_SCOPE="positive" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      SSML_GATE_SCORE_MODE="peer_confident_student_uncertain" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_GUIDANCE_MODE="hybrid" \
      SSML_PEER_CORRECT_ONLY="1" \
      SSML_STUDENT_INCORRECT_ONLY="0" \
      SSML_STUDENT_TRUE_PROB_MAX="$student_true_prob_max" \
      SSML_DISAGREEMENT_ONLY="0" \
      SSML_CLASS_BALANCED_TOPK="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_WORSE_ONLY_UPDATE="0" \
      SSML_TOPK_RATIO="$topk_ratio" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
      LAMBDA_IMITATION="$lambda" \
      MARGIN="$margin" \
      SSML_PEER_TRUE_PROB_THRESHOLD="$prob_threshold" \
      SSML_PEER_STUDENT_PROB_GAP_MIN="$prob_gap" \
      SSML_AUG_CONSISTENCY_WEIGHT="$SSML_AUG_CONSISTENCY_WEIGHT" \
      SSML_AUG_CONSISTENCY_SHIFT="$SSML_AUG_CONSISTENCY_SHIFT" \
      SSML_AUG_CONSISTENCY_FLIP_PROB="$SSML_AUG_CONSISTENCY_FLIP_PROB" \
      SSML_AUG_CONSISTENCY_NOISE_STD="$SSML_AUG_CONSISTENCY_NOISE_STD" \
      SSML_PER_CLASS_BUDGET="$per_class_budget" \
      DISTILL_TEMPERATURE="$distill_temperature" \
      SSML_ANCHOR_WEIGHT="$anchor_weight" \
      bash scripts/paper_rerun/run_core_classification.sh
done
