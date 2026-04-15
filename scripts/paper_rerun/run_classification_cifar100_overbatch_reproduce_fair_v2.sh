#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_overbatch_reproduce_fair_v2}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-3072}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_overbatch_reproduce_fair_v2}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-scaled_fair_bs${BATCH_SIZE}}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"
CASE_SPECS="${CASE_SPECS:-oxtra44_thr40_gap1_pc16_aug100:0.44:0.025:0.012:0.000:0.40:0.01:16:5.5:0.0002:0.92:1.00:4:0.50:0.04:1:10:35:0.25 oxtra38_thr45_gap1_pc12_aug100:0.38:0.015:0.015:0.000:0.45:0.01:12:6.0:0.0003:0.95:1.00:2:0.50:0.03:1:15:45:0.35 oxtra42_thr42_gap1_pc18_aug125:0.42:0.02:0.018:0.000:0.42:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:1:12:36:0.25}"
if [[ -z "${BEST_CKPT_TEMPLATE:-}" ]]; then
  BEST_CKPT_TEMPLATE="results/classification_cifar100_bestckpt_pool_v1/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
fi

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "${OUTPUT_ROOT}/${label}/seeds_${SEEDS}" \
    "classification_cifar100_overbatch_reproduce_fair_v2/$label" \
    "$LOG_DIR/$label.log" \
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
      PROTOCOL_ID="$PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      FREEZE_BN_STATS="1" \
      SSML_TOPK_SCOPE="positive" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      SSML_GATE_SCORE_MODE="peer_confident_student_uncertain" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_GUIDANCE_MODE="hybrid" \
      SSML_PEER_CORRECT_ONLY="1" \
      SSML_STUDENT_INCORRECT_ONLY="0" \
      SSML_DISAGREEMENT_ONLY="0" \
      SSML_CLASS_BALANCED_TOPK="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_WORSE_ONLY_UPDATE="0" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_cifar100_overbatch_reproduce_fair_v2] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[classification_cifar100_overbatch_reproduce_fair_v2] protocol_id=$PROTOCOL_ID hardware_profile=$HARDWARE_PROFILE"
echo "[classification_cifar100_overbatch_reproduce_fair_v2] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_overbatch_reproduce_fair_v2] best_ckpt_template=$BEST_CKPT_TEMPLATE"
echo "[classification_cifar100_overbatch_reproduce_fair_v2] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label topk_ratio alpha lambda margin prob_threshold prob_gap per_class_budget distill_temperature anchor_weight student_true_prob_max aug_weight aug_shift aug_flip aug_noise warmup decay_start decay_end decay_min_scale <<< "$spec"
  run_case \
    "$label" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    LAMBDA_IMITATION="$lambda" \
    MARGIN="$margin" \
    SSML_PEER_TRUE_PROB_THRESHOLD="$prob_threshold" \
    SSML_PEER_STUDENT_PROB_GAP_MIN="$prob_gap" \
    SSML_PER_CLASS_BUDGET="$per_class_budget" \
    DISTILL_TEMPERATURE="$distill_temperature" \
    SSML_ANCHOR_WEIGHT="$anchor_weight" \
    SSML_STUDENT_TRUE_PROB_MAX="$student_true_prob_max" \
    SSML_AUG_CONSISTENCY_WEIGHT="$aug_weight" \
    SSML_AUG_CONSISTENCY_SHIFT="$aug_shift" \
    SSML_AUG_CONSISTENCY_FLIP_PROB="$aug_flip" \
    SSML_AUG_CONSISTENCY_NOISE_STD="$aug_noise"
done

echo "[classification_cifar100_overbatch_reproduce_fair_v2] done"
