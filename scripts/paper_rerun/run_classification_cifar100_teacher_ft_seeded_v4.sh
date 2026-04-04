#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_teacher_ft_seeded_v4}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_teacher_ft_seeded_v4}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
CASE_SPECS="${CASE_SPECS:-pcu_late_pb16_thr40_gap24_aug78_92_ag03:peer_confident_student_uncertain:0.16:0.05:0.010:0.024:0.40:0.024:4:7.0:0.0008:0.84:0.50:2:0.02:0.78:0.92:0.03:10:35:100:0.30 pcu_late_pb14_thr42_gap26_aug80_92_ag04:peer_confident_student_uncertain:0.14:0.05:0.010:0.026:0.42:0.026:4:7.0:0.0008:0.82:0.50:2:0.02:0.80:0.92:0.04:12:40:105:0.25 uh_late_pb18_thr36_gap20_aug76_94_ag02:useful_hard_sample_confident:0.18:0.04:0.012:0.020:0.36:0.020:4:6.0:0.0006:0.86:0.45:2:0.03:0.76:0.94:0.02:8:30:95:0.30 uh_late_pb16_thr38_gap22_aug78_94_ag03:useful_hard_sample_confident:0.16:0.04:0.011:0.022:0.38:0.022:4:6.0:0.0006:0.84:0.45:2:0.03:0.78:0.94:0.03:10:35:100:0.25}"
DEFAULT_BEST_CKPT_TEMPLATE="results/classification_cifar100_bestckpt_pool_v1/classification/classification/cifar100/"'{model}'"_independent_${CLASSIFICATION_IMITATION_LOSS}_seed"'{'seed'}'"/best_model.pt"
BEST_CKPT_TEMPLATE="${BEST_CKPT_TEMPLATE:-$DEFAULT_BEST_CKPT_TEMPLATE}"

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "classification_cifar100_teacher_ft_seeded_v4/$label" \
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
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      SSML_TOPK_SCOPE="positive" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
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

echo "[classification_cifar100_teacher_ft_seeded_v4] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_teacher_ft_seeded_v4] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[classification_cifar100_teacher_ft_seeded_v4] best_ckpt_template=$BEST_CKPT_TEMPLATE"
echo "[classification_cifar100_teacher_ft_seeded_v4] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label gate_score_mode topk_ratio alpha lambda margin prob_threshold prob_gap per_class_budget distill_temperature anchor_weight student_true_prob_max aug_consistency_weight aug_shift aug_flip peer_aug_min student_aug_max aug_gap_min warmup decay_start decay_end decay_min_scale <<< "$spec"
  run_case \
    "$label" \
    SSML_GATE_SCORE_MODE="$gate_score_mode" \
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
    SSML_AUG_CONSISTENCY_WEIGHT="$aug_consistency_weight" \
    SSML_AUG_CONSISTENCY_SHIFT="$aug_shift" \
    SSML_AUG_CONSISTENCY_FLIP_PROB="$aug_flip" \
    SSML_PEER_AUG_CONSISTENCY_MIN="$peer_aug_min" \
    SSML_STUDENT_AUG_CONSISTENCY_MAX="$student_aug_max" \
    SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN="$aug_gap_min" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale"
done

echo "[classification_cifar100_teacher_ft_seeded_v4] done"
