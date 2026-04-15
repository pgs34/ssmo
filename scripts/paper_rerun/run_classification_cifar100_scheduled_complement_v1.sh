#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_scheduled_complement_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_scheduled_complement_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-default}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"
CASE_SPECS="${CASE_SPECS:-pcu_sched_df10_x05_r30_60:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:55:0.30:0.10:0.90:0.50:30:60 pcu_sched_df15_x08_r35_65:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:55:0.30:0.15:0.90:0.80:35:65 uh_sched_df10_x05_r30_60:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.80:2:0.50:0.03:0.72:0.92:0.03:4:18:55:0.30:0.10:0.90:0.50:30:60}"
DEFAULT_INIT_TEMPLATE='results/classification_neural_ode_cifar100_v11_backup/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
DEFAULT_PEER_TEMPLATE='results/classification_neural_ode_cifar100_v11_main/baseline/classification/{dataset}/{model}_independent_{classification_imitation_loss}_seed{seed}/model.pt'
BEST_CKPT_TEMPLATE="${BEST_CKPT_TEMPLATE:-$DEFAULT_INIT_TEMPLATE}"
PEER_BEST_CKPT_TEMPLATE="${PEER_BEST_CKPT_TEMPLATE:-$DEFAULT_PEER_TEMPLATE}"

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "classification_cifar100_scheduled_complement_v1/$label" \
    "classification_cifar100_scheduled_complement_v1/$label" \
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
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_BEST_CKPT_TEMPLATE" \
      SSML_TOPK_SCOPE="positive" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      SSML_SCORE_TRANSFORM="log1p" \
      SSML_GUIDANCE_MODE="hybrid" \
      SSML_PEER_CORRECT_ONLY="1" \
      SSML_STUDENT_INCORRECT_ONLY="1" \
      SSML_DISAGREEMENT_ONLY="1" \
      SSML_CLASS_BALANCED_TOPK="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_WORSE_ONLY_UPDATE="0" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_cifar100_scheduled_complement_v1] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_scheduled_complement_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[classification_cifar100_scheduled_complement_v1] protocol_id=$PROTOCOL_ID hardware_profile=$HARDWARE_PROFILE"
echo "[classification_cifar100_scheduled_complement_v1] best_ckpt_template=$BEST_CKPT_TEMPLATE"
echo "[classification_cifar100_scheduled_complement_v1] peer_best_ckpt_template=$PEER_BEST_CKPT_TEMPLATE"
echo "[classification_cifar100_scheduled_complement_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label gate_mode topk_ratio alpha lambda margin prob_threshold prob_gap per_class_budget distill_temperature anchor_weight student_true_prob_max aug_weight aug_shift aug_flip aug_noise peer_aug_min student_aug_max aug_gap_min warmup decay_start decay_end decay_min_scale disagreement_floor_ratio deficit_ema_momentum extra_class_budget_scale ramp_start ramp_end <<< "$spec"
  run_case \
    "$label" \
    SSML_GATE_SCORE_MODE="$gate_mode" \
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
    SSML_AUG_CONSISTENCY_NOISE_STD="$aug_noise" \
    SSML_PEER_AUG_CONSISTENCY_MIN="$peer_aug_min" \
    SSML_STUDENT_AUG_CONSISTENCY_MAX="$student_aug_max" \
    SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN="$aug_gap_min" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    SSML_DISAGREEMENT_FLOOR_RATIO="$disagreement_floor_ratio" \
    SSML_DEFICIT_EMA_MOMENTUM="$deficit_ema_momentum" \
    SSML_EXTRA_CLASS_BUDGET_SCALE="$extra_class_budget_scale" \
    SSML_COMPLEMENT_RAMP_START_EPOCH="$ramp_start" \
    SSML_COMPLEMENT_RAMP_END_EPOCH="$ramp_end"
done

echo "[classification_cifar100_scheduled_complement_v1] done"
