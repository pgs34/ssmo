#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_strict128_followup_v1.lock"
flock -n 9 || {
  echo "[node0_cifar100_strict128_followup_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_strict128_followup_v1/node0}"
mkdir -p "$LOG_DIR"

GPU0_SEEDS="${GPU0_SEEDS:-0 2}"
GPU1_SEEDS="${GPU1_SEEDS:-1}"
FOLLOWUP_OUTPUT_ROOT="${FOLLOWUP_OUTPUT_ROOT:-results/classification_cifar100_strict128_followup_v1}"
POOL_ROOT="${POOL_ROOT:-results/classification_cifar100_bestckpt_pool_v3}"
PROBE_OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-$FOLLOWUP_OUTPUT_ROOT/probes}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
FOLLOWUP_PROTOCOL_ID="${FOLLOWUP_PROTOCOL_ID:-strict128_followup_v1}"
POOL_PROTOCOL_ID="${POOL_PROTOCOL_ID:-bestckpt_pool_v3}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-rtx4090}"
STRICT_INDEPENDENT_LABEL="${STRICT_INDEPENDENT_LABEL:-strict128_independent_v3}"
STRICT_DML_LABEL="${STRICT_DML_LABEL:-strict128_dml_v3}"
BASE_PREVIEW_CASE="${BASE_PREVIEW_CASE:-uh_sched_mem_v2}"
PROMOTION_MARGIN="${PROMOTION_MARGIN:-0.001}"
if [[ -z "${BEST_CKPT_TEMPLATE:-}" ]]; then
  BEST_CKPT_TEMPLATE="$POOL_ROOT/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
fi
if [[ -z "${PEER_BEST_CKPT_TEMPLATE:-}" ]]; then
  PEER_BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE"
fi
REPORT_PATH="${REPORT_PATH:-$LOG_DIR/narrow_exploit_report.json}"
PROMOTED_CASE_PATH="${PROMOTED_CASE_PATH:-$LOG_DIR/promoted_case.txt}"
PLOT_SCRIPT="${PLOT_SCRIPT:-scripts/paper_rerun/generate_top_level_best_plots.py}"

BASE_SSML_CASE_SPECS=(
  "uh_sched_mem_v2:useful_hard_sample_confident:0.24:0.12:0.04:0.012:0.000:0.33:0.38:0.00:0.02:4:6.0:0.0004:0.88:0.8:2:0.5:0.02:0.00:1.0:0.00:5:18:55:0.30:0.10:0.90:0.50:30:60:0.00"
  "uh_sched_mem_aug72:useful_hard_sample_confident:0.24:0.12:0.04:0.012:0.000:0.34:0.39:0.00:0.02:4:6.0:0.0004:0.88:0.9:2:0.5:0.02:0.72:0.90:0.03:5:20:65:0.35:0.10:0.90:0.50:30:65:0.00"
)
SEED2_ONLY_SSML_CASE_SPECS=(
  "pcu_sched_mem_aug72:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
  "uh_sched_mem_dense:useful_hard_sample_confident:0.28:0.14:0.05:0.015:0.000:0.33:0.40:0.00:0.02:5:6.0:0.0004:0.86:0.9:2:0.5:0.02:0.68:0.92:0.02:6:20:70:0.40:0.15:0.90:0.80:25:70:0.00"
)
PROBE_CASE_SPEC="${PROBE_CASE_SPEC:-pcu_sched_df10_x05_r30_60:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:55:0.30:0.10:0.90:0.50:30:60}"
PROBE_CASE_LABEL="${PROBE_CASE_SPEC%%:*}"

mkdir -p "$PROBE_OUTPUT_ROOT"

pool_best_model_path() {
  local seed="$1"
  printf '%s\n' "$POOL_ROOT/classification/classification/cifar100/resnet34_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/best_model.pt"
}

summary_path_for_followup_label() {
  local label="$1"
  local seed="$2"
  local run_dir="$FOLLOWUP_OUTPUT_ROOT/$label/classification/cifar100"

  if [[ "$label" == "$STRICT_INDEPENDENT_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  if [[ "$label" == "$STRICT_DML_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_gelu_dml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  printf '%s\n' "$run_dir/resnet34_gelu_ssml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
}

summary_path_for_probe_label() {
  local label="$1"
  local seed="$2"
  printf '%s\n' "$PROBE_OUTPUT_ROOT/$label/classification/cifar100/resnet34_gelu_ssml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
}

summary_path_for_case() {
  local label="$1"
  local seed="$2"
  if [[ "$label" == "$PROBE_CASE_LABEL" ]]; then
    summary_path_for_probe_label "$label" "$seed"
    return 0
  fi
  summary_path_for_followup_label "$label" "$seed"
}

maybe_run_pool_seed() {
  local gpu="$1"
  local seed="$2"
  local best_model
  best_model="$(pool_best_model_path "$seed")"
  if [[ -f "$best_model" ]]; then
    echo "[node0_cifar100_strict128_followup_v1] skip_completed pool seed=$seed best_model=$best_model"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_strict128_followup_v1_pool_gpu${gpu}_seed${seed}" \
    "$LOG_DIR/pool_gpu${gpu}_seed${seed}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="$seed" \
      EPOCHS="${POOL_EPOCHS:-100}" \
      BATCH_SIZE="${POOL_BATCH_SIZE_4090:-128}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      INDEPENDENT_MODELS="resnet34_gelu" \
      POOL_ROOT="$POOL_ROOT" \
      PROTOCOL_ID="$POOL_PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      LOG_DIR="$LOG_DIR/pool_gpu${gpu}_seed${seed}" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_bestckpt_pool_v2.sh
}

maybe_run_followup_group() {
  local gpu="$1"
  local seed="$2"
  local run_group="$3"
  local label="$4"
  local summary_path
  summary_path="$(summary_path_for_followup_label "$label" "$seed")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_strict128_followup_v1] skip_completed label=$label seed=$seed summary=$summary_path"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_strict128_followup_v1_${run_group}_gpu${gpu}_seed${seed}" \
    "$LOG_DIR/${run_group}_gpu${gpu}_seed${seed}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="$run_group" \
      SEEDS="$seed" \
      EPOCHS="${STRICT_EPOCHS:-100}" \
      BATCH_SIZE="${STRICT_BATCH_SIZE_4090:-128}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      LOG_DIR="$LOG_DIR/${run_group}_gpu${gpu}_seed${seed}" \
      OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
      PROTOCOL_ID="$FOLLOWUP_PROTOCOL_ID" \
      STRICT_INDEPENDENT_LABEL="$STRICT_INDEPENDENT_LABEL" \
      STRICT_DML_LABEL="$STRICT_DML_LABEL" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_strict128_followup_v1.sh
}

maybe_run_followup_ssml_case() {
  local gpu="$1"
  local seed="$2"
  local case_spec="$3"
  local label="${case_spec%%:*}"
  local summary_path
  summary_path="$(summary_path_for_followup_label "$label" "$seed")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_strict128_followup_v1] skip_completed label=$label seed=$seed summary=$summary_path"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_strict128_followup_v1_${label}_gpu${gpu}_seed${seed}" \
    "$LOG_DIR/${label}_gpu${gpu}_seed${seed}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="ssml" \
      SEEDS="$seed" \
      EPOCHS="${STRICT_EPOCHS:-100}" \
      BATCH_SIZE="${STRICT_BATCH_SIZE_4090:-128}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      SSML_CASE_SPECS="$case_spec" \
      LOG_DIR="$LOG_DIR/${label}_gpu${gpu}_seed${seed}" \
      OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
      PROTOCOL_ID="$FOLLOWUP_PROTOCOL_ID" \
      STRICT_INDEPENDENT_LABEL="$STRICT_INDEPENDENT_LABEL" \
      STRICT_DML_LABEL="$STRICT_DML_LABEL" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_strict128_followup_v1.sh
}

maybe_run_probe_case() {
  local gpu="$1"
  local seed="$2"
  local summary_path
  summary_path="$(summary_path_for_probe_label "$PROBE_CASE_LABEL" "$seed")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_strict128_followup_v1] skip_completed label=$PROBE_CASE_LABEL seed=$seed summary=$summary_path"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_strict128_followup_v1_${PROBE_CASE_LABEL}_gpu${gpu}_seed${seed}" \
    "$LOG_DIR/${PROBE_CASE_LABEL}_gpu${gpu}_seed${seed}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="$seed" \
      EPOCHS="${STRICT_EPOCHS:-100}" \
      BATCH_SIZE="${STRICT_BATCH_SIZE_4090:-128}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      OUTPUT_ROOT="$PROBE_OUTPUT_ROOT" \
      PROTOCOL_ID="$FOLLOWUP_PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_BEST_CKPT_TEMPLATE="$PEER_BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      CASE_SPECS="$PROBE_CASE_SPEC" \
      LOG_DIR="$LOG_DIR/${PROBE_CASE_LABEL}_gpu${gpu}_seed${seed}" \
      bash scripts/paper_rerun/run_classification_cifar100_scheduled_complement_v1.sh
}

evaluate_promoted_case() {
  REPORT_PATH="$REPORT_PATH" \
  FOLLOWUP_OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
  PROBE_OUTPUT_ROOT="$PROBE_OUTPUT_ROOT" \
  BASE_PREVIEW_CASE="$BASE_PREVIEW_CASE" \
  STRICT_DML_LABEL="$STRICT_DML_LABEL" \
  PROBE_CASE_LABEL="$PROBE_CASE_LABEL" \
  CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
  PROMOTION_MARGIN="$PROMOTION_MARGIN" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

followup_root = Path(os.environ["FOLLOWUP_OUTPUT_ROOT"])
probe_root = Path(os.environ["PROBE_OUTPUT_ROOT"])
base_case = os.environ["BASE_PREVIEW_CASE"]
strict_dml_label = os.environ["STRICT_DML_LABEL"]
probe_case_label = os.environ["PROBE_CASE_LABEL"]
classification_imitation_loss = os.environ["CLASSIFICATION_IMITATION_LOSS"]
promotion_margin = float(os.environ["PROMOTION_MARGIN"])

candidates = [
    base_case,
    "uh_sched_mem_aug72",
    "pcu_sched_mem_aug72",
    "uh_sched_mem_dense",
    probe_case_label,
]

def summary_path(label: str, seed: int) -> Path:
    if label == probe_case_label:
        root = probe_root / label
    else:
        root = followup_root / label
    if label == strict_dml_label:
        run_name = f"resnet34_gelu_dml_{classification_imitation_loss}_seed{seed}"
    elif label.startswith("strict128_independent"):
        run_name = f"resnet34_gelu_independent_{classification_imitation_loss}_seed{seed}"
    else:
        run_name = f"resnet34_gelu_ssml_{classification_imitation_loss}_seed{seed}"
    return root / "classification/cifar100" / run_name / "summary.json"

def load_best_val_acc(label: str, seed: int):
    path = summary_path(label, seed)
    if not path.exists():
        return None
    with path.open() as f:
        return float(json.load(f)["best_val_acc"])

base_score = load_best_val_acc(base_case, 2)
dml_score = load_best_val_acc(strict_dml_label, 2)
seed2_scores = {}
for label in candidates:
    score = load_best_val_acc(label, 2)
    if score is not None:
        seed2_scores[label] = score

promoted = None
if base_score is not None and dml_score is not None:
    eligible = [
        (label, score)
        for label, score in seed2_scores.items()
        if label != base_case and score > base_score and score >= dml_score + promotion_margin
    ]
    if eligible:
        promoted = max(eligible, key=lambda item: item[1])[0]

print(promoted or "")
PY
}

write_followup_report() {
  local promoted_case="${1:-}"
  REPORT_PATH="$REPORT_PATH" \
  FOLLOWUP_OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
  PROBE_OUTPUT_ROOT="$PROBE_OUTPUT_ROOT" \
  BASE_PREVIEW_CASE="$BASE_PREVIEW_CASE" \
  STRICT_INDEPENDENT_LABEL="$STRICT_INDEPENDENT_LABEL" \
  STRICT_DML_LABEL="$STRICT_DML_LABEL" \
  PROBE_CASE_LABEL="$PROBE_CASE_LABEL" \
  CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
  PROMOTION_MARGIN="$PROMOTION_MARGIN" \
  PROMOTED_CASE="$promoted_case" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["REPORT_PATH"])
followup_root = Path(os.environ["FOLLOWUP_OUTPUT_ROOT"])
probe_root = Path(os.environ["PROBE_OUTPUT_ROOT"])
base_case = os.environ["BASE_PREVIEW_CASE"]
strict_independent_label = os.environ["STRICT_INDEPENDENT_LABEL"]
strict_dml_label = os.environ["STRICT_DML_LABEL"]
probe_case_label = os.environ["PROBE_CASE_LABEL"]
classification_imitation_loss = os.environ["CLASSIFICATION_IMITATION_LOSS"]
promotion_margin = float(os.environ["PROMOTION_MARGIN"])
promoted_case = os.environ.get("PROMOTED_CASE", "").strip() or None

candidate_labels = [
    base_case,
    "uh_sched_mem_aug72",
    "pcu_sched_mem_aug72",
    "uh_sched_mem_dense",
    probe_case_label,
]

def summary_path(label: str, seed: int) -> Path:
    if label == probe_case_label:
        root = probe_root / label
    else:
        root = followup_root / label
    if label == strict_dml_label:
        run_name = f"resnet34_gelu_dml_{classification_imitation_loss}_seed{seed}"
    elif label == strict_independent_label:
        run_name = f"resnet34_gelu_independent_{classification_imitation_loss}_seed{seed}"
    else:
        run_name = f"resnet34_gelu_ssml_{classification_imitation_loss}_seed{seed}"
    return root / "classification/cifar100" / run_name / "summary.json"

def load_score(label: str, seed: int):
    path = summary_path(label, seed)
    if not path.exists():
        return None
    with path.open() as f:
        return float(json.load(f)["best_val_acc"])

def all_complete(label: str, seeds):
    return all(summary_path(label, seed).exists() for seed in seeds)

seed2_scores = {}
for label in candidate_labels:
    score = load_score(label, 2)
    if score is not None:
        seed2_scores[label] = score

preview_case = promoted_case or base_case
preview_mode = "aggressive_diagnostic"
if all_complete(strict_independent_label, (0, 1, 2)) and all_complete(strict_dml_label, (0, 1, 2)) and all_complete(preview_case, (0, 1, 2)):
    preview_mode = "corrected_followup_3seed"

report = {
    "track": "classification_cifar100_strict128_followup_v1",
    "base_case": base_case,
    "probe_case": probe_case_label,
    "candidate_labels": candidate_labels,
    "seed2_scores": seed2_scores,
    "controls_complete": {
        strict_independent_label: all_complete(strict_independent_label, (0, 1, 2)),
        strict_dml_label: all_complete(strict_dml_label, (0, 1, 2)),
    },
    "promotion_margin_over_dml": promotion_margin,
    "promoted_case": promoted_case,
    "preview_case": preview_case,
    "preview_mode": preview_mode,
    "cases_complete_3seed": {
        label: all_complete(label, (0, 1, 2))
        for label in candidate_labels
    },
}

report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open("w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

backfill_promoted_case() {
  local promoted_case="$1"
  local -a backfill_pairs=("0:${CLASSIFICATION_GPU0:-0}" "1:${CLASSIFICATION_GPU1:-1}")

  for pair in "${backfill_pairs[@]}"; do
    IFS=':' read -r seed gpu <<< "$pair"
    if [[ -f "$(summary_path_for_case "$promoted_case" "$seed")" ]]; then
      echo "[node0_cifar100_strict128_followup_v1] skip_completed promoted_case=$promoted_case seed=$seed"
      continue
    fi
    if [[ "$promoted_case" == "$PROBE_CASE_LABEL" ]]; then
      maybe_run_probe_case "$gpu" "$seed"
      continue
    fi
    local case_spec
    for case_spec in "${SEED2_ONLY_SSML_CASE_SPECS[@]}"; do
      if [[ "${case_spec%%:*}" == "$promoted_case" ]]; then
        maybe_run_followup_ssml_case "$gpu" "$seed" "$case_spec"
        break
      fi
    done
  done
}

refresh_followup_plots() {
  if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "[node0_cifar100_strict128_followup_v1] skip_plot_refresh missing_script=$PLOT_SCRIPT"
    return 0
  fi
  python3 "$PLOT_SCRIPT"
}

run_seed_sequence() {
  local gpu="$1"
  shift
  local seeds=("$@")
  local case_spec

  for seed in "${seeds[@]}"; do
    maybe_run_pool_seed "$gpu" "$seed"
    maybe_run_followup_group "$gpu" "$seed" "independent" "$STRICT_INDEPENDENT_LABEL"
    maybe_run_followup_group "$gpu" "$seed" "dml" "$STRICT_DML_LABEL"

    for case_spec in "${BASE_SSML_CASE_SPECS[@]}"; do
      maybe_run_followup_ssml_case "$gpu" "$seed" "$case_spec"
    done

    if [[ "$seed" == "2" ]]; then
      for case_spec in "${SEED2_ONLY_SSML_CASE_SPECS[@]}"; do
        maybe_run_followup_ssml_case "$gpu" "$seed" "$case_spec"
      done
      maybe_run_probe_case "$gpu" "$seed"
    fi
  done
}

read -r -a GPU0_SEED_ARRAY <<< "$GPU0_SEEDS"
read -r -a GPU1_SEED_ARRAY <<< "$GPU1_SEEDS"

run_seed_sequence "${CLASSIFICATION_GPU0:-0}" "${GPU0_SEED_ARRAY[@]}" &
PID0=$!
run_seed_sequence "${CLASSIFICATION_GPU1:-1}" "${GPU1_SEED_ARRAY[@]}" &
PID1=$!

echo "[node0_cifar100_strict128_followup_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0 seeds=$GPU0_SEEDS"
echo "[node0_cifar100_strict128_followup_v1] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1 seeds=$GPU1_SEEDS"

wait "$PID0"
wait "$PID1"

PROMOTED_CASE="$(evaluate_promoted_case)"
printf '%s\n' "$PROMOTED_CASE" > "$PROMOTED_CASE_PATH"

if [[ -n "$PROMOTED_CASE" ]]; then
  echo "[node0_cifar100_strict128_followup_v1] promoted_case=$PROMOTED_CASE backfill_seeds=0,1"
  backfill_promoted_case "$PROMOTED_CASE"
else
  echo "[node0_cifar100_strict128_followup_v1] no_promotion seed2_non_base_probe_did_not_clear_threshold"
fi

write_followup_report "$PROMOTED_CASE"
refresh_followup_plots

echo "[node0_cifar100_strict128_followup_v1] all jobs finished"
