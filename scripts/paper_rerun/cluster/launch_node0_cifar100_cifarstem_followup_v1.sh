#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_cifarstem_followup_v1.lock"
flock -n 9 || {
  echo "[node0_cifar100_cifarstem_followup_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_cifarstem_followup_v1/node0}"
mkdir -p "$LOG_DIR"

FOLLOWUP_OUTPUT_ROOT="${FOLLOWUP_OUTPUT_ROOT:-results/classification_cifar100_cifarstem_followup_v1}"
POOL_ROOT="${POOL_ROOT:-results/classification_cifar100_bestckpt_pool_cifarstem_v1}"
POOL_REUSE_ROOT="${POOL_REUSE_ROOT:-results/classification_cifar100_bestckpt_pool_v2/classification/classification/cifar100}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
FOLLOWUP_PROTOCOL_ID="${FOLLOWUP_PROTOCOL_ID:-cifarstem_followup_v1}"
POOL_PROTOCOL_ID="${POOL_PROTOCOL_ID:-bestckpt_pool_cifarstem_v1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-rtx4090}"
CIFARSTEM_INDEPENDENT_LABEL="${CIFARSTEM_INDEPENDENT_LABEL:-cifarstem_independent_v1}"
CIFARSTEM_DML_LABEL="${CIFARSTEM_DML_LABEL:-cifarstem_dml_v1}"
FOLLOWUP_SUMMARY_MD="${FOLLOWUP_SUMMARY_MD:-Results_Summary.md}"
REPORT_PATH="${REPORT_PATH:-$LOG_DIR/cifarstem_followup_report.json}"
PROMOTED_CASES_PATH="${PROMOTED_CASES_PATH:-$LOG_DIR/promoted_cases.json}"
PLOT_REFRESH_SCRIPT="${PLOT_REFRESH_SCRIPT:-scripts/paper_rerun/refresh_top_level_best_plots.sh}"
GPU0_PROBE_CASE_SPECS="${GPU0_PROBE_CASE_SPECS:-pcu_cifarstem_sched_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 pcu_cifarstem_dense_v1:peer_confident_student_uncertain:0.28:0.12:0.05:0.012:0.000:0.35:0.40:0.01:0.03:6:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.15:0.90:1.00:25:70:0.00 oxtra42_cifarstem_v1:useful_hard_sample_confident:0.42:0.42:0.020:0.018:0.000:0.42:0.42:0.01:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:0.00:1.00:0.00:12:18:36:0.25:0.00:0.00:0.00:-1:-1:0.00}"
GPU1_PROBE_CASE_SPECS="${GPU1_PROBE_CASE_SPECS:-pcu_cifarstem_sched_l10_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.010:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 pcu_cifarstem_sched_l09_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.009:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 pcu_cifarstem_sched_l08_t7_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.008:0.000:0.35:0.40:0.01:0.03:5:7.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 oxtra35_cifarstem_relax_v1:useful_hard_sample_confident:0.35:0.35:0.020:0.015:0.000:0.40:0.40:0.01:0.01:12:6.0:0.0002:0.90:1.10:3:0.50:0.04:0.00:1.00:0.00:8:18:45:0.25:0.00:0.00:0.00:-1:-1:0.00}"
ALL_PROBE_CASE_SPECS="${ALL_PROBE_CASE_SPECS:-$GPU0_PROBE_CASE_SPECS $GPU1_PROBE_CASE_SPECS}"
if [[ -z "${BEST_CKPT_TEMPLATE:-}" ]]; then
  BEST_CKPT_TEMPLATE="$POOL_ROOT/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
fi

pool_seed_dir() {
  local seed="$1"
  printf '%s\n' "$POOL_ROOT/classification/classification/cifar100/resnet34_cifar_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}"
}

pool_best_model_path() {
  local seed="$1"
  printf '%s\n' "$(pool_seed_dir "$seed")/best_model.pt"
}

pool_reuse_seed_dir() {
  local seed="$1"
  printf '%s\n' "$POOL_REUSE_ROOT/resnet34_cifar_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}"
}

summary_path_for_label() {
  local label="$1"
  local seed="$2"
  local run_dir="$FOLLOWUP_OUTPUT_ROOT/$label/classification/cifar100"

  if [[ "$label" == "$CIFARSTEM_INDEPENDENT_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  if [[ "$label" == "$CIFARSTEM_DML_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_dml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  printf '%s\n' "$run_dir/resnet34_cifar_gelu_ssml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
}

curve_path_for_label() {
  local label="$1"
  local seed="$2"
  local run_dir="$FOLLOWUP_OUTPUT_ROOT/$label/classification/cifar100"

  if [[ "$label" == "$CIFARSTEM_INDEPENDENT_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/curves.npz"
    return 0
  fi
  if [[ "$label" == "$CIFARSTEM_DML_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_dml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/curves.npz"
    return 0
  fi
  printf '%s\n' "$run_dir/resnet34_cifar_gelu_ssml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/curves.npz"
}

ensure_reused_pool_seed() {
  local seed="$1"
  local dst_dir
  local src_dir
  dst_dir="$(pool_seed_dir "$seed")"
  src_dir="$(pool_reuse_seed_dir "$seed")"
  if [[ -f "$dst_dir/best_model.pt" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1] reuse_pool_skip seed=$seed best_model=$dst_dir/best_model.pt"
    return 0
  fi
  if [[ ! -d "$src_dir" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1] reuse_pool_missing seed=$seed src_dir=$src_dir" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dst_dir")"
  cp -a "$src_dir" "$dst_dir"
  echo "[node0_cifar100_cifarstem_followup_v1] reused_pool seed=$seed src=$src_dir dst=$dst_dir"
}

maybe_run_pool_seed0() {
  local gpu="$1"
  local best_model
  best_model="$(pool_best_model_path 0)"
  if [[ -f "$best_model" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1] skip_completed pool seed=0 best_model=$best_model"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_cifarstem_followup_v1_pool_gpu${gpu}_seed0" \
    "$LOG_DIR/pool_gpu${gpu}_seed0.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="0" \
      EPOCHS="${POOL_EPOCHS:-100}" \
      BATCH_SIZE="${POOL_BATCH_SIZE_4090:-1536}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      POOL_ROOT="$POOL_ROOT" \
      PROTOCOL_ID="$POOL_PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      LOG_DIR="$LOG_DIR/pool_gpu${gpu}_seed0" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_bestckpt_pool_cifarstem_v1.sh
}

maybe_run_followup_group() {
  local gpu="$1"
  local seed="$2"
  local run_group="$3"
  local label="$4"
  local summary_path
  summary_path="$(summary_path_for_label "$label" "$seed")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1] skip_completed label=$label seed=$seed summary=$summary_path"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_cifarstem_followup_v1_${run_group}_gpu${gpu}_seed${seed}" \
    "$LOG_DIR/${run_group}_gpu${gpu}_seed${seed}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="$run_group" \
      SEEDS="$seed" \
      EPOCHS="${FOLLOWUP_EPOCHS:-100}" \
      BATCH_SIZE="${FOLLOWUP_BATCH_SIZE_4090:-1536}" \
      INDEPENDENT_BATCH_SIZE="${FOLLOWUP_INDEPENDENT_BATCH_SIZE_4090:-1536}" \
      DML_BATCH_SIZE="${FOLLOWUP_DUAL_BATCH_SIZE_4090:-768}" \
      SSML_BATCH_SIZE="${FOLLOWUP_DUAL_BATCH_SIZE_4090:-768}" \
      DML_LR="${FOLLOWUP_DUAL_LR_4090:-0.05}" \
      SSML_LR="${FOLLOWUP_DUAL_LR_4090:-0.05}" \
      DML_SCHEDULER_WARMUP_EPOCHS="${FOLLOWUP_DUAL_SCHEDULER_WARMUP_EPOCHS_4090:-5}" \
      SSML_SCHEDULER_WARMUP_EPOCHS="${FOLLOWUP_DUAL_SCHEDULER_WARMUP_EPOCHS_4090:-5}" \
      DML_SCHEDULER_MIN_SCALE="${FOLLOWUP_DUAL_SCHEDULER_MIN_SCALE_4090:-0.10}" \
      SSML_SCHEDULER_MIN_SCALE="${FOLLOWUP_DUAL_SCHEDULER_MIN_SCALE_4090:-0.10}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      LOG_DIR="$LOG_DIR/${run_group}_gpu${gpu}_seed${seed}" \
      OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
      PROTOCOL_ID="$FOLLOWUP_PROTOCOL_ID" \
      CIFARSTEM_INDEPENDENT_LABEL="$CIFARSTEM_INDEPENDENT_LABEL" \
      CIFARSTEM_DML_LABEL="$CIFARSTEM_DML_LABEL" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_cifarstem_followup_v1.sh
}

maybe_run_followup_ssml_case() {
  local gpu="$1"
  local seed="$2"
  local case_spec="$3"
  local label="${case_spec%%:*}"
  local summary_path
  summary_path="$(summary_path_for_label "$label" "$seed")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1] skip_completed label=$label seed=$seed summary=$summary_path"
    return 0
  fi

  run_logged_job \
    "node0/cifar100_cifarstem_followup_v1_${label}_gpu${gpu}_seed${seed}" \
    "$LOG_DIR/${label}_gpu${gpu}_seed${seed}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="ssml" \
      SEEDS="$seed" \
      EPOCHS="${FOLLOWUP_EPOCHS:-100}" \
      BATCH_SIZE="${FOLLOWUP_BATCH_SIZE_4090:-1536}" \
      INDEPENDENT_BATCH_SIZE="${FOLLOWUP_INDEPENDENT_BATCH_SIZE_4090:-1536}" \
      DML_BATCH_SIZE="${FOLLOWUP_DUAL_BATCH_SIZE_4090:-768}" \
      SSML_BATCH_SIZE="${FOLLOWUP_DUAL_BATCH_SIZE_4090:-768}" \
      DML_LR="${FOLLOWUP_DUAL_LR_4090:-0.05}" \
      SSML_LR="${FOLLOWUP_DUAL_LR_4090:-0.05}" \
      DML_SCHEDULER_WARMUP_EPOCHS="${FOLLOWUP_DUAL_SCHEDULER_WARMUP_EPOCHS_4090:-5}" \
      SSML_SCHEDULER_WARMUP_EPOCHS="${FOLLOWUP_DUAL_SCHEDULER_WARMUP_EPOCHS_4090:-5}" \
      DML_SCHEDULER_MIN_SCALE="${FOLLOWUP_DUAL_SCHEDULER_MIN_SCALE_4090:-0.10}" \
      SSML_SCHEDULER_MIN_SCALE="${FOLLOWUP_DUAL_SCHEDULER_MIN_SCALE_4090:-0.10}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      SSML_CASE_SPECS="$case_spec" \
      LOG_DIR="$LOG_DIR/${label}_gpu${gpu}_seed${seed}" \
      OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
      PROTOCOL_ID="$FOLLOWUP_PROTOCOL_ID" \
      CIFARSTEM_INDEPENDENT_LABEL="$CIFARSTEM_INDEPENDENT_LABEL" \
      CIFARSTEM_DML_LABEL="$CIFARSTEM_DML_LABEL" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_cifarstem_followup_v1.sh
}

evaluate_promoted_cases() {
  REPORT_PATH="$REPORT_PATH" \
  FOLLOWUP_OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
  CIFARSTEM_INDEPENDENT_LABEL="$CIFARSTEM_INDEPENDENT_LABEL" \
  CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
  ALL_PROBE_CASE_SPECS="$ALL_PROBE_CASE_SPECS" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

followup_root = Path(os.environ["FOLLOWUP_OUTPUT_ROOT"])
independent_label = os.environ["CIFARSTEM_INDEPENDENT_LABEL"]
loss = os.environ["CLASSIFICATION_IMITATION_LOSS"]
case_specs = [spec for spec in os.environ["ALL_PROBE_CASE_SPECS"].split() if spec]

family_map = {}
labels = []
for spec in case_specs:
    label = spec.split(":", 1)[0]
    labels.append(label)
    family_map[label] = "pcu" if label.startswith("pcu_") else "oxtra"

def summary_path(label: str, seed: int) -> Path:
    run_dir = followup_root / label / "classification/cifar100"
    if label == independent_label:
        run_name = f"resnet34_cifar_gelu_independent_{loss}_seed{seed}"
    else:
        run_name = f"resnet34_cifar_gelu_ssml_{loss}_seed{seed}"
    return run_dir / run_name / "summary.json"

def load_score(label: str, seed: int):
    path = summary_path(label, seed)
    if not path.exists():
        return None
    with path.open() as f:
        return float(json.load(f)["best_val_acc"])

independent_seed2 = load_score(independent_label, 2)
eligible = []
if independent_seed2 is not None:
    for label in labels:
        score = load_score(label, 2)
        if score is not None and score > independent_seed2:
            eligible.append((label, score, family_map[label]))

eligible.sort(key=lambda item: item[1], reverse=True)
promoted = []
if eligible:
    first_label, _, first_family = eligible[0]
    promoted.append(first_label)
    for label, _, family in eligible[1:]:
        if family != first_family:
            promoted.append(label)
            break

print(json.dumps(promoted))
PY
}

write_followup_report() {
  local promoted_cases_json="${1:-[]}"
  REPORT_PATH="$REPORT_PATH" \
  FOLLOWUP_OUTPUT_ROOT="$FOLLOWUP_OUTPUT_ROOT" \
  POOL_ROOT="$POOL_ROOT" \
  CIFARSTEM_INDEPENDENT_LABEL="$CIFARSTEM_INDEPENDENT_LABEL" \
  CIFARSTEM_DML_LABEL="$CIFARSTEM_DML_LABEL" \
  CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
  ALL_PROBE_CASE_SPECS="$ALL_PROBE_CASE_SPECS" \
  PROMOTED_CASES_JSON="$promoted_cases_json" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["REPORT_PATH"])
followup_root = Path(os.environ["FOLLOWUP_OUTPUT_ROOT"])
pool_root = Path(os.environ["POOL_ROOT"])
independent_label = os.environ["CIFARSTEM_INDEPENDENT_LABEL"]
dml_label = os.environ["CIFARSTEM_DML_LABEL"]
loss = os.environ["CLASSIFICATION_IMITATION_LOSS"]
case_specs = [spec for spec in os.environ["ALL_PROBE_CASE_SPECS"].split() if spec]
promoted_cases = json.loads(os.environ["PROMOTED_CASES_JSON"])

candidate_labels = [spec.split(":", 1)[0] for spec in case_specs]
family_map = {label: ("pcu" if label.startswith("pcu_") else "oxtra") for label in candidate_labels}

def summary_path(label: str, seed: int) -> Path:
    run_dir = followup_root / label / "classification/cifar100"
    if label == independent_label:
        run_name = f"resnet34_cifar_gelu_independent_{loss}_seed{seed}"
    elif label == dml_label:
        run_name = f"resnet34_cifar_gelu_dml_{loss}_seed{seed}"
    else:
        run_name = f"resnet34_cifar_gelu_ssml_{loss}_seed{seed}"
    return run_dir / run_name / "summary.json"

def curve_path(label: str, seed: int) -> Path:
    run_dir = followup_root / label / "classification/cifar100"
    if label == independent_label:
        run_name = f"resnet34_cifar_gelu_independent_{loss}_seed{seed}"
    elif label == dml_label:
        run_name = f"resnet34_cifar_gelu_dml_{loss}_seed{seed}"
    else:
        run_name = f"resnet34_cifar_gelu_ssml_{loss}_seed{seed}"
    return run_dir / run_name / "curves.npz"

def load_score(label: str, seed: int):
    path = summary_path(label, seed)
    if not path.exists():
        return None
    with path.open() as f:
        return float(json.load(f)["best_val_acc"])

def available_seeds(label: str):
    return [seed for seed in (0, 1, 2) if summary_path(label, seed).exists()]

def mean_score(label: str, seeds):
    scores = [load_score(label, seed) for seed in seeds]
    if any(score is None for score in scores):
        return None
    return float(sum(scores) / len(scores))

seed2_scores = {}
for label in [independent_label, dml_label, *candidate_labels]:
    score = load_score(label, 2)
    if score is not None:
        seed2_scores[label] = score

three_seed_cases = [
    label for label in candidate_labels
    if available_seeds(label) == [0, 1, 2]
]
preview_case = None
preview_seeds = []
preview_mode = "pending"

controls_3seed = available_seeds(independent_label) == [0, 1, 2] and available_seeds(dml_label) == [0, 1, 2]
if controls_3seed and three_seed_cases:
    preview_case = max(three_seed_cases, key=lambda label: mean_score(label, (0, 1, 2)) or float("-inf"))
    preview_seeds = [0, 1, 2]
    preview_mode = "matched_3seed"
elif independent_label in seed2_scores and dml_label in seed2_scores:
    seed2_cases = [label for label in candidate_labels if label in seed2_scores]
    if seed2_cases:
        preview_case = max(seed2_cases, key=lambda label: seed2_scores[label])
        preview_seeds = [2]
        preview_mode = "seed2_probe_only"

latest_row = None
if preview_case is not None and preview_seeds:
    indep_score = mean_score(independent_label, preview_seeds)
    dml_score = mean_score(dml_label, preview_seeds)
    ssml_score = mean_score(preview_case, preview_seeds)
    if indep_score is not None and dml_score is not None and ssml_score is not None:
        latest_row = {
            "track": "CIFAR-100 cifarstem_followup_v1",
            "backbone": "resnet34_cifar_gelu x resnet34_cifar_gelu",
            "protocol": "matched 3-seed" if preview_mode == "matched_3seed" else "seed2 probe only",
            "independent": indep_score,
            "dml": dml_score,
            "ssml": ssml_score,
            "ssml_case": preview_case,
            "ssml_family": family_map.get(preview_case, "unknown"),
            "verdict": (
                "SSML > independent and DML"
                if ssml_score > indep_score and ssml_score > dml_score
                else "SSML > independent only"
                if ssml_score > indep_score
                else "SSML <= independent or DML"
            ),
        }

pool_complete = {
    seed: (pool_root / "classification/classification/cifar100" / f"resnet34_cifar_gelu_independent_{loss}_seed{seed}" / "best_model.pt").exists()
    for seed in (0, 1, 2)
}

report = {
    "track": "classification_cifar100_cifarstem_followup_v1",
    "pool_root": str(pool_root),
    "candidate_labels": candidate_labels,
    "candidate_families": family_map,
    "promoted_cases": promoted_cases,
    "pool_complete": pool_complete,
    "seed2_scores": seed2_scores,
    "preview_case": preview_case,
    "preview_mode": preview_mode,
    "preview_seeds": preview_seeds,
    "latest_matched_row": latest_row,
    "controls_complete_3seed": {
        independent_label: available_seeds(independent_label) == [0, 1, 2],
        dml_label: available_seeds(dml_label) == [0, 1, 2],
    },
    "cases_complete_3seed": {
        label: available_seeds(label) == [0, 1, 2]
        for label in candidate_labels
    },
    "available_seeds": {
        label: available_seeds(label)
        for label in [independent_label, dml_label, *candidate_labels]
    },
    "curve_paths": {
        label: {str(seed): str(curve_path(label, seed)) for seed in available_seeds(label)}
        for label in [independent_label, dml_label, *candidate_labels]
    },
}

report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open("w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

upsert_results_summary_appendix() {
  REPORT_PATH="$REPORT_PATH" \
  SUMMARY_PATH="$FOLLOWUP_SUMMARY_MD" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_PATH"])
report_path = Path(os.environ["REPORT_PATH"])
start_marker = "<!-- CIFAR100_CIFARSTEM_FOLLOWUP_V1_START -->"
end_marker = "<!-- CIFAR100_CIFARSTEM_FOLLOWUP_V1_END -->"

report = {}
if report_path.exists():
    with report_path.open() as f:
        report = json.load(f)

latest_row = report.get("latest_matched_row")
preview_mode = report.get("preview_mode", "pending")
preview_case = report.get("preview_case") or "pending"
promoted_cases = report.get("promoted_cases") or []

if latest_row:
    row = (
        f"| {latest_row['track']} | `{latest_row['backbone']}` | `{latest_row['protocol']}` | "
        f"`{latest_row['independent']:.6f}` | `{latest_row['dml']:.6f}` | `{latest_row['ssml']:.6f}` | "
        f"`{latest_row['verdict']}` | preview SSML case = `{latest_row['ssml_case']}` ({latest_row['ssml_family']}); "
        f"current preview mode = `{preview_mode}` |"
    )
else:
    row = (
        "| CIFAR-100 cifarstem_followup_v1 | `resnet34_cifar_gelu x resnet34_cifar_gelu` | `pending launch / awaiting matched controls` | "
        "pending | pending | pending | pending | seed0 pool bootstrap + seed2 control/probe sweep is the next gate |"
    )

promoted_line = ", ".join(f"`{label}`" for label in promoted_cases) if promoted_cases else "`none yet`"
section = f"""
{start_marker}

## CIFAR-100 cifarstem_followup_v1 Appendix

### Why backbone pivot

SSML 자체가 전반적으로 망가진 것은 아니다. 다른 domain과 일부 classification setting에서는 이미 개선 신호와 승리가 있었고, 현재 CIFAR-100 clean homogeneous는 `resnet34_gelu` strict track의 backbone/stem 병목이 더 크게 보인다. 그래서 이번 pivot은 방법론 포기가 아니라, capacity와 inductive bias를 바꿔 같은 SSML logic를 다시 검증하는 CIFAR-100 병목 분리 실험이다.

### Latest matched result

| Track | Backbone | Protocol | Independent | DML | SSML | Current verdict | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
{row}

Promoted backfill targets: {promoted_line}

![CIFAR-100 only validation error](./test_error_cifar100_only.png)
{end_marker}
"""

existing = summary_path.read_text()
if start_marker in existing and end_marker in existing:
    prefix = existing.split(start_marker, 1)[0].rstrip()
    suffix = existing.split(end_marker, 1)[1].lstrip()
    updated = prefix + "\n\n" + section.strip() + ("\n\n" + suffix if suffix else "\n")
else:
    updated = existing.rstrip() + "\n\n" + section.strip() + "\n"
summary_path.write_text(updated)
PY
}

refresh_followup_plots() {
  if [[ ! -f "$PLOT_REFRESH_SCRIPT" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1] skip_plot_refresh missing_script=$PLOT_REFRESH_SCRIPT"
    return 0
  fi
  bash "$PLOT_REFRESH_SCRIPT"
}

run_seed2_queue() {
  local gpu="$1"
  local run_independent="$2"
  local run_dml="$3"
  shift 3
  local case_specs=("$@")
  local case_spec

  if [[ "$run_independent" == "1" ]]; then
    maybe_run_followup_group "$gpu" "2" "independent" "$CIFARSTEM_INDEPENDENT_LABEL"
  fi
  if [[ "$run_dml" == "1" ]]; then
    maybe_run_followup_group "$gpu" "2" "dml" "$CIFARSTEM_DML_LABEL"
  fi

  for case_spec in "${case_specs[@]}"; do
    maybe_run_followup_ssml_case "$gpu" "2" "$case_spec"
  done
}

backfill_controls_and_promoted() {
  local promoted_cases_json="$1"
  python3 - <<'PY' "$promoted_cases_json"
import json
import sys
for label in json.loads(sys.argv[1]):
    print(label)
PY
}

run_backfill_queue() {
  local gpu="$1"
  local seed="$2"
  shift 2
  local promoted_labels=("$@")
  local label
  local case_spec

  maybe_run_followup_group "$gpu" "$seed" "independent" "$CIFARSTEM_INDEPENDENT_LABEL"
  maybe_run_followup_group "$gpu" "$seed" "dml" "$CIFARSTEM_DML_LABEL"

  for label in "${promoted_labels[@]}"; do
    for case_spec in $ALL_PROBE_CASE_SPECS; do
      if [[ "${case_spec%%:*}" == "$label" ]]; then
        maybe_run_followup_ssml_case "$gpu" "$seed" "$case_spec"
        break
      fi
    done
  done
}

ensure_reused_pool_seed 1
ensure_reused_pool_seed 2
write_followup_report "[]"
upsert_results_summary_appendix
refresh_followup_plots

read -r -a GPU0_CASE_ARRAY <<< "$GPU0_PROBE_CASE_SPECS"
read -r -a GPU1_CASE_ARRAY <<< "$GPU1_PROBE_CASE_SPECS"

maybe_run_pool_seed0 "${CLASSIFICATION_GPU0:-0}" &
POOL_PID=$!
run_seed2_queue "${CLASSIFICATION_GPU1:-1}" "0" "1" "${GPU1_CASE_ARRAY[@]}" &
PID1=$!

echo "[node0_cifar100_cifarstem_followup_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$POOL_PID pool_seed0"
echo "[node0_cifar100_cifarstem_followup_v1] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1 seed2_queue"

wait "$POOL_PID"
run_seed2_queue "${CLASSIFICATION_GPU0:-0}" "1" "0" "${GPU0_CASE_ARRAY[@]}" &
PID0=$!
echo "[node0_cifar100_cifarstem_followup_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0 seed2_queue"

wait "$PID0"
wait "$PID1"

PROMOTED_CASES_JSON="$(evaluate_promoted_cases)"
printf '%s\n' "$PROMOTED_CASES_JSON" > "$PROMOTED_CASES_PATH"

if [[ "$PROMOTED_CASES_JSON" != "[]" ]]; then
  mapfile -t PROMOTED_LABELS < <(backfill_controls_and_promoted "$PROMOTED_CASES_JSON")
  echo "[node0_cifar100_cifarstem_followup_v1] promoted_cases=${PROMOTED_LABELS[*]}"
  run_backfill_queue "${CLASSIFICATION_GPU0:-0}" "0" "${PROMOTED_LABELS[@]}" &
  BACKFILL_PID0=$!
  run_backfill_queue "${CLASSIFICATION_GPU1:-1}" "1" "${PROMOTED_LABELS[@]}" &
  BACKFILL_PID1=$!
  wait "$BACKFILL_PID0"
  wait "$BACKFILL_PID1"
else
  echo "[node0_cifar100_cifarstem_followup_v1] no_promotion seed2_probe_did_not_clear_independent_gate"
fi

write_followup_report "$PROMOTED_CASES_JSON"
upsert_results_summary_appendix
refresh_followup_plots

echo "[node0_cifar100_cifarstem_followup_v1] all jobs finished"
