#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_strict128_followup_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_strict128_followup_v1}"
PROTOCOL_ID="${PROTOCOL_ID:-strict128_followup_v1}"
STRICT_INDEPENDENT_LABEL="${STRICT_INDEPENDENT_LABEL:-strict128_independent_v3}"
STRICT_DML_LABEL="${STRICT_DML_LABEL:-strict128_dml_v3}"

DEFAULT_POOL_TEMPLATE="results/classification_cifar100_bestckpt_pool_v3/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
BEST_CKPT_TEMPLATE="${BEST_CKPT_TEMPLATE:-$DEFAULT_POOL_TEMPLATE}"

# Focused around the currently previewed combo:
# - rerun the exact uh_sched_mem structure with corrected pool/lr wiring
# - add augfilter-aware memory variants that borrow the stronger gating filters
SSML_CASE_SPECS="${SSML_CASE_SPECS:-uh_sched_mem_v2:useful_hard_sample_confident:0.24:0.12:0.04:0.012:0.000:0.33:0.38:0.00:0.02:4:6.0:0.0004:0.88:0.8:2:0.5:0.02:0.00:1.0:0.00:5:18:55:0.30:0.10:0.90:0.50:30:60:0.00 uh_sched_mem_aug72:useful_hard_sample_confident:0.24:0.12:0.04:0.012:0.000:0.34:0.39:0.00:0.02:4:6.0:0.0004:0.88:0.9:2:0.5:0.02:0.72:0.90:0.03:5:20:65:0.35:0.10:0.90:0.50:30:65:0.00 pcu_sched_mem_aug72:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 uh_sched_mem_dense:useful_hard_sample_confident:0.28:0.14:0.05:0.015:0.000:0.33:0.40:0.00:0.02:5:6.0:0.0004:0.86:0.9:2:0.5:0.02:0.68:0.92:0.02:6:20:70:0.40:0.15:0.90:0.80:25:70:0.00}"

echo "[classification_cifar100_strict128_followup_v1] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_strict128_followup_v1] best_ckpt_template=$BEST_CKPT_TEMPLATE"
echo "[classification_cifar100_strict128_followup_v1] strict_independent_label=$STRICT_INDEPENDENT_LABEL"
echo "[classification_cifar100_strict128_followup_v1] strict_dml_label=$STRICT_DML_LABEL"

exec env \
  LOG_DIR="$LOG_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  PROTOCOL_ID="$PROTOCOL_ID" \
  STRICT_INDEPENDENT_LABEL="$STRICT_INDEPENDENT_LABEL" \
  STRICT_DML_LABEL="$STRICT_DML_LABEL" \
  BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
  SSML_CASE_SPECS="$SSML_CASE_SPECS" \
  CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
  bash scripts/paper_rerun/run_classification_cifar100_strict128_aggressive_v1.sh
