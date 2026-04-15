#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_cifarstem_followup_v1/node0}"
mkdir -p "$LOG_DIR"

POOL_PATTERN="${POOL_PATTERN:-run_classification_cifar100_bestckpt_pool_cifarstem_v1.sh}"
GPU1_QUEUE_PATTERN="${GPU1_QUEUE_PATTERN:-launch_node0_cifar100_cifarstem_followup_v1_gpu1_seed2.sh}"
POLL_SECONDS="${POLL_SECONDS:-30}"

while pgrep -f "$POOL_PATTERN" >/dev/null || pgrep -f "$GPU1_QUEUE_PATTERN" >/dev/null; do
  sleep "$POLL_SECONDS"
done

env \
  FOLLOWUP_DUAL_LR_4090="${FOLLOWUP_DUAL_LR_4090:-0.025}" \
  FOLLOWUP_DUAL_SCHEDULER_WARMUP_EPOCHS_4090="${FOLLOWUP_DUAL_SCHEDULER_WARMUP_EPOCHS_4090:-8}" \
  FOLLOWUP_DUAL_SCHEDULER_MIN_SCALE_4090="${FOLLOWUP_DUAL_SCHEDULER_MIN_SCALE_4090:-0.20}" \
  bash scripts/paper_rerun/cluster/launch_node0_cifar100_cifarstem_followup_v1.sh
