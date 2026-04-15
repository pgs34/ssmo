#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_local_cifar100_cifarstem_autofinish_v1.lock"
flock -n 9 || {
  echo "[launch_local_cifar100_cifarstem_autofinish_v1] launcher already running"
  exit 0
}

LOG_ROOT="${LOG_ROOT:-results/logs/classification_cifar100_cifarstem_followup_v1/autofinish}"
mkdir -p "$LOG_ROOT"
LOG_PATH="${LOG_PATH:-$LOG_ROOT/launcher.out}"

echo "[launch_local_cifar100_cifarstem_autofinish_v1] log=$LOG_PATH"
nohup python scripts/paper_rerun/auto_finalize_cifar100_cifarstem_followup_v1.py \
  > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
echo "[launch_local_cifar100_cifarstem_autofinish_v1] pid=$PID"
