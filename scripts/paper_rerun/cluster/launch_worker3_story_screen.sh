#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-story_screen_v2}"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker3_story_cluster}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/story_screen" \
  "$LOG_DIR/worker3_story_screen.log" \
  env WORKER3_PHASES="${WORKER3_PHASES:-classification}" \
  bash scripts/paper_rerun/run_story_screen_worker3.sh

echo "[worker3_story_cluster] phases=${WORKER3_PHASES:-classification}"
echo "[worker3_story_cluster] results_root=$(paper_results_root)"
echo "[worker3_story_cluster] job finished"
