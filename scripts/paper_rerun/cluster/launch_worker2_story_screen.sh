#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-story_screen_v2}"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker2_story_cluster}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker2/story_screen" \
  "$LOG_DIR/worker2_story_screen.log" \
  bash scripts/paper_rerun/run_story_screen_worker2.sh

echo "[worker2_story_cluster] results_root=$(paper_results_root)"
echo "[worker2_story_cluster] job finished"
