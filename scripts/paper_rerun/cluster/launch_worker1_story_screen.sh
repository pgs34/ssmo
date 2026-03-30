#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-story_screen_v2}"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker1_story_cluster}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker1/story_screen" \
  "$LOG_DIR/worker1_story_screen.log" \
  bash scripts/paper_rerun/run_story_screen_worker1.sh

echo "[worker1_story_cluster] results_root=$(paper_results_root)"
echo "[worker1_story_cluster] job finished"
