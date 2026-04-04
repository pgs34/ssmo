#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/namkyeong/anaconda3}"
CONDA_ENV="${CONDA_ENV:-ssml}"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-paper_rerun_canonical}"

activate_paper_env() {
  if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
  export PYTHONUNBUFFERED=1
}

paper_results_root() {
  printf '%s\n' "results/$PAPER_RERUN_TAG"
}

array_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

collect_unique_models() {
  local pairs="$1"
  local pair model peer
  local -a models=()
  for pair in $pairs; do
    IFS=':' read -r model peer <<< "$pair"
    for candidate in "$model" "$peer"; do
      if [[ -z "${candidate:-}" ]]; then
        continue
      fi
      if ! array_contains "$candidate" "${models[@]}"; then
        models+=("$candidate")
      fi
    done
  done
  printf '%s\n' "${models[@]}"
}

pair_is_distinct() {
  local model="$1"
  local peer="$2"
  [[ -n "${peer:-}" && "$model" != "$peer" ]]
}

run_logged_job() {
  local label="$1"
  local logfile="$2"
  shift 2

  mkdir -p "$(dirname "$logfile")"
  echo "[$label] log -> $logfile"

  (
    set -o pipefail
    "$@" 2>&1 \
      | tee "$logfile" \
      | awk -v label="$label" '{
          if ($0 ~ /^\[/ || $0 ~ /epoch/ || $0 ~ /run_dir=/ || $0 ~ /done$/) {
            print "[" label "] " $0
            fflush()
          }
        }' \
  )
}

sanitize_lock_name() {
  printf '%s' "$1" | tr '/ :=' '____' | tr -cd '[:alnum:]_.-'
}

run_locked_job() {
  local lock_key="$1"
  local label="$2"
  local logfile="$3"
  shift 3

  local lock_root="$ROOT_DIR/results/.locks"
  local lock_name
  local lock_file

  mkdir -p "$lock_root"
  lock_name="$(sanitize_lock_name "$lock_key")"
  lock_file="$lock_root/${lock_name}.lock"

  (
    exec 9>"$lock_file"
    if ! flock -n 9; then
      echo "[$label] skip_locked lock=$lock_file"
      exit 0
    fi
    echo "[$label] lock -> $lock_file"
    run_logged_job "$label" "$logfile" "$@"
  )
}
