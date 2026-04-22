#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/namkyeong/anaconda3}"
CONDA_ENV="${CONDA_ENV:-ssml}"
FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR:-results}"

activate_paper_env() {
  if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
  export PYTHONUNBUFFERED=1
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
  else
    export PYTHONPATH="$ROOT_DIR"
  fi
}

paper_results_root() {
  printf '%s\n' "$FINAL_RESULTS_DIR"
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

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[%s] dry_run' "$label"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  (
    set -o pipefail
    "$@" 2>&1 \
      | tee "$logfile" \
      | awk -v label="$label" '
          /Traceback \(most recent call last\):/ { in_traceback=1 }
          {
            show = ($0 ~ /^\[/)
            show = show || ($0 ~ /epoch/)
            show = show || ($0 ~ /run_dir=/)
            show = show || ($0 ~ /done$/)
            show = show || ($0 ~ /Traceback \(most recent call last\):/)
            show = show || ($0 ~ /Error:/)
            show = show || ($0 ~ /Exception:/)
            show = show || in_traceback
            if (show) {
              print "[" label "] " $0
              fflush()
            }
            if ($0 ~ /Error:/ || $0 ~ /Exception:/) {
              in_traceback=0
            }
          }
        ' \
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

parallel_exec_init() {
  local max_jobs="${1:-all}"
  PARALLEL_EXEC_MAX="$max_jobs"
  if [[ "$PARALLEL_EXEC_MAX" == "all" || "$PARALLEL_EXEC_MAX" == "max" || "$PARALLEL_EXEC_MAX" == "0" ]]; then
    PARALLEL_EXEC_MAX=0
  fi
  PARALLEL_EXEC_FAILED=0
  PARALLEL_EXEC_PIDS=()
  PARALLEL_EXEC_LABELS=()
}

parallel_exec_limit_label() {
  if [[ -z "${PARALLEL_EXEC_MAX:-}" || "${PARALLEL_EXEC_MAX:-0}" == "0" ]]; then
    printf 'all'
  else
    printf '%s' "$PARALLEL_EXEC_MAX"
  fi
}

parallel_exec_cleanup() {
  local pid
  for pid in "${PARALLEL_EXEC_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}

parallel_exec_reap_one() {
  local finished_pid=""
  local finished_status=0
  local idx

  if (( ${#PARALLEL_EXEC_PIDS[@]} == 0 )); then
    return 0
  fi

  if wait -n -p finished_pid "${PARALLEL_EXEC_PIDS[@]}"; then
    finished_status=0
  else
    finished_status=$?
    PARALLEL_EXEC_FAILED=1
  fi

  for idx in "${!PARALLEL_EXEC_PIDS[@]}"; do
    if [[ "${PARALLEL_EXEC_PIDS[$idx]}" == "$finished_pid" ]]; then
      echo "[parallel] slot_free ${PARALLEL_EXEC_LABELS[$idx]} pid=$finished_pid status=$finished_status"
      unset 'PARALLEL_EXEC_PIDS[idx]'
      unset 'PARALLEL_EXEC_LABELS[idx]'
      PARALLEL_EXEC_PIDS=("${PARALLEL_EXEC_PIDS[@]}")
      PARALLEL_EXEC_LABELS=("${PARALLEL_EXEC_LABELS[@]}")
      break
    fi
  done
}

parallel_exec_submit() {
  local label="$1"
  shift
  local pid

  if [[ -n "${PARALLEL_EXEC_MAX:-}" ]] && (( PARALLEL_EXEC_MAX > 0 )); then
    while (( ${#PARALLEL_EXEC_PIDS[@]} >= PARALLEL_EXEC_MAX )); do
      parallel_exec_reap_one
    done
  fi

  "$@" &
  pid=$!
  PARALLEL_EXEC_PIDS+=("$pid")
  PARALLEL_EXEC_LABELS+=("$label")
  echo "[parallel] launch $label pid=$pid active=${#PARALLEL_EXEC_PIDS[@]}/$(parallel_exec_limit_label)"
}

parallel_exec_wait_all() {
  while (( ${#PARALLEL_EXEC_PIDS[@]} > 0 )); do
    parallel_exec_reap_one
  done
  return "$PARALLEL_EXEC_FAILED"
}

ensure_burgers_data() {
  local data_root="${1:-$ROOT_DIR/data}"
  local burgers_file="${2:-$data_root/burgers_data_R10.mat}"
  local burgers_zip_file="${3:-$data_root/burgers_data_R10.mat.zip}"
  local burgers_gdown_id="${BURGERS_GDOWN_ID:-}"
  local burgers_gdown_url="${BURGERS_GDOWN_URL:-}"

  mkdir -p "$data_root"
  if [[ -f "$burgers_file" ]]; then
    echo "[common] burgers data present: $burgers_file"
    return 0
  fi

  if [[ -f "$burgers_zip_file" ]]; then
    echo "[common] restoring burgers data from local archive: $burgers_zip_file"
    python3 - "$burgers_zip_file" <<'PY'
from pathlib import Path
import sys
import zipfile

zip_path = Path(sys.argv[1])
out_dir = zip_path.parent
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
PY
  fi

  if [[ -f "$burgers_file" ]]; then
    echo "[common] burgers data restored: $burgers_file"
    return 0
  fi

  if [[ -n "$burgers_gdown_id" ]]; then
    echo "[common] downloading burgers data via gdown id=$burgers_gdown_id"
    gdown --id "$burgers_gdown_id" -O "$burgers_file"
  elif [[ -n "$burgers_gdown_url" ]]; then
    echo "[common] downloading burgers data via gdown url"
    gdown "$burgers_gdown_url" -O "$burgers_file"
  fi

  if [[ ! -f "$burgers_file" ]]; then
    echo "[common] missing $burgers_file" >&2
    return 1
  fi
}
