#!/usr/bin/env bash
set -euo pipefail

FILTER="${1:-}"

if [[ -n "$FILTER" ]]; then
  echo "[python_status] filter=$FILTER"
else
  echo "[python_status] filter=<none>"
fi

printf '%-8s %-8s %-10s %-6s %-6s %-6s %-12s %s\n' \
  "PID" "PPID" "ELAPSED" "%CPU" "%MEM" "STATE" "CUDA" "CMD"

found=0
while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  read -r pid ppid etime pcpu pmem state cmd <<<"$line"
  [[ "$cmd" == *python* ]] || continue
  [[ "$cmd" == *"check_python_pids.sh"* ]] && continue
  if [[ -n "$FILTER" && "$cmd" != *"$FILTER"* ]]; then
    continue
  fi

  found=1
  cuda_visible_devices="<unset>"
  if [[ -r "/proc/$pid/environ" ]]; then
    cuda_env="$(
      tr '\0' '\n' <"/proc/$pid/environ" \
        | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" { print $2; exit }'
    )"
    if [[ -n "${cuda_env:-}" ]]; then
      cuda_visible_devices="$cuda_env"
    fi
  fi

  cwd="?"
  if [[ -e "/proc/$pid/cwd" ]]; then
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || printf '?')"
  fi

  printf '%-8s %-8s %-10s %-6s %-6s %-6s %-12s %s\n' \
    "$pid" "$ppid" "$etime" "$pcpu" "$pmem" "$state" "$cuda_visible_devices" "$cmd"
  echo "  cwd=$cwd"
done < <(ps -eo pid=,ppid=,etime=,pcpu=,pmem=,state=,args=)

if [[ "$found" -eq 0 ]]; then
  echo "[python_status] no matching python processes"
fi
