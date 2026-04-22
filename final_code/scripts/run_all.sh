#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for experiment in \
  weather \
  electricity \
  etth1 \
  burgers \
  darcy \
  cifar10 \
  cifar100_cifarstem
do
  echo "[run_all] start $experiment"
  bash "$SCRIPT_DIR/run_experiment.sh" "$experiment"
  echo "[run_all] done $experiment"
done
