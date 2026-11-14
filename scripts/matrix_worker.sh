#!/usr/bin/env bash
# Parallel matrix worker: takes "task:algo:seed" specs, claims each run via an
# atomic mkdir lock, skips finished runs. Usage: matrix_worker.sh <spec...>
set -u
cd "$(dirname "$0")/.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}
OUT=${MATRIX_OUT:-results}
mkdir -p "$OUT" logs
for spec in "$@"; do
  IFS=: read -r task algo seed <<< "$spec"
  dir="$OUT/${task}_${algo}_s${seed}"
  if [ -f "$dir/eval.json" ]; then echo "[skip-done] $spec"; continue; fi
  if ! mkdir "$dir" 2>/dev/null; then
    if [ -f "$dir/claim" ]; then echo "[skip-claim] $spec"; continue; fi
  fi
  echo "$$" > "$dir/claim" 2>/dev/null || true
  echo "[claim] $(date +%H:%M:%S) $spec"
  PYTHONPATH=. "$PY" scripts/run_matrix.py --task "$task" --algo "$algo" \
    --seed "$seed" --timesteps 150000 --out "$OUT" --eval-episodes 25 \
    > "logs/matrix_${task}_${algo}_s${seed}.log" 2>&1
  echo "[done ] $(date +%H:%M:%S) exit=$? $spec"
  rm -f "$dir/claim"
done
echo "WORKER_DONE $*"
