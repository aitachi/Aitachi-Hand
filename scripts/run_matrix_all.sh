#!/usr/bin/env bash
# Full comparison matrix: 2 tasks x 9 algorithms x 3 seeds @150k steps.
# Resumable: a run is skipped when its eval.json already exists.
set -u
cd "$(dirname "$0")/.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}
OUT=${MATRIX_OUT:-results}
mkdir -p "$OUT" logs
ALGOS="ppo td3 sac ppo_rb ppo_espo ppo_ppg ppo_exo ppo_pbrs ppo_rnd"
for task in free seq; do
  for algo in $ALGOS; do
    for seed in 1 2 3; do
      dir="$OUT/${task}_${algo}_s${seed}"
      if [ -f "$dir/eval.json" ]; then
        echo "[skip] $dir"
        continue
      fi
      echo "[run ] $(date +%H:%M:%S) $task $algo seed=$seed"
      PYTHONPATH=. "$PY" scripts/run_matrix.py --task "$task" --algo "$algo" \
        --seed "$seed" --timesteps 150000 --out "$OUT" --eval-episodes 25 \
        > "logs/matrix_${task}_${algo}_s${seed}.log" 2>&1
      echo "[done] $(date +%H:%M:%S) exit=$? $dir"
    done
  done
done
echo "MATRIX_ALL_DONE $(date)"
