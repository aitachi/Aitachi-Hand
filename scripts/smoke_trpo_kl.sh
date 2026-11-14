#!/usr/bin/env bash
# Smoke the two new algorithms on both tasks.
set -u
cd "$(dirname "$0")/.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}
for spec in "free trpo" "free ppo_kl" "seq trpo" "seq ppo_kl"; do
  set -- $spec
  echo "=== $1 $2 ==="
  PYTHONPATH=. timeout 480 "$PY" scripts/run_matrix.py --task "$1" --algo "$2" \
    --seed 0 --timesteps 2048 --out /tmp/ahand_smoke --eval-episodes 3 --smoke 2>/dev/null | tail -1
done
echo "SMOKE_TRPO_KL_DONE"
