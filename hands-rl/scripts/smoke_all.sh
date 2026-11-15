#!/usr/bin/env bash
# Smoke-test every algorithm path on both tasks (tiny budgets).
set -u
cd "$(dirname "$0")/../.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}
OUT=${SMOKE_OUT:-/tmp/ahand_smoke}
for spec in "seq ppo" "free ppo_rb" "free ppo_exo" "free ppo_espo" "free ppo_ppg" \
            "free ppo_pbrs" "free ppo_rnd" "free td3" "free sac" "seq ppo_ppg" \
            "seq ppo_rnd" "seq td3" "seq sac"; do
  task=${spec%% *}
  algo=${spec##* }
  echo "=== ${task} ${algo} ==="
  PYTHONPATH=. timeout 420 "$PY" scripts/run_matrix.py --task "$task" --algo "$algo" \
    --seed 0 --timesteps 2048 --out "$OUT" --eval-episodes 3 --smoke 2>/dev/null | tail -1
done
echo "SMOKE_ALL_DONE"
