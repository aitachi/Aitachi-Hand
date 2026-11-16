#!/usr/bin/env bash
# Negative-result attribution worker. Specs:
#   trpo500:<task>:<seed>   TRPO 500k budget+drift (dense checkpoints + ckpt eval)
#   rnd005:<task>:<seed>    RND lambda=0.05 ablation (150k, matrix protocol)
# Results land in hands-rl/results_attribution/ (never mixed into results/).
set -u
cd "$(dirname "$0")/../.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
OUT=hands-rl/results_attribution
mkdir -p "$OUT" hands-rl/logs

run_one() { # dir task algo seed steps ckpt evalck tag extra
  local dir="$OUT/$1" task=$2 algo=$3 seed=$4 steps=$5 ckpt=$6 evalck=$7 tag=$8; shift 8
  if [ -f "$dir/eval.json" ]; then echo "[skip] $dir"; return; fi
  if [ -f "$dir/claim" ]; then echo "[skip-claim] $dir"; return; fi
  mkdir -p "$dir"; echo $$ > "$dir/claim"
  echo "[run ] $(date +%H:%M:%S) $dir"
  PYTHONPATH=. "$PY" hands-rl/scripts/run_matrix.py --task "$task" --algo "$algo" \
    --seed "$seed" --timesteps "$steps" --out "$OUT" --eval-episodes 25 \
    ${ckpt:+--checkpoint-every "$ckpt"} ${evalck:+--eval-checkpoints "$evalck"} \
    --tag "$tag" "$@" \
    > "hands-rl/logs/attr_$tag.log" 2>&1
  echo "[done] $(date +%H:%M:%S) exit=$? $dir"
  rm -f "$dir/claim"
}

for spec in "$@"; do
  IFS=: read -r kind task seed <<< "$spec"
  case "$kind" in
    trpo500)
      run_one "${task}_trpo_s${seed}_trpo500" "$task" trpo "$seed" 500000 10000 \
              "10000,50000,150000,300000,500000" trpo500
      ;;
    rnd005)
      run_one "${task}_ppo_rnd_s${seed}_rnd005" "$task" ppo_rnd "$seed" 150000 0 "" \
              rnd005 --rnd-weight 0.05
      ;;
    trpolowstd)
      run_one "${task}_trpo_s${seed}_trpolowstd" "$task" trpo "$seed" 150000 10000 \
              "10000,50000,150000" trpolowstd --trpo-log-std -1.5
      ;;
    *) echo "[bad-spec] $spec" ;;
  esac
done
echo "ATTRIBUTION_WORKER_DONE $*"
