#!/usr/bin/env bash
# Kill a hung/misbehaving trainer by task/algo/seed signature; clean its claim.
# Usage: kill_run.sh <task> <algo> <seed>
set -u
TASK="$1"; ALGO="$2"; SEED="$3"
PID=$(ps -e --no-headers -o pid,cmd | grep "run_matrix.py --task $TASK --algo $ALGO --seed $SEED" | grep -v grep | awk '{print $1}')
if [ -n "$PID" ]; then
  kill -9 "$PID" 2>/dev/null && echo "killed $PID ($TASK $ALGO s$SEED)"
  sleep 1
  pkill -9 -P "$PID" 2>/dev/null
else
  echo "no process found for $TASK $ALGO s$SEED"
fi
rm -f "/mnt/c/Users/ASUS/Desktop/Aitachi-Hand/results/${TASK}_${ALGO}_s${SEED}/claim"
echo cleaned
