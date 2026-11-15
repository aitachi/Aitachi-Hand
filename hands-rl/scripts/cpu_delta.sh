#!/usr/bin/env bash
# 10-second CPU-tick delta for every running trainer (progress liveness).
ps -e --no-headers -o pid=,cmd= | grep "run_matrix.py" | grep -v grep | \
while read -r PID _; do
  A=$(awk '{print $14+$15}' "/proc/$PID/stat" 2>/dev/null)
  T1=$(awk '{print $2}' "/proc/$PID/stat" 2>/dev/null)
  sleep 10
  B=$(awk '{print $14+$15}' "/proc/$PID/stat" 2>/dev/null)
  echo "pid=$PID delta=$(( (B - A) )) ticks/10s  ($T1)"
done
