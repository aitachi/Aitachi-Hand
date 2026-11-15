#!/usr/bin/env bash
# Dump process-tree states for a trainer (args: task algo seed).
TASK="$1"; ALGO="$2"; SEED="$3"
PID=$(ps -e --no-headers -o pid=,cmd= | grep -- "run_matrix.py --task $TASK --algo $ALGO --seed $SEED" | grep -v grep | head -1 | awk '{print $1}')
echo "main=$PID"
[ -z "$PID" ] && exit 0
grep -E "State|Threads" "/proc/$PID/status" 2>/dev/null
echo "-- children --"
ps --ppid "$PID" -o pid=,stat=,wchan:32=,cmd= 2>/dev/null | cut -c1-90 | head -10
