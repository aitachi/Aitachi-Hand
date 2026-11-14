#!/usr/bin/env bash
# One-line health: elapsed vs CPU for every running trainer.
ps -e --no-headers -o etime,time,cmd | grep "run_matrix.py" | grep -v grep | \
  sed 's|.*--task \([a-z]*\) --algo \([a-z0-9_]*\) --seed \([0-9]*\).*|\0 :: \1 \2 s\3|' | \
  awk '{n=split($0,a,"::"); split(a[1],f," "); print f[1], f[2], a[2]}'
