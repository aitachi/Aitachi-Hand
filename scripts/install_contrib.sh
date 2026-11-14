#!/usr/bin/env bash
set -e
PY=/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python
"$PY" -m pip install -q sb3-contrib==2.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
"$PY" -c "from sb3_contrib import TRPO; print('TRPO_OK')"
"$PY" -c "import stable_baselines3 as sb3; print('sb3', sb3.__version__)"
