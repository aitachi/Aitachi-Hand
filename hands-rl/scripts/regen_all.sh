#!/usr/bin/env bash
# Regenerate all data-derived artifacts: summary, README, paper snippet, figures.
set -e
cd "$(dirname "$0")/../.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}
"$PY" scripts/aggregate.py
"$PY" scripts/gen_readme.py
"$PY" scripts/gen_results_snippet.py
"$PY" scripts/make_figures.py
echo "REGEN_ALL_DONE"
