#!/usr/bin/env bash
# One-shot finalization for the 66-run matrix (idempotent).
# Local:  bash scripts/finalize.sh          (PDF builds happen on Windows side)
# This script handles the WSL/Linux side: regen all data artifacts + site.
set -e
cd "$(dirname "$0")/../.."
PY=${AHAND_PY:-/mnt/c/Users/ASUS/Desktop/AHand/style1101/.venv/bin/python}

echo "== [1/4] regen data artifacts =="
"$PY" hands-rl/scripts/aggregate.py
"$PY" hands-rl/scripts/gen_readme.py
"$PY" hands-rl/scripts/gen_results_snippet.py
"$PY" hands-rl/scripts/make_figures.py

echo "== [2/4] gen site =="
"$PY" hands-rl/scripts/gen_site.py

echo "== [3/4] cleanup junk storm dirs =="
rm -rf results/*_ss[0-9]

echo "== [4/4] done (Windows side: rebuild textbook+paper PDFs, scp site, push) =="
echo "  cd textbook && bash build_pdf.sh"
echo "  cd paper && pdflatex x2 (env whitelist)"
echo "  python scripts/gen_site.py && scp -r site/* byte01:/var/www/Aitachi-Hand/"
echo "  push: local GITHUB_TOKEN=... python scripts/push_github.py"
echo "        OR byte01 relay: tar -> scp -> ssh byte01 'cd /root/ahand_push && GITHUB_TOKEN=... python3 scripts/push_github.py'"
echo "FINALIZE_SH_DONE"
