#!/usr/bin/env bash
# Build the Aitachi-Hand textbook PDF (MiKTeX xelatex).
# env whitelist is REQUIRED: MiKTeX helper subprocesses crash without
# SystemRoot/TEMP (symptom: "FATAL ... Windows error" from dvipdfmx / fmt build).
set -e
cd "$(dirname "$0")"
export PATH="/c/Program Files/MiKTeX/miktex/bin/x64:$PATH"
run() { env -i HOME="$HOME" PATH="$PATH" SystemRoot="$SystemRoot" TEMP="$TEMP" \
  TMP="$TMP" USERPROFILE="$USERPROFILE" ComSpec="$ComSpec" PATHEXT="$PATHEXT" \
  xelatex -interaction=nonstopmode -halt-on-error main.tex; }
run > build1.log 2>&1 || { tail -30 build1.log; exit 1; }
run > build2.log 2>&1 || { tail -30 build2.log; exit 1; }
echo "errors: $(grep -c '^!' main.log || true)"
ls -la main.pdf
echo "PDF_OK $(date +%H:%M:%S)"
