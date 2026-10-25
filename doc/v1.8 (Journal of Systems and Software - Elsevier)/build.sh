#!/usr/bin/env bash
# Build the JSS (Elsevier) submission: pdflatex -> bibtex -> pdflatex x2.
# Needs cas-dc.cls, cas-common.sty and cas-model2-names.bst, all in this folder.
set -euo pipefail
cd -- "$(dirname -- "$0")"
main=glaukopis_jss
command -v pdflatex >/dev/null || { echo 'pdflatex is required.' >&2; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
bibtex "$main" >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" | tail -n 3
grep -E "Warning: (Citation|Reference)|undefined" "$main.log" || true
echo "Built: $PWD/$main.pdf"
