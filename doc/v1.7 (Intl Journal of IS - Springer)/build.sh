#!/usr/bin/env bash
# Build the IJIS (Springer) submission: pdflatex -> bibtex -> pdflatex x2.
# Needs svjour3.cls, svglov3.clo and spmpsci.bst, all in this folder.
set -euo pipefail
cd -- "$(dirname -- "$0")"
main=glaukopis_ijis
command -v pdflatex >/dev/null || { echo 'pdflatex is required.' >&2; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
bibtex "$main" >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" | tail -n 3
grep -E "Warning: (Citation|Reference)" "$main.log" || true
echo "Built: $PWD/$main.pdf"
