#!/usr/bin/env bash
# Build the JSS (Elsevier) submission: the article, then its supplementary material.
# The supplement reads glaukopis_jss.aux for cross-references, so the article is built first.
# Needs cas-dc.cls, cas-common.sty and cas-model2-names.bst, all in this folder.
set -euo pipefail
cd -- "$(dirname -- "$0")"
command -v pdflatex >/dev/null || { echo 'pdflatex is required.' >&2; exit 1; }
for main in glaukopis_jss glaukopis_jss_supplementary; do
  pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
  bibtex "$main" >/dev/null || true
  pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" | tail -n 2
  grep -E "Warning: (Citation|Reference)|undefined" "$main.log" | grep -v 'Font shape' || true
  echo "Built: $PWD/$main.pdf"
done
