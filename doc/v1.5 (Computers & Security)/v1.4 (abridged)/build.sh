#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
command -v pdflatex >/dev/null || { echo 'pdflatex is required.' >&2; exit 1; }
main=glaukopis_abridged_v1.4
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex"
if command -v bibtex >/dev/null; then
  bibtex "$main"
elif command -v bibtex8 >/dev/null; then
  bibtex8 "$main"
else
  test -s "$main.bbl" || { echo 'No BibTeX executable or prebuilt .bbl.' >&2; exit 1; }
fi
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex"
echo "Built: $PWD/$main.pdf"
