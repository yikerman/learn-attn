#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/learn"

echo "Compiling babygpt-tutorial.tex ..."
pdflatex -interaction=nonstopmode babygpt-tutorial.tex > /dev/null
pdflatex -interaction=nonstopmode babygpt-tutorial.tex > /dev/null  # second pass for hyperref/TOC
echo "  -> babygpt-tutorial.pdf"

# Clean auxiliary files
rm -f *.aux *.log *.out *.toc

echo "Done. PDF in learn/"
