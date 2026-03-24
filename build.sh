#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/learn"

for f in 0*.tex; do
    name="${f%.tex}"
    echo "Compiling $f ..."
    pdflatex -interaction=nonstopmode "$f" > /dev/null
    pdflatex -interaction=nonstopmode "$f" > /dev/null  # second pass for hyperref
    echo "  -> ${name}.pdf"
done

# Clean auxiliary files
rm -f *.aux *.log *.out *.toc

echo "Done. PDFs in learn/"
