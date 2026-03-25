#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Build basic tutorial (BabyGPT)
echo "Compiling babygpt-tutorial.tex ..."
cd "$ROOT/docs/basic"
pdflatex -interaction=nonstopmode babygpt-tutorial.tex > /dev/null
pdflatex -interaction=nonstopmode babygpt-tutorial.tex > /dev/null  # second pass for hyperref/TOC
rm -f *.aux *.log *.out *.toc
echo "  -> docs/basic/babygpt-tutorial.pdf"

# Build advanced tutorial (MicroGPT)
if [ -f "$ROOT/docs/advanced/microgpt-tutorial.tex" ]; then
  echo "Compiling microgpt-tutorial.tex ..."
  cd "$ROOT/docs/advanced"
  pdflatex -interaction=nonstopmode microgpt-tutorial.tex > /dev/null
  pdflatex -interaction=nonstopmode microgpt-tutorial.tex > /dev/null
  rm -f *.aux *.log *.out *.toc
  echo "  -> docs/advanced/microgpt-tutorial.pdf"
fi

echo "Done."
