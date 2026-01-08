@echo off
REM Build the LaTeX paper to PDF
REM Requires pdflatex (from MiKTeX or TeX Live)

echo Building paper.tex...
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo.
echo Build complete! Output: paper.pdf
pause
