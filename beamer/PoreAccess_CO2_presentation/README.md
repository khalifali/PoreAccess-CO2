# Beamer presentation

Both versions contain 32 slides. The results section uses eleven standalone
figures, each on its own slide, with editable explanatory text. Speaker notes
follow the same slide order and refer to the individual figures.

From this directory, compile the audience deck twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results.tex
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results.tex
```

For the presenter version:

```bash
cd with_notes
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results_with_notes.tex
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results_with_notes.tex
```

The presenter PDF places each slide on the left and its notes on the right.
Set `\presenternotesfalse` instead of `\presenternotestrue` to hide notes.
Each deck keeps its figure PDFs in its own `figures` folder so it can be copied
and compiled independently. Regenerate the canonical plots using
`code_work/generate_co2_publication_figures.py`, then copy the standalone PDFs
into both figure folders when updating results.
