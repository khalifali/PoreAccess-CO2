# Beamer presentation

Both versions contain 34 main slides with 6 optional backup slides disabled by default. The results section uses eleven standalone
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

## Presentation story

1. Motivation: why inlet-connected pathways may explain differences that porosity misses.
2. Pore-network reference: geometry, coupled pore and bead balances, equilibrium, and mean uptake.
3. Reduction: a separate 100-slice 1D grid, per-bed diffusivity optimization, and accessibility-based prediction.
4. Results and robustness.
5. Conclusions and next steps.

The audience deck is the visual baseline for both versions. The presenter version
uses matching slide bodies with a transcript explaining equations and transitions.

## Optional appendix

The six detailed PNM/Sobol backup slides are preserved in `pnm_backup_slides.tex`
in each deck directory. They are excluded from the default PDF and outline.
To include them, uncomment the final `\input{pnm_backup_slides.tex}` line in the
corresponding main source and compile twice. Keep the backup file with the main
source when copying either deck to another directory.
