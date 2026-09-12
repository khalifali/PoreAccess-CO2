# Beamer presentation

Both audience and presenter versions contain 34 main slides and use G_access as the
sole inlet-access descriptor. D_eff remains the fitted/predicted coefficient used
by the 1D transient model. Speaker notes follow the same slide order and describe
both coefficient prediction and held-out uptake-history validation.

Compile twice from each deck directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results.tex
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results.tex
cd with_notes
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results_with_notes.tex
pdflatex -interaction=nonstopmode -halt-on-error poreaccess_co2_results_with_notes.tex
```

The presenter PDF places the slide on the left and its transcript on the right.
Set `\presenternotesfalse` instead of `\presenternotestrue` to hide notes.
Each version retains its own figures folder and appendix sources for independent use.

## Main story

1. Inlet-connected pathways may explain variability missed by porosity.
2. The PNM supplies reference mean uptake histories.
3. Per-bed curve fitting gives D_eff; steady diffusion gives G_access.
4. Compare proportional, linear-with-intercept and power-law G_access relationships.
5. Validate both held-out diffusivity and the resulting 1D uptake history.
6. Assess interior-plane sensitivity and state the limits of the current ensemble.

Slide 7 motivates 15% inlet CO2 with a cited DOE coal-flue-gas concentration range.
The dry 298.15 K condition is a simplified cooled reference, not hot industrial flue gas.
Slide 8 defines molecular gas diffusivity D_m = 1.5e-5 m2/s.
Slides 27–31 use the new conductance-only analysis and figures. Slide 29 reports
mean held-out curve errors of 2.61% (power), 2.64% (linear), 4.80% (proportional)
and 7.58% (porosity). The earlier 1.58% is the per-bed fitted reduction error.

## Two independent optional appendices

Both are excluded by default. Uncomment either or both inputs at the end of the
main source and compile twice:

```latex
\input{pnm_backup_slides.tex}       % 6 detailed PNM/Sobol pages
\input{normalization_appendix.tex} % 5 normalization/comparison pages
```

`normalization_appendix.tex` explains D_access = G_access H / A_tube, the full-height
convention despite a conductance ending at an interior plane, and why dimensional
analysis alone does not derive D_eff. It preserves the normalized power-law and
depth comparisons. It is separate from the original PNM construction appendix.
A shared guard starts the appendix once when both files are enabled.

## Reproducing the new analysis

From `code_work`:

```bash
python3 analyze_co2_conductance_closure.py
python3 plot_co2_conductance_closure.py
```

The outputs in `code_work/co2_conductance_closure` contain all model scores,
80 held-out uptake histories, per-bed curve errors and five PDF/SVG figures.
Copy its figure PDFs into both deck figure directories before rebuilding.
The older `generate_co2_publication_figures.py` outputs remain valid for the
unchanged results and optional normalized-descriptor comparisons.
