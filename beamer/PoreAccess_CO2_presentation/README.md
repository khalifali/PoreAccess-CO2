# Beamer presentation

Both audience and presenter versions contain 31 main slides and 15 appendix slides and use G_access as the
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
Slide 8 summarizes the PNM reference; slide 10 summarizes the 100-cell 1D model
and its fitting target. Both include schematic TikZ curves, not simulation data.
The PNM geometry appendix defines molecular gas diffusivity D_m = 1.5e-5 m2/s.
Slides 23–27 use the conductance-only analysis and figures. Slide 25 reports
mean held-out curve errors of 2.61% (power), 2.64% (linear), 4.80% (proportional)
and 7.58% (porosity). The earlier 1.58% is the per-bed fitted reduction error.

## Model appendices and optional normalization

Both decks include the model details after the 31-slide main story:

```latex
\input{pnm_backup_slides.tex}    % PNM equations and construction sections (13 pages)
\input{one_d_model_details.tex} % 1D section divider + 2 detailed slides
% \input{normalization_appendix.tex} % 5 optional comparison pages
```

A thank-you/questions slide closes the main story. Appendix A contains PNM
equations, Appendix B covers network construction, and Appendix C contains the
homogeneous 1D model; each section starts with a divider.

The existing PNM appendix now inputs `pnm_model_details.tex` before its original
construction pages. It retains the four former main-body PNM slides (8–11),
including the Sobol explanation in the presenter notes. `one_d_model_details.tex`
retains the former main-body 1D slides (13–14). All original detailed content and
presenter notes remain, with explicit spacing in the Langmuir products.

`normalization_appendix.tex` remains unchanged and optional. It explains
D_access = G_access H / A_tube, the full-height convention despite a conductance
ending at an interior plane, and why dimensional analysis alone does not derive
D_eff. Uncomment its input to include its five pages. A shared guard starts the
appendix only once. Each deck retains its own appendix sources for independent use.

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

The PNM equation appendix includes the prescribed LDF coefficient: literature
D_particle/R_p^2, the spherical factor-15 conversion, and the temperature assumption.
