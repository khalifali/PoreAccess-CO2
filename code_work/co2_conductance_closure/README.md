# Conductance-only transport closure

The main presentation uses the calculated steady inlet-access conductance G_access
to predict the effective diffusivity fitted to each complete PNM uptake history.
The optional normalized descriptor is retained in the older
`co2_effective_diffusivity_closure` analysis and the separate normalization appendix.

Run from `code_work`:

```bash
python3 analyze_co2_conductance_closure.py
python3 plot_co2_conductance_closure.py
```

The three conductance forms are proportional, linear with intercept, and power law.
The porosity-only linear model is a baseline independent of interior-plane depth.
Every coefficient is refitted inside each leave-one-bed-out fold at each depth.
Power-law fitting uses log coordinates; comparison scores use physical diffusivity units.
Positive predicted diffusivities are required; nonpositive predictions raise an error
and are never clipped. All current held-out predictions are positive.

At the 5-diameter plane, each predicted diffusivity is used in the original 100-cell
1D model with the stored per-bed material, geometric, boundary and solver settings.
The initial gas concentration and solid loading are zero, matching the original fits.
The omitted PNM curve is used only for evaluation, not coefficient fitting.
There are 80 propagated histories (20 beds times 4 models).
Curve NRMSE uses trapezoidal time weights and final PNM uptake as its denominator.

| Model | Held-out diffusivity R2 | Mean curve NRMSE |
|---|---:|---:|
| G_access power law | 0.906 | 2.61% |
| G_access linear + intercept | 0.898 | 2.64% |
| G_access proportional | 0.558 | 4.80% |
| Porosity linear | -0.234 | 7.58% |

Power-law and linear predictions are close; the evidence does not establish a decisive
winner or universal transfer across geometries. The power law has a positive zero-access
limit, whereas the fitted linear model becomes nonphysical if extrapolated far below
the observed conductance range. Model and depth choices were compared within the same
20-bed ensemble: this is internal validation, not nested selection or external validation.

`held_out_curves.csv.gz` contains all reference and predicted histories; `curve_metrics.csv`
contains per-bed errors and mass residuals. The five PDF/SVG figures are reproducible
from these committed outputs. Original simulations and per-bed fits are unchanged.

Verification: the 5-diameter raw-linear R2 matches the previous stored comparison.
The same solver-parameter reconstruction reproduces the original seed-18427 fitted
final loading to relative tolerance 1e-8. Maximum absolute mass residual across the
80 new solves is below 1.6e-13 mol.
