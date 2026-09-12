# PoreAccess–CO2

Structure-resolved diffusion and adsorption in DEM-generated zeolite 13X beds,
followed by an interpretable accessibility-to-effective-diffusivity closure.
The present ensemble contains 20 beds at one bead size, tube diameter and
material parameter set. No flow, heat transfer or competitive adsorption is modeled.

## Start here

- [Review, verified findings and remaining modelling questions](docs/PNM_REVIEW.md)
- [Standalone publication figures](code_work/co2_publication_figures/README.md)
- [Detailed reproducibility record](code_work/REPRODUCIBILITY_CO2_PACKED_BED.md)
- [Presentation source](beamer/PoreAccess_CO2_presentation/poreaccess_co2_results.tex)
- [Paper source](paper/co2_packed_bed_adsorption_paper_draft.tex)

## Maintained workflow

Run commands from `code_work`. Python dependencies are NumPy, SciPy, pandas,
Matplotlib and scikit-learn. LAMMPS is required only to generate new packings.

1. `run_20_co2_packings.sh` generates packings; `summarize_co2_packing_campaign.py`
   checks them. Keep final particle and contact snapshots; intermediate dumps are ignored.
2. `analyze_co2_packing_porosity.py` checks porosity.
3. `extract_co2_pore_network.py --all-cases` now defaults to Sobol power 22 and
   `co2_pore_networks_power22`. Existing committed networks are unchanged.
4. `bash run_20_co2_adsorption_cases_finite_inlet.sh` runs the production campaign.
5. `summarize_co2_adsorption_campaign.py` and `extract_co2_inlet_accessibility.py`
   build the response and geometry tables. See the detailed guide for arguments.
6. `fit_co2_homogeneous_campaign.py` fits the 100-cell reduced model.
7. `analyze_co2_effective_diffusivity_closure.py` evaluates closure candidates.
8. `python3 generate_co2_publication_figures.py` creates 11 independent figures,
   each in PDF, SVG and 600 dpi PNG, without panel letters.

## Verification from committed inputs

```bash
cd code_work
python3 -m unittest discover -s tests -v
python3 audit_pore_networks.py
python3 evaluate_co2_held_out_curves.py
python3 generate_co2_publication_figures.py
```

The first command checks small analytical networks; the next two audit the
20 existing networks and evaluate complete held-out uptake curves. They do not
rerun DEM or refit the original network simulations.

Final `.data` and `.restart` packings are committed, but final `.dump` snapshots
were omitted in the initial upload. Restore those from the local project to
rerun the existing geometry and mechanical-QA scripts without regenerating DEM.
The `.gitignore` now permits them.

Historical fixed-inlet outputs, power-20 campaign networks, preliminary
homogeneous fits, an older nested analysis script, and LaTeX build products were
removed on the cleanup branch. Their originals remain in Git history. The
Sobol, diffusion and corrected grid-convergence validation cases are retained.
The audience and presenter-note decks contain 34 main slides. The main story uses
G_access directly to predict D_eff and complete uptake histories. Separate optional
appendices cover PNM construction (6 pages) and diffusivity normalization (5 pages).
See `beamer/PoreAccess_CO2_presentation/README.md` for the independent include switches.

The direct-conductance comparison is reproducible with:

```bash
cd code_work
python3 analyze_co2_conductance_closure.py
python3 plot_co2_conductance_closure.py
```

It compares three G_access relationships and a porosity baseline, using leave-one-bed-out
diffusivity predictions at four depths and 80 propagated 1D uptake histories at 5 bead
diameters. Full results are in `code_work/co2_conductance_closure/README.md`.
