# Pore-network review — 11 September 2026

Reviewed baseline: `4637a408713b931014950dcfc7a4feb455b44ea6`.
This is a code and numerical audit, not proof of physical validity or absence
of all errors. Original production networks, adsorption trajectories and fits
were not regenerated or overwritten.

## Verified

Eight analytical tests check known tetrahedron/triangle circumcentres, graph
conservation, normalized incidence, series/parallel conductance with isolated
components, Toth derivatives, exponential fixed-inlet diffusion, and finite/fixed
inlet adsorption. Adsorption Jacobians agree with central finite differences in
both uptake and desorption branches. Tiny-system inventory checks pass.

All 20 stored networks passed the existing QA, volume and incidence checks.
The largest relative graph row-sum error is 2.64e-16 or less. Audit CSV:
`../code_work/pnm_review/network_audit.csv`.

The fixed previously selected power-law closure gives LOOCV diffusivity R2
0.8900. Propagating each stored held-out D prediction through the conservative
100-cell 1D model gives mean time-weighted uptake NRMSE **2.70%**, range
**1.26–5.92%**, normalized by each network's loading at 5000 s. The largest
absolute final-loading relative error is **10.52%**. Maximum absolute inventory
residual across these new 1D runs is **1.50e-13 mol**. Results and per-bed curves:
`../code_work/pnm_review/held_out_curves/`.

These are held-out coefficient predictions within the same 20-bed ensemble,
not independent external validation and not nested selection of depth/model.
The original 1.58% mean curve error measures per-bed fitted reduction, a different
quantity from the 2.70% mean predictive error above.

## Fixes

- Reject all-failed QA datasets in figure and closure readers instead of silently
  falling back to all failed cases.
- Extraction defaults now agree with production: power 22 and the power22 folder.
- Closed/nonpositive face clearances are skipped rather than opened by a radius floor.
  The current stored networks have positive clearances and were not regenerated.
- Validate finite positive transport geometry, initial states and output counts.
- Check concentration/loading lower bounds at all saved output times, not only
  at the final time. This does not certify values between output times.
- Finite-inlet launcher checks metadata/settings, final state and completed time
  series before skipping; conflicting successful settings stop the launcher.
- Accessibility records the depth reference explicitly; optional `--bed-bottom 0`
  tests a physical-bottom definition. Omitting it preserves the original results.
- Document the actual concentration-weighted positive uptake sink. The old module
  equation only described the fixed-weight desorption branch.

## Geometry assumptions that remain open

1. **Boundary representation.** Pores are surviving tetrahedral circumcentres.
   Circumcentres outside the cylinder are discarded. Only original convex-hull
   faces label inlet/top/wall pores; interfaces against discarded tetrahedra do
   not create a wall-conforming boundary network. Volume conservation alone
   cannot validate this topology. Test reflected/boundary-conforming constructions
   or a resolved diffusion benchmark before treating the network as ground truth.
2. **Throat area.** `pi*(face_circumradius - bead_radius)^2` is a circular-clearance
   surrogate, not the exact open triangular area or an integrated pore/throat
   resistance. Across 361,425 throats, 17.47–19.41% per bed have a face circumcentre
   outside the connecting pore-centre segment. This is possible for Delaunay duals
   and is not by itself a coding error. Sampling 17 positions on every segment
   found no solid intersections and no clearances below the assigned radius
   (2 micrometre tolerance). This supports centreline validity but does not validate
   diffusive conductance. Compare another throat-resistance model.
3. **Volume allocation.** Sobol void samples go to the nearest retained pore centre;
   their fractions are renormalized to an analytically estimated total void volume.
   These are surrogate control volumes, not clipped Delaunay void volumes. The
   half-count stabilizer and overlap-neglecting solid-volume sum need to remain
   documented. Sobol convergence is necessary but does not establish geometric fidelity.
4. **Depth convention.** The current 5dp plane is measured above the lowest pore,
   whose z varies from 0.000000377 to 0.000205170 m across beds. It is not exactly
   5dp above the physical bottom. Reports should state this or rerun a bottom-based
   sensitivity. Do not relabel existing curves as bottom-based results.
5. **Boundary reduction.** PNM inlet conductances use individual pore heights and
   equal allocations of total tube area. The 1D boundary uses one transfer length
   of 0.5dp. These have different aggregate resistances; fitted D can absorb that
   mismatch. Test matching the 1D coefficient to total PNM boundary conductance / area.
6. **Bed truncation.** The pore domain ends at the 99th percentile bead-top height,
   while adsorption uses full particle masses. Check consistency with the 1D solid
   capacity and quantify the small mass/domain mismatch before claiming identical beds.

## Publication assessment

The physical reduction and the weak porosity baseline form a credible methods
story. R2 is not an acceptance threshold. A stronger submission would freeze
all choices and test additional unseen packings, report full-curve predictive
errors, quantify computational savings, and address the geometry/boundary
sensitivities above. Increasing the dataset solely to improve R2 is not the goal.

Because 5dp and the power law were selected after comparing results on this
ensemble, report current LOOCV as internal validation. Use independent beds or
nested model selection for an unbiased evaluation of the selection procedure:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

A high correlation between two outputs of the same approximate network does not
independently validate that network's physical conductance.

## Cleanup

Removed 264 tracked historical/build files before adding this review and its
outputs: old fixed-inlet campaign, old power-20 campaign, preliminary uncorrected
or superseded homogeneous fits, the nested older ML script, redundant presentation
ZIP, concept backup, and LaTeX intermediates. Kept production data and corrected
100/200/400-cell, Sobol and diffusion validation cases. No Git history was rewritten.
Optional transient-baseline and inlet-sensitivity tools remain useful comparisons.
