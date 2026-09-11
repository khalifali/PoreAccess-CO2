# CO2 structure--performance analysis

Analysed **20** QA-passing, independent bed realizations.

All reported prediction metrics are leave-one-bed-out results. They are not training scores.

## Model comparison

| target | model | loo_r2 | loo_rmse | loo_mae |
| --- | --- | --- | --- | --- |
| final_uptake_mol_kg | porosity_only_ridge | -0.22105 | 0.03146 | 0.025484 |
| final_uptake_mol_kg | compact_ridge | 0.4192 | 0.021697 | 0.017623 |
| final_uptake_mol_kg | compact_random_forest | 0.37992 | 0.022419 | 0.017669 |
| t50_final_uptake_s | porosity_only_ridge | -0.080779 | 65.195 | 52.51 |
| t50_final_uptake_s | compact_ridge | 0.60447 | 39.44 | 32.967 |
| t50_final_uptake_s | compact_random_forest | 0.30479 | 52.288 | 41.661 |
| underutilized_particle_fraction | porosity_only_ridge | -0.22665 | 0.020842 | 0.016958 |
| underutilized_particle_fraction | compact_ridge | 0.26102 | 0.016177 | 0.012458 |
| underutilized_particle_fraction | compact_random_forest | 0.42652 | 0.01425 | 0.011087 |

## Interpretation

This is an exploratory small-ensemble analysis. A descriptor should be considered physically useful only when its effect is reasonably stable, its bootstrap interval is informative, and the associated model improves on the porosity-only LOOCV baseline. Inlet-pore count describes inlet topology; it is not itself an inlet-area measurement.
