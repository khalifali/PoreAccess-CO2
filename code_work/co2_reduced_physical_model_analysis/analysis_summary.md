# CO2 structure--performance analysis

Analysed **20** QA-passing, independent bed realizations.

All reported prediction metrics are leave-one-bed-out results. They are not training scores.

## Model comparison

| target | model | loo_r2 | loo_rmse | loo_mae |
| --- | --- | --- | --- | --- |
| final_uptake_mol_kg | porosity_only_ridge | -0.22105 | 0.03146 | 0.025484 |
| final_uptake_mol_kg | inlet_count_only_ridge | 0.52528 | 0.019616 | 0.015245 |
| final_uptake_mol_kg | effective_conductance_only_ridge | 0.92449 | 0.0078234 | 0.0067604 |
| final_uptake_mol_kg | second_layer_only_ridge | 0.51884 | 0.019749 | 0.01564 |
| final_uptake_mol_kg | conductance_plus_tortuosity_ridge | 0.91355 | 0.0083707 | 0.0072303 |
| final_uptake_mol_kg | conductance_plus_polar_coverage_ridge | 0.90807 | 0.0086322 | 0.0070277 |
| final_uptake_mol_kg | physical_three_ridge | 0.93109 | 0.0074734 | 0.0058807 |
| t50_final_uptake_s | porosity_only_ridge | -0.080779 | 65.195 | 52.51 |
| t50_final_uptake_s | inlet_count_only_ridge | 0.61722 | 38.799 | 31.27 |
| t50_final_uptake_s | effective_conductance_only_ridge | 0.1213 | 58.785 | 48.397 |
| t50_final_uptake_s | second_layer_only_ridge | 0.65084 | 37.056 | 28.949 |
| t50_final_uptake_s | conductance_plus_tortuosity_ridge | 0.3534 | 50.427 | 42.712 |
| t50_final_uptake_s | conductance_plus_polar_coverage_ridge | 0.11603 | 58.961 | 46.056 |
| t50_final_uptake_s | physical_three_ridge | 0.63091 | 38.099 | 29.932 |
| underutilized_particle_fraction | porosity_only_ridge | -0.22665 | 0.020842 | 0.016958 |
| underutilized_particle_fraction | inlet_count_only_ridge | 0.38689 | 0.014735 | 0.011759 |
| underutilized_particle_fraction | effective_conductance_only_ridge | 0.86288 | 0.0069682 | 0.0053519 |
| underutilized_particle_fraction | second_layer_only_ridge | 0.40111 | 0.014563 | 0.01132 |
| underutilized_particle_fraction | conductance_plus_tortuosity_ridge | 0.83738 | 0.0075885 | 0.0058624 |
| underutilized_particle_fraction | conductance_plus_polar_coverage_ridge | 0.84121 | 0.0074985 | 0.0057068 |
| underutilized_particle_fraction | physical_three_ridge | 0.81851 | 0.0080168 | 0.0063921 |

## Interpretation

This is an exploratory small-ensemble analysis. A descriptor should be considered physically useful only when its effect is reasonably stable, its bootstrap interval is informative, and the associated model improves on the porosity-only LOOCV baseline. Inlet-pore count describes inlet topology; it is not itself an inlet-area measurement.
