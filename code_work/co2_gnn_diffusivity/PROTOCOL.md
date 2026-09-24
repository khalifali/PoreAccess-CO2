# Direct effective diffusivity experiment

Fixed before training: reuse the regional particle GNN, width 32, three residual layers, with a scalar log diffusivity output. Standardize log targets using training beds only. No uptake curves are loaded or predicted. Labels are the existing fitted effective diffusivities. GNN inputs contain particle geometry only, without pore inlet labels or conductance.

Five shuffled outer folds (seed 20260924), 16 training and four test beds. Within each training fold use 12/4 inner training/validation split (seed 900 + zero based fold). Select weight decay 1e-5 or 1e-4 and training epoch by inner log RMSE; AdamW lr .001, max 1500 epochs, check every 10, patience 100, clip gradients at 5. Refit on 16 beds with seeds 11,29,47 and average log predictions. Compare with power law G_access fitted on the identical training beds, arithmetic mean and geometric mean baselines. Report MAPE, R2 and log RMSE on all 20 held out predictions.

Time graph construction plus three model inference against cached pore network conductance solve plus power law evaluation, five warm repeats per bed on one CPU thread. Exclude disk loading and pore network extraction; no 1D transient solve. Do not claim end to end speedup from cached network timings.

These are internal exploratory results on an already studied campaign, not an independent validation dataset. Targets are fitted simulation quantities, not measured material properties.
