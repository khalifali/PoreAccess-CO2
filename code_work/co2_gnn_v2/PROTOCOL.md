# GNN v2 exploratory protocol

Fixed before v2 training: 32 latent features, three residual message-passing
layers, LayerNorm, distance and signed axial displacement message bases with
learned feature projections, regional mean/max pooling (bottom 0–2dp, 2–5dp,
whole bed), 13-knot constrained decoder unchanged. AdamW lr .001, gradient
clipping 5, weight decay candidates 1e-5 and 1e-4, up to 1500 epochs.

First fit the first three beds in seed order as a training-only diagnostic.
Stop at <1% mean training curve NRMSE or 1500 epochs. This is not validation.

Then use five shuffled outer folds (seed 20260924), holding out four entire beds
per fold. The other 16 beds have a deterministic 12/4 inner training/validation
split (seed 900+fold). Select weight decay and epoch from inner validation only,
checking every 10 epochs with patience 100. Refit on all 16 for the selected
number of epochs using three initializations (11,29,47), average the predictions.
Report each initialization and the ensemble. Test beds never select settings.

Five outer folds keep the nested comparison tractable on this CPU. Refit both
the original GNN and G_access power-law + 1D on these SAME 16 training beds.
Do not compare their historical 19-bed training results as matched baselines.
This is further exploratory model development on the existing 20-bed ensemble,
not an untouched external validation dataset. No claim of a universally best GNN.
