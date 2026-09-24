# Prespecified exploratory comparison

Written before training or inspecting GNN test predictions.

Target: the 0–5000 s mean uptake curve from the finite-inlet 20-bed PNM campaign.
One independent sample is an entire packing, not a particle or time point.
Outer evaluation: leave one bed out; all curves and particles of that bed excluded
from training. No tuning on outer scores. All variants reported, no best-run selection.

Direct GNN: particle positions and the known constant bead size/tube geometry only.
No pore network, inlet labels, G_access, fitted D, seed identifiers or uptake values
are input features. Distance graph cutoff 1.25 bead diameters. Two graph-convolution
layers, width 16, deterministic rotation-invariant radial/axial inputs and distance
weights. Whole-bed and near-bottom pooling. Fixed AdamW, learning rate .005,
weight decay .001, 200 epochs. Three initializations (11, 29, 47). Report each and
their equal-weight prediction ensemble. No hyperparameter or epoch selection.

Output: monotone piecewise-linear curves on fixed knots
0, 1, 5, 20, 60, 150, 350, 700, 1200, 2000, 3000, 4000, 5000 s.
Positive increments, zero initial loading, final loading bounded by the known
inlet-equilibrium loading. Fold-specific mean curves initialize the decoder.
Loss: time-weighted curve squared error scaled by the training mean final loading.

Baselines: training mean curve; fixed-alpha ridge on position-derived geometric
summaries predicting log curve increments; original raw G_access power-law closure
fitted on the other 19 fitted diffusivities, followed by the existing 100-cell 1D
solver. Historical choice of that closure/depth used this ensemble and must remain
identified as such. The GNN labels and physics-closure labels share the PNM origin.

Accuracy: time-weighted curve NRMSE normalized by each reference final uptake,
final-uptake MAE/MAPE/R2, and per-bed predictions. Report interpolation floor so
curve compression error is separate from learning error. Any train metrics are
diagnostic only. This is internal validation, not independent physical validation.

Timing: same CPU, single BLAS/PyTorch thread, warm inference, repeated timings.
Separate particle graph construction, resident model prediction (single and
three-model ensemble), cached-network G_access solve, algebraic closure, and 1D
uptake solve. Report model training separately. Cached-network timing excludes
PNM extraction/Sobol allocation and is NOT end-to-end raw-geometry speed.
No 1D solve is required by the direct-curve GNN. Model loading and disk I/O excluded.
Known common particle generation cost and original training-label simulation costs
are not timed. Large speedups alone do not compensate for poor predictive accuracy.
