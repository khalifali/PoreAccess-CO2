#!/usr/bin/env python3
"""Build a readable report from the complete frozen GNN comparison outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('co2_gnn_curve_comparison'))
    a=p.parse_args();root=a.output
    summary=pd.read_csv(root/'summary.csv').set_index('model')
    per=pd.read_csv(root/'per_bed_metrics.csv')
    timing=pd.read_csv(root/'timing_totals.csv')
    training=pd.read_csv(root/'training.csv')
    if per.groupby('model').seed.nunique().min()!=20 or len(training)!=60:
        raise ValueError('Report requires the complete 20-bed, three-initialization comparison')
    errors=per.pivot(index='seed',columns='model',values='curve_nrmse')
    paired=100*(errors.gnn_ensemble-errors.G_access_power_1D).to_numpy()
    rng=np.random.default_rng(20260924)
    bootstrap=np.mean(paired[rng.integers(0,len(paired),(10000,len(paired)))],axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    pd.DataFrame([dict(comparison='GNN ensemble minus G_access plus 1D',
        mean_difference_percentage_points=paired.mean(),bootstrap_2p5=lo,bootstrap_97p5=hi,
        gnn_better_beds=int((paired<0).sum()),n_beds=20)]).to_csv(root/'paired_comparison.csv',index=False)
    label={'mean_curve':'Training mean curve','geometry_ridge':'Geometry ridge',
           'G_access_power_1D':'G_access power law + 1D','gnn_ensemble':'Direct GNN, three-model ensemble',
           'interpolation_oracle':'Interpolation of the true curve (not a predictor)'}
    names=['mean_curve','geometry_ridge','G_access_power_1D','gnn_ensemble']
    gnn=summary.loc['gnn_ensemble'];physics=summary.loc['G_access_power_1D'];mean=summary.loc['mean_curve']
    lines=['# Direct particle GNN versus the G_access uptake workflow','',
           '## Main result','',
           f'The GNN ensemble has mean held-out curve NRMSE **{100*gnn.mean_curve_nrmse:.2f}%**, compared with **{100*physics.mean_curve_nrmse:.2f}%** for G_access plus the 1D model and **{100*mean.mean_curve_nrmse:.2f}%** for the training mean curve.', '',
           'This evaluates one small, fixed GNN on only 20 independent beds. It does not establish the best achievable performance of GNNs or other machine-learning approaches. The G_access closure and 5dp depth were previously selected using this ensemble, so their reported result is internal validation of an existing choice.', '',
           '## Accuracy on exactly the same held-out beds','',
           '| Method | Mean curve NRMSE | Worst curve NRMSE | Final uptake MAPE | Final uptake R2 |',
           '|---|---:|---:|---:|---:|']
    for name in names:
        s=summary.loc[name]
        lines.append(f'| {label[name]} | {100*s.mean_curve_nrmse:.2f}% | {100*s.worst_curve_nrmse:.2f}% | {s.final_mape_percent:.2f}% | {s.final_r2:.3f} |')
    lines += ['', 'Curve NRMSE uses trapezoidal time weighting and divides each bed’s error by its PNM uptake at 5000 s. The three random initializations are not 60 independent test beds. Each of the 20 test beds is excluded entirely from that fold’s training.', '',
              '![Held-out accuracy](accuracy.png)','',
              '![Representative held-out curves](held_out_examples.png)','',
              f'The GNN has lower curve error than the physics closure on **{int((paired<0).sum())}/20** beds. Mean paired difference is **{paired.mean():+.2f} percentage points** (GNN minus physics). A descriptive paired bed bootstrap gives [{lo:+.2f}, {hi:+.2f}] percentage points; it does not account for dependence from overlapping cross-validation training sets.', '',
              '## GNN learning and curve representation','',
              f'The mean training-curve NRMSE over 60 fits is **{100*training.mean_train_curve_nrmse.mean():.2f}%**. Individual initialization test errors are:']
    for name in summary.index:
        if name.startswith('gnn_init_'):
            lines.append(f'- {name}: {100*summary.loc[name,"mean_curve_nrmse"]:.2f}% mean curve NRMSE.')
    lines += ['',f'Interpolating the true curve at the 13 prescribed knots gives mean NRMSE **{100*summary.loc["interpolation_oracle","mean_curve_nrmse"]:.3f}%**. This uses the test labels and is only a curve-representation diagnostic, never a predictive baseline.', '',
              'The decoder guarantees zero initial uptake, nondecreasing loading and an equilibrium upper bound at the campaign conditions. It does not enforce the adsorption mass-balance equations. Piecewise-linear interpolation is restricted to 0–5000 s; no extrapolation or new operating conditions are validated.', '',
              '## Inference timing','',
              'Median across the 20 per-bed timing medians, with five warm repeats on the same CPU and one computational thread. Inputs and weights are already resident in memory. Graph construction is included explicitly; model loading and disk I/O are excluded.', '',
              '| Stage | Median time |','|---|---:|']
    stages={'particle_graph_build':'Particle graph construction and tensor assembly',
            'gnn_single_curve':'One GNN, graph already available',
            'gnn_ensemble_curve':'Three GNNs, graph already available',
            'gnn_single_with_graph':'Particle positions to curve, one GNN',
            'gnn_ensemble_with_graph':'Particle positions to curve, three GNNs',
            'G_access_cached_network':'Steady G_access solve, pore network already available',
            'power_law_mapping':'Algebraic conductance to diffusivity mapping',
            'one_D_uptake':'100-cell 1D uptake solve',
            'physics_cached_network_to_curve':'Cached pore network to uptake curve'}
    for stage,display in stages.items():
        lines.append(f'| {display} | {1000*timing[stage].median():.3f} ms |')
    ratio=(timing.physics_cached_network_to_curve/timing.gnn_ensemble_with_graph).median()
    lines += ['',f'The median per-bed timing ratio is **{ratio:.1f}x** for the cached-network physics workflow relative to the GNN ensemble with graph construction. This is **not an end-to-end raw-particle speedup**: PNM extraction, Sobol volume allocation and geometry-to-1D porosity/height preparation are excluded from the physics timing.', '',
              '![Warm inference timing](timing.png)','',
              f'The 60 GNN training loops took **{training.training_s.sum():.1f} s** in total. This excludes graph preparation and evaluation. Training-label PNM simulations and fitted-diffusivity generation are also excluded and must be considered when assessing total development cost.', '',
              '## Interpretation','',
              'Speed should be compared at acceptable predictive accuracy. This GNN stays near the mean-curve baseline even on its training beds, indicating limited learning in this prescribed representation/optimization setup; it is not evidence that neural networks cannot represent inlet effects. More epochs, richer geometry, normalization, architecture changes or additional independent beds may change the outcome, but selecting those using these test scores would require a new validation design.', '',
              'The physics and GNN workflows both predict this particular PNM model, including its approximate inlet representation. Neither is independently validated against experimental uptake or resolved transport. The GNN uses no pore topology or inlet labels, so its task is harder than learning from the already-extracted network.', '',
              '## Reproduction and exact inputs','',
              'Run from `code_work` with NumPy, SciPy, pandas, Matplotlib, scikit-learn, threadpoolctl and CPU-capable PyTorch installed:', '',
              '```bash','python compare_co2_gnn_uptake.py','python summarize_co2_gnn_comparison.py',
              'python -m unittest discover -s tests -p "test_gnn_uptake.py" -v','```','', 'To demonstrate direct prediction on the first held-out bed without building a pore network:', '', '```bash', 'python predict_co2_gnn_uptake.py --particle-dump co2_13x_seed_18427/particles_final.dump --output co2_gnn_curve_comparison/example_direct_prediction.csv', '```', '',
              'The protocol was written before inspecting test predictions. Hyperparameters and 200 epochs are fixed, with no selection of the best initialization. Geometry ridge uses fold-fitted standardization and fixed regularization. Training targets, including the mean curve initializing the GNN decoder, exclude the test bed.', '',
              'Particle positions are read from the existing NPZ files for convenience, but the GNN graph builder reads only the particle-position array. It constructs its own distance-neighbour graph. The physically fixed bead diameter and tube geometry supply length scales; G_access, pore labels, fitted diffusivity, seed ID and adsorption descriptors are not GNN features.', '',
              'All per-bed curves (`held_out_curves.csv.gz`), training seeds, timings and file hashes are included. The particle-input audit verifies exact agreement with the original DEM dumps for all 20 beds. `example_held_out_model.pt` is the first fold’s first initialization, not a production model trained on all 20 beds. Representative plots show the lowest, middle and highest final-uptake beds and do not determine reported aggregate scores.', '',
              'Structural tests verify particle-permutation invariance, rotation about the tube axis, independence between graphs in a batch, interpolation and decoder bounds.', '',
              'Implementation references: [PyTorch CPU threading and tensor operations](https://docs.pytorch.org/docs/stable/torch.html) and [scikit-learn guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(root/'README.md')


if __name__=='__main__':main()
