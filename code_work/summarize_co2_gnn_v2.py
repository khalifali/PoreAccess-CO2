#!/usr/bin/env python3
"""Report all models on the same five outer folds; no selection by test score."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def main():
    root=Path('co2_gnn_v2')
    summary=pd.read_csv(root/'summary.csv').set_index('model')
    per=pd.read_csv(root/'per_bed_metrics.csv');selection=pd.read_csv(root/'selection.csv')
    history=pd.read_csv(root/'training_history.csv');timing=pd.read_csv(root/'timings.csv')
    curves=pd.read_csv(root/'held_out_curves.csv.gz');diagnostic=json.loads((root/'diagnostic.json').read_text())
    assert per.groupby('model').seed.nunique().eq(20).all()
    assert not per.duplicated(['seed','model']).any()
    for row in selection.itertuples():
        train=set(row.train_seeds.split());valid=set(row.validation_seeds.split());test=set(row.test_seeds.split())
        assert len(train)==12 and len(valid)==4 and len(test)==4
        assert not(train&valid or train&test or valid&test)
    names=['mean_curve','gnn_v1_ensemble','gnn_v2_ensemble','G_access_power_1D']
    labels=['Mean curve','Original GNN','Regional GNN','G_access + 1D']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.boxplot([100*per[per.model==n].curve_nrmse for n in names],tick_labels=labels)
    ax.set(ylabel='Held out curve NRMSE [%]',title='Same five outer folds, four test beds per fold')
    for ext in ['png','pdf']:fig.savefig(root/f'accuracy.{ext}',dpi=200)
    plt.close(fig)
    final=curves[(curves.model=='mean_curve')&(curves.time_s==5000)].sort_values('reference')
    representatives=final.seed.iloc[[0,10,-1]].tolist()
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,seed in zip(axes,representatives):
        ref=curves[(curves.seed==seed)&(curves.model=='mean_curve')]
        ax.plot(ref.time_s,ref.reference,color='black',label='PNM reference')
        for name,color,label in [('gnn_v1_ensemble','0.65','Original GNN'),('gnn_v2_ensemble','#F37D20','Regional GNN'),('G_access_power_1D','#0065BD','G_access + 1D')]:
            c=curves[(curves.seed==seed)&(curves.model==name)]
            ax.plot(c.time_s,c.prediction,color=color,label=label)
        ax.set(title=f'Bed {seed}',xlabel='Time [s]',ylabel='Uptake [mol/kg]')
    axes[0].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(root/f'held_out_examples.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for fold,g in selection.groupby('fold'):
        row=g.sort_values(['inner_nrmse','weight_decay','selected_epoch']).iloc[0]
        h=history[(history.fold==fold)&(history.stage=='inner')&np.isclose(history.weight_decay,row.weight_decay,rtol=1e-8,atol=0)]
        axes[0].plot(h.epoch,100*h.train_nrmse,label=f'Fold {fold+1}')
        axes[1].plot(h.epoch,100*h.validation_nrmse,label=f'Fold {fold+1}')
    axes[0].set(title='Inner training beds',xlabel='Epoch',ylabel='Mean curve NRMSE [%]')
    axes[1].set(title='Inner validation beds',xlabel='Epoch',ylabel='Mean curve NRMSE [%]')
    axes[1].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(root/f'learning_curves.{ext}',dpi=200)
    plt.close(fig)
    original=summary.loc['gnn_v1_ensemble'];new=summary.loc['gnn_v2_ensemble'];physics=summary.loc['G_access_power_1D']
    paired=per.pivot(index='seed',columns='model',values='curve_nrmse')
    lines=['# Regional residual GNN comparison','',
           '## Result','',
           f'The proposed model fits three training beds to **{100*diagnostic["train_nrmse"]:.2f}%** curve NRMSE in **{diagnostic["epoch"]} epochs**. This is a training-only diagnostic, not a generalization score.', '',
           f'On the same five outer validation folds, the proposed GNN has mean curve NRMSE **{100*new.mean_curve_nrmse:.2f}%**, versus **{100*original.mean_curve_nrmse:.2f}%** for the original GNN and **{100*physics.mean_curve_nrmse:.2f}%** for G_access plus 1D.', '',
           '| Model | Mean curve NRMSE | Worst curve NRMSE | Final uptake MAPE | Final uptake R2 |','|---|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        s=summary.loc[name];lines.append(f'| {label} | {100*s.mean_curve_nrmse:.2f}% | {100*s.worst_curve_nrmse:.2f}% | {s.final_mape_percent:.2f}% | {s.final_r2:.3f} |')
    lines+=['','![Matched accuracy comparison](accuracy.png)','','![Held-out examples](held_out_examples.png)','',
            f'The regional GNN improves on the original on {(paired.gnn_v2_ensemble<paired.gnn_v1_ensemble).sum()}/20 beds, and on G_access plus 1D on {(paired.gnn_v2_ensemble<paired.G_access_power_1D).sum()}/20 beds. No model or initialization is selected using these outer scores.', '',
            'The improvement over the original GNN is small, and the model remains close to the mean-curve baseline. Inner training errors fall below 1% while validation errors generally rise with longer training. This indicates overfitting in this experiment: greater fitting capacity has not delivered useful generalization. These results do not justify replacing the G_access closure with this GNN.', '',
            '## Architecture and selection','',
            f'The model has **{diagnostic["parameters"]:,} trainable parameters**, 32 particle features per layer, three residual message-passing layers with LayerNorm, and separate mean/max pooling over bottom 0–2dp, 2–5dp and the whole bed. These form a 192-component bed representation, followed by a 32-unit hidden readout.', '',
            'Distance and signed axial displacement enter fixed sparse geometric message bases, each with a learned feature projection at every layer. This is not a pore-network graph and uses no inlet labels, G_access or fitted diffusivity as GNN features. The constrained 13-knot decoder is unchanged: zero initial uptake, monotone increments and an equilibrium upper bound.', '',
            'AdamW uses learning rate .001 and gradient clipping 5. Each outer fold has 16 training beds and four untouched test beds. Within the 16, a deterministic 12/4 training/validation split selects weight decay (1e-5 or 1e-4) and epoch, up to 1500 epochs, evaluated every 10 epochs with patience 100. The chosen settings are then refitted on all 16 beds with initializations 11, 29 and 47. Their predictions are averaged.', '',
            '| Outer fold | Selected weight decay | Refit epochs | Inner validation NRMSE |','|---|---:|---:|---:|']
    for fold,g in selection.groupby('fold'):
        row=g.sort_values(['inner_nrmse','weight_decay','selected_epoch']).iloc[0]
        lines.append(f'| {fold+1} | {row.weight_decay:g} | {int(row.selected_epoch)} | {100*row.inner_nrmse:.2f}% |')
    lines+=['','All candidate scores, training/validation/test seed lists and learning curves are stored. A single inner holdout is used, not exhaustive inner cross-validation; selection remains noisy with only 16 development beds per fold.', '',
            '![Inner training and validation histories](learning_curves.png)', '', '## Timing','',
            f'Warm single-CPU-thread timing, median across 20 beds: graph preparation **{1000*timing.graph_build_s.median():.2f} ms**, three-model ensemble inference **{1000*timing.ensemble_inference_s.median():.2f} ms**, combined **{1000*timing.total_s.median():.2f} ms**. Each measurement uses five repeats after warmup and excludes loading files or weights.', '',
            'Do not treat timings from different benchmark runs as an exact matched speedup. The earlier G_access plus 1D timing also excluded pore-network extraction. The larger GNN adds computation relative to the original architecture.', '',
            '## Limits and interpretation','',
            'All methods here are refitted on the SAME outer training beds. These five-fold numbers should not be compared as identical experiments to the earlier leave-one-bed-out results, which trained on 19 beds. Curves use trapezoidal time-weighted error normalized by each test bed’s final PNM uptake.', '',
            'The three-bed diagnostic confirms fitting capacity, not generalization. This is exploratory development on the same 20-bed ensemble already used to develop the physics closure and the initial GNN. Inner validation prevents direct use of the current outer test labels for settings, but does not turn the campaign into a new external test set. No claim that one architecture is generally superior is warranted.', '',
            'Both models target the existing PNM simulations rather than experimental truth. Predictions at new bead sizes, sorbents, temperatures or inlet definitions are unvalidated. Curve constraints do not enforce local mass conservation.', '',
            '## Reproduce','',
            'From `code_work`:','', '```bash','python compare_co2_gnn_v2.py --diagnostic-only','python compare_co2_gnn_v2.py','python summarize_co2_gnn_v2.py','python -m unittest discover -s tests -p "test_gnn_v2.py" -v','```','',
            'The diagnostic and main comparison use separate output files in `co2_gnn_v2`; previous GNN and presentation artifacts remain unchanged.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    provenance={'torch':str(torch.__version__),'numpy':np.__version__,'platform':platform.platform(),
                'outer_split_seed':20260924,'inner_split_seed':'900 + zero-based fold','threads':1,
                'inputs':{}}
    for p in [Path('co2_effective_diffusivity_closure/closure_dataset.csv'),*sorted(Path('co2_pore_networks_power22').glob('seed_*/pore_network.npz')),
              *sorted(Path('co2_adsorption_campaign_finite_inlet_5000s').glob('seed_*/timeseries.csv'))]:
        provenance['inputs'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(summary.to_string());print(root/'README.md')


if __name__=='__main__':main()
