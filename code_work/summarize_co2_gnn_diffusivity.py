#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path('co2_gnn_diffusivity');s=pd.read_csv(root/'summary.csv');p=pd.read_csv(root/'predictions.csv');t=pd.read_csv(root/'timings.csv')
    assert p.groupby('model').seed.nunique().eq(20).all() and not p.duplicated(['model','seed']).any()
    for r in pd.read_csv(root/'selection.csv').itertuples():
        a,b,c=map(lambda x:set(x.split()),[r.train_seeds,r.validation_seeds,r.test_seeds])
        assert len(a)==12 and len(b)==len(c)==4 and not(a&b or a&c or b&c)
    fig,axs=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    for ax,name,title in zip(axs,['gnn','G_access','mean_diffusivity'],['Particle GNN','G_access','Mean diffusivity']):
        d=p[p.model==name];lo=min(d.reference.min(),d.prediction.min())*1e6;hi=max(d.reference.max(),d.prediction.max())*1e6
        ax.scatter(d.reference*1e6,d.prediction*1e6,color='#0065BD');ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(title=title,xlabel='Fitted diffusivity [10⁻⁶ m²/s]',ylabel='Predicted diffusivity [10⁻⁶ m²/s]');ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(root/f'diffusivity_predictions.{ext}',dpi=180)
    lines=['# Direct GNN prediction of effective diffusivity','','All predictions below are for held out beds. No uptake curves were predicted.','','| Model | Relative error (MAPE) | R² | Log RMSE |','|---|---:|---:|---:|']
    for r in s.itertuples():lines.append(f'| {r.model} | {r.mape_percent:.2f}% | {r.r2:.3f} | {r.log_rmse:.3f} |')
    lines+=['','The GNN remains essentially at the mean baseline: direct scalar prediction did not provide useful generalization in this experiment. G_access is substantially more accurate. The GNN is also slower than the cached-network conductance calculation under the measured conditions; pore extraction costs could change an end-to-end comparison.', '', '![Diffusivity predictions](diffusivity_predictions.png)','','The target is effective diffusivity fitted from the existing PNM simulations. The GNN receives only particle geometry. It uses 32 latent features per particle, three residual message layers and regional mean/max pooling, followed by a scalar log diffusivity prediction. Training targets are standardized using training beds only. Three initializations are averaged in log space.', '',
    'Five outer folds each hold out four beds. A separate inner 12/4 split selects weight decay and epoch, then refits on all 16 outer training beds. G_access power law and both mean baselines use the identical training beds. Detailed split membership and training histories are saved. See PROTOCOL.md for the fixed settings.', '',
    f'Median warm CPU times: graph construction {1000*t.graph_s.median():.2f} ms; three GNN inference passes {1000*t.inference_s.median():.2f} ms; combined GNN {1000*t.gnn_total_s.median():.2f} ms; cached pore network G_access calculation plus diffusivity mapping {1000*t.G_access_s.median():.2f} ms. Five repeats per bed, one CPU thread. Disk loading and pore network extraction are excluded. There is no 1D transient solve in this comparison.', '',
    'This is exploratory internal validation on 20 previously studied beds. It does not establish performance for new packing regimes or independently measured diffusivities. Many particles in a graph do not increase the number of independent labeled beds.', '',
    'Reproduce from code_work: `python compare_co2_gnn_diffusivity.py` then `python summarize_co2_gnn_diffusivity.py`.']
    (root/'README.md').write_text('\n'.join(lines)+'\n');print(s.to_string(index=False));print(t.median(numeric_only=True).to_string())
if __name__=='__main__':main()
