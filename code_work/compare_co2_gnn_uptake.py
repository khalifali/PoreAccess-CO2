#!/usr/bin/env python3
"""Direct particle-graph uptake predictor versus G_access -> D_eff -> 1D.

See co2_gnn_curve_comparison/PROTOCOL.md for the frozen exploratory protocol.
Run from code_work. No production data or model outputs are overwritten.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from extract_co2_inlet_accessibility import effective_conductance
from extract_co2_pore_network import read_last_snapshot, particle_data
from solve_co2_homogeneous_1d import HomogeneousBed

KNOTS = np.array([0,1,5,20,60,150,350,700,1200,2000,3000,4000,5000.])
DP, RADIUS = .0016, .008
QEQ = 5.332*(5.093e-5*8.314462618*298.15*6.05)/(1+5.093e-5*8.314462618*298.15*6.05)


def time_weights(t):
    w = np.r_[(t[1]-t[0])/2, (t[2:]-t[:-2])/2, (t[-1]-t[-2])/2]
    return w/w.sum()


def interpolation_matrix(times):
    return np.column_stack([np.interp(times, KNOTS, np.eye(len(KNOTS))[i]) for i in range(len(KNOTS))])


def particle_graph(xyz):
    """No network descriptors or labels: all features derive from particle positions."""
    p = np.asarray(xyz, float)/DP
    pairs = cKDTree(p).query_pairs(1.25, output_type='ndarray')
    n = len(p)
    distance = np.linalg.norm(p[pairs[:,0]]-p[pairs[:,1]], axis=1)
    row = np.r_[pairs[:,0], pairs[:,1], np.arange(n)]
    col = np.r_[pairs[:,1], pairs[:,0], np.arange(n)]
    val = np.r_[np.exp(-distance), np.exp(-distance), np.ones(n)]
    deg = np.bincount(row, weights=val, minlength=n)
    val /= np.sqrt(deg[row]*deg[col])
    radial = np.linalg.norm(p[:,:2],axis=1)
    z=p[:,2]
    count=np.bincount(pairs.ravel(),minlength=n)
    features=np.column_stack([z/25,radial/5,(5-radial)/5,np.exp(-z/2),
                              np.minimum(z,5)/5,count/12,np.ones(n)])
    near=np.exp(-z/2); near/=near.sum()
    adj=torch.sparse_coo_tensor(np.vstack([row,col]),val.astype('float32'),(n,n)).coalesce()
    return dict(x=torch.tensor(features,dtype=torch.float32),adj=adj,
                near=torch.tensor(near,dtype=torch.float32),xyz=xyz)


def geometry_summary(xyz):
    p=np.asarray(xyz)/DP; z=p[:,2]; r=np.linalg.norm(p[:,:2],axis=1)
    nearest=cKDTree(p).query(p,k=2)[0][:,1]-1
    count=np.asarray(cKDTree(p).query_ball_point(p,1.25,return_length=True))-1
    result=[]
    for mask in [np.ones(len(p),bool),z<=5]:
        result.append(mask.mean())
        for v in [z,r,nearest,count]:
            a=v[mask]
            result.extend([a.mean(),a.std(),*np.quantile(a,[.1,.5,.9])])
    result.extend(np.histogram(z,bins=[0,1,2,3,5,10,15,20,25])[0]/len(z))
    return np.array(result)


def pack(graphs):
    xs=[];indices=[];values=[];groups=[];near=[];offset=0
    for i,g in enumerate(graphs):
        xs.append(g['x']);indices.append(g['adj'].indices()+offset);values.append(g['adj'].values())
        groups.append(torch.full((len(g['x']),),i,dtype=torch.long));near.append(g['near'])
        offset+=len(g['x'])
    return (torch.cat(xs),torch.sparse_coo_tensor(torch.cat(indices,dim=1),torch.cat(values),(offset,offset)).coalesce(),
            torch.cat(groups),torch.cat(near),len(graphs))


class CurveGNN(nn.Module):
    def __init__(self, mean_knots, width=16):
        super().__init__()
        self.layer1=nn.Linear(7,width);self.layer2=nn.Linear(width,width)
        self.head=nn.Sequential(nn.Linear(width*2,width),nn.Tanh(),nn.Linear(width,len(KNOTS)))
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
        mean_knots=np.asarray(mean_knots)
        increments=np.maximum(np.diff(mean_knots),1e-12)
        self.register_buffer('base_parts',torch.tensor(np.log(increments/increments.sum()),dtype=torch.float32))
        f=float(np.clip(mean_knots[-1]/QEQ,1e-6,1-1e-6))
        self.register_buffer('base_final',torch.tensor(np.log(f/(1-f)),dtype=torch.float32))

    def forward(self,batch):
        x,adj,group,near,n=batch
        h=torch.tanh(self.layer1(torch.sparse.mm(adj,x)))
        h=torch.tanh(self.layer2(torch.sparse.mm(adj,h)))
        whole=torch.zeros(n,h.shape[1],device=h.device).index_add(0,group,h)
        whole=whole/torch.bincount(group,minlength=n)[:,None]
        lower=torch.zeros_like(whole).index_add(0,group,h*near[:,None])
        logits=self.head(torch.cat([whole,lower],dim=1))
        final=QEQ*torch.sigmoid(self.base_final+logits[:,0])
        increments=final[:,None]*torch.softmax(self.base_parts+logits[:,1:],dim=1)
        return torch.cat([torch.zeros(n,1,device=h.device),increments.cumsum(dim=1)],dim=1)


def train_model(graphs, curves, times, initialization, epochs):
    torch.manual_seed(initialization)
    mean_knots=np.mean([np.interp(KNOTS,times,c) for c in curves],axis=0)
    model=CurveGNN(mean_knots)
    batch=pack(graphs)
    target=torch.tensor(curves,dtype=torch.float32)
    interp=torch.tensor(interpolation_matrix(times),dtype=torch.float32)
    weights=torch.tensor(time_weights(times),dtype=torch.float32)
    scale=float(np.mean(curves[:,-1]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
    start=perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted=model(batch)@interp.T
        loss=(((predicted-target)/scale)**2*weights).sum(dim=1).mean()
        loss.backward();optimizer.step()
    elapsed=perf_counter()-start
    model.eval()
    with torch.inference_mode():
        pred=(model(batch)@interp.T).numpy()
    train_nrmse=np.sqrt(np.sum((pred-curves)**2*time_weights(times),axis=1))/curves[:,-1]
    return model,elapsed,float(train_nrmse.mean())


def bed_model(row):
    return HomogeneousBed(SimpleNamespace(cells=100,bed_height=row.bed_height_m,porosity=row.porosity,
        tube_radius=RADIUS,particle_density=1600.,temperature=298.15,qsat=5.332,b_pa=5.093e-5,
        toth_exponent=1.,k_ldf=8.55e-3,inlet_mass_transfer_coefficient=None,
        boundary_diffusivity=1.5e-5,inlet_transfer_length=.0008,inlet_concentration=6.05,
        initial_concentration=0.,initial_loading=0.,t_end=5000.,rtol=1e-6,atol=1e-10,max_step=np.inf))


def access_solve(net):
    return effective_conductance(len(net['pore_xyz']),net['throat_conns'],
        1.5e-5*net['throat_area']/net['throat_length'],net['pore_inlet'].astype(bool),
        net['pore_xyz'][:,2]>=net['pore_xyz'][:,2].min()+5*DP)[0]


def timed(fn,repeats):
    fn()  # warmup
    samples=[]
    for _ in range(repeats):
        start=perf_counter();fn();samples.append(perf_counter()-start)
    return float(np.median(samples)),float(np.quantile(samples,.1)),float(np.quantile(samples,.9))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('co2_gnn_curve_comparison'))
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--initializations',type=int,nargs='+',default=[11,29,47])
    parser.add_argument('--timing-repeats',type=int,default=5)
    a=parser.parse_args()
    if a.epochs<1 or a.timing_repeats<1: parser.error('Counts must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    threadpool_limits(limits=1)
    a.output.mkdir(exist_ok=True,parents=True)
    data=pd.read_csv('co2_effective_diffusivity_closure/closure_dataset.csv').sort_values('seed').reset_index(drop=True)
    if len(data)!=20 or data.seed.nunique()!=20: raise ValueError('Expected 20 distinct beds')
    graphs=[];positions=[];networks=[];curves=[];summaries=[];hashes=[];input_audit=[]
    for row in data.itertuples():
        path=Path(f'co2_pore_networks_power22/seed_{row.seed}/pore_network.npz')
        with np.load(path) as z: net={k:z[k] for k in z.files}
        # Positions only are passed to the GNN. PNM data below is used solely for the physical baseline.
        xyz=net['particle_xyz']
        raw_path=Path(f'co2_13x_seed_{row.seed}/particles_final.dump')
        columns, raw_values, _=read_last_snapshot(raw_path)
        _, raw_xyz, raw_radii=particle_data(columns,raw_values)
        np.testing.assert_allclose(xyz,raw_xyz,rtol=0,atol=0)
        np.testing.assert_allclose(net['particle_radius'],raw_radii,rtol=0,atol=0)
        input_audit.append(dict(seed=int(row.seed),particles=len(xyz),matches_raw_dem=True,
                               raw_dump_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest()))
        positions.append(xyz);graphs.append(particle_graph(xyz));networks.append(net)
        summaries.append(geometry_summary(xyz))
        ts_path=Path(f'co2_adsorption_campaign_finite_inlet_5000s/seed_{row.seed}/timeseries.csv')
        ts=pd.read_csv(ts_path)
        if curves: np.testing.assert_allclose(ts.time_s,times,rtol=0,atol=1e-10)
        else: times=ts.time_s.to_numpy()
        curves.append(ts.mean_loading_mol_kg.to_numpy())
        hashes.append(dict(seed=int(row.seed),network_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           curve_sha256=hashlib.sha256(ts_path.read_bytes()).hexdigest()))
    pd.DataFrame(input_audit).to_csv(a.output/'particle_input_audit.csv',index=False)
    curves=np.array(curves);summaries=np.array(summaries);w=time_weights(times);interp=interpolation_matrix(times)
    if not np.isfinite(curves).all() or np.any(np.diff(curves,axis=1)<-1e-10): raise ValueError('Invalid curves')
    metadata=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,cpu=platform.processor(),
        platform=platform.platform(),threads=1,cuda_used=False,knots_s=KNOTS.tolist(),inputs=hashes)
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                metadata['cpu_model']=line.split(':',1)[1].strip()
                break
    (a.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    metrics=[];outputs=[];timings=[];training=[];raw_g=data.effective_inlet_to_interior_conductance_m3_s.to_numpy()
    for i,row in enumerate(data.itertuples()):
        keep=np.arange(len(data))!=i
        fold_predictions={'mean_curve':curves[keep].mean(axis=0)}
        knots=np.array([np.interp(KNOTS,times,c) for c in curves[keep]])
        increments=np.maximum(np.diff(knots,axis=1),1e-12)
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=10.))
        ridge.fit(summaries[keep],np.log(increments))
        inc=np.exp(ridge.predict(summaries[[i]])[0]);inc*=min(1.,QEQ/inc.sum())
        fold_predictions['geometry_ridge']=np.r_[0.,inc.cumsum()]@interp.T
        exponent,intercept=np.polyfit(np.log(raw_g[keep]),np.log(data.effective_diffusivity_m2_s.to_numpy()[keep]),1)
        deff=float(np.exp(intercept+exponent*np.log(raw_g[i])))
        physical=bed_model(row)
        sol=physical.solve(deff,times)
        fold_predictions['G_access_power_1D']=physical.inventories(sol).mean_loading_mol_kg.to_numpy()
        model_predictions=[];models=[];single=pack([graphs[i]])
        for init in a.initializations:
            model,seconds,train_error=train_model([g for j,g in enumerate(graphs) if keep[j]],curves[keep],times,init,a.epochs)
            with torch.inference_mode(): pred=model(single).numpy()[0]@interp.T
            models.append(model);model_predictions.append(pred);fold_predictions[f'gnn_init_{init}']=pred
            training.append(dict(test_seed=row.seed,initialization=init,epochs=a.epochs,training_s=seconds,
                                 mean_train_curve_nrmse=train_error,training_seeds=' '.join(map(str,data.seed[keep]))))
            print(f'bed {i+1}/20 seed {row.seed}, init {init}: train NRMSE {train_error:.4f}, {seconds:.1f}s',flush=True)
        fold_predictions['gnn_ensemble']=np.mean(model_predictions,axis=0)
        fold_predictions['interpolation_oracle']=np.interp(KNOTS,times,curves[i])@interp.T
        for name,pred in fold_predictions.items():
            err=pred-curves[i]
            metrics.append(dict(seed=row.seed,model=name,curve_nrmse=float(np.sqrt(w@(err*err))/curves[i,-1]),
                final_reference=curves[i,-1],final_prediction=pred[-1],final_relative_error=err[-1]/curves[i,-1],
                monotonic=bool(np.all(np.diff(pred)>=-1e-9)),within_bounds=bool(pred.min()>=-1e-9 and pred.max()<=QEQ+1e-9)))
            outputs.append(pd.DataFrame(dict(seed=row.seed,model=name,time_s=times,reference=curves[i],prediction=pred)))
        def infer_single():
            with torch.inference_mode(): return models[0](single).numpy()[0]@interp.T
        def infer_ensemble():
            with torch.inference_mode(): return np.mean([m(single).numpy()[0]@interp.T for m in models],axis=0)
        stages={'particle_graph_build':lambda:pack([particle_graph(positions[i])]),
                'gnn_single_curve':infer_single,'gnn_ensemble_curve':infer_ensemble,
                'G_access_cached_network':lambda:access_solve(networks[i]),
                'power_law_mapping':lambda:float(np.exp(intercept+exponent*np.log(raw_g[i]))),
                'one_D_uptake':lambda:physical.inventories(physical.solve(deff,times))}
        for stage,fn in stages.items():
            median,lo,hi=timed(fn,a.timing_repeats)
            timings.append(dict(seed=row.seed,stage=stage,median_s=median,p10_s=lo,p90_s=hi,repeats=a.timing_repeats))
        pd.DataFrame(metrics).to_csv(a.output/'per_bed_metrics.csv',index=False)
        pd.DataFrame(training).to_csv(a.output/'training.csv',index=False)
        pd.DataFrame(timings).to_csv(a.output/'timings.csv',index=False)
        pd.concat(outputs,ignore_index=True).to_csv(a.output/'held_out_curves.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        if i==0:
            torch.save(dict(state_dict=models[0].state_dict(),mean_knots=knots.mean(axis=0).tolist(),test_seed=int(row.seed),
                            knots_s=KNOTS.tolist(),training_seeds=data.seed[keep].tolist()),a.output/'example_held_out_model.pt')
    result=pd.DataFrame(metrics)
    summary=[]
    for name,g in result.groupby('model'):
        summary.append(dict(model=name,mean_curve_nrmse=g.curve_nrmse.mean(),median_curve_nrmse=g.curve_nrmse.median(),
            worst_curve_nrmse=g.curve_nrmse.max(),final_r2=r2_score(g.final_reference,g.final_prediction),
            final_mape_percent=100*abs(g.final_relative_error).mean(),all_monotonic=g.monotonic.all(),all_bounded=g.within_bounds.all()))
    summary=pd.DataFrame(summary);summary.to_csv(a.output/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    names=['mean_curve','geometry_ridge','G_access_power_1D','gnn_ensemble']
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.boxplot([100*result[result.model==n].curve_nrmse for n in names],tick_labels=['Mean curve','Geometry ridge','G_access + 1D','GNN ensemble'])
    ax.set(ylabel='Held out curve NRMSE [%]',title='Leave-one-bed-out evaluation on 20 packings')
    fig.savefig(a.output/'accuracy.png',dpi=200);fig.savefig(a.output/'accuracy.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    representatives=np.argsort(curves[:,-1])[[0,10,-1]]
    for ax,j in zip(axes,representatives):
        seed=int(data.seed.iloc[j]);ax.plot(times,curves[j],color='black',label='PNM reference')
        for n,c in [('G_access_power_1D','#0065BD'),('gnn_ensemble','#F37D20')]:
            match=next(f for f in outputs if f.seed.iloc[0]==seed and f.model.iloc[0]==n)
            ax.plot(times,match.prediction,color=c,label='G_access + 1D' if n=='G_access_power_1D' else 'GNN ensemble')
        ax.set(title=f'Bed {seed}',xlabel='Time [s]',ylabel='Mean uptake [mol/kg]')
    axes[0].legend(fontsize=8)
    fig.savefig(a.output/'held_out_examples.png',dpi=200);fig.savefig(a.output/'held_out_examples.pdf');plt.close(fig)
    tm=pd.DataFrame(timings).pivot(index='seed',columns='stage',values='median_s')
    tm['gnn_single_with_graph']=tm.particle_graph_build+tm.gnn_single_curve
    tm['gnn_ensemble_with_graph']=tm.particle_graph_build+tm.gnn_ensemble_curve
    tm['physics_cached_network_to_curve']=tm.G_access_cached_network+tm.power_law_mapping+tm.one_D_uptake
    tm.to_csv(a.output/'timing_totals.csv')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    columns=['gnn_single_with_graph','gnn_ensemble_with_graph','G_access_cached_network','one_D_uptake','physics_cached_network_to_curve']
    ax.bar(range(len(columns)),[1000*tm[c].median() for c in columns],color=['#F37D20','#F37D20','#0065BD','#0065BD','#0065BD'])
    ax.set(yscale='log',ylabel='Median CPU time [ms, log scale]',xticks=range(len(columns)),
           xticklabels=['GNN + graph','3 GNNs + graph','G_access solve','1D solve','G_access + 1D'],title='Warm inference; PNM extraction excluded')
    fig.savefig(a.output/'timing.png',dpi=200);fig.savefig(a.output/'timing.pdf');plt.close(fig)


if __name__=='__main__':main()
