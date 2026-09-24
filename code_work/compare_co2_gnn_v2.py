#!/usr/bin/env python3
"""Residual geometry-message GNN, regional pooling and nested bed-level validation."""
from compare_co2_gnn_uptake import *
import copy
from sklearn.model_selection import KFold


class RegionalGNN(nn.Module):
    def __init__(self,mean_knots):
        super().__init__()
        self.encoder=nn.Linear(7,32)
        self.messages=nn.ModuleList([nn.ModuleList([nn.Linear(32,32,bias=False) for _ in range(3)]) for _ in range(3)])
        self.self_layers=nn.ModuleList([nn.Linear(32,32) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(32) for _ in range(3)])
        self.head=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,32),nn.SiLU(),nn.Linear(32,len(KNOTS)))
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
        increments=np.maximum(np.diff(mean_knots),1e-12)
        self.register_buffer('base_parts',torch.tensor(np.log(increments/increments.sum()),dtype=torch.float32))
        f=np.clip(mean_knots[-1]/QEQ,1e-6,1-1e-6)
        self.register_buffer('base_final',torch.tensor(np.log(f/(1-f)),dtype=torch.float32))

    def forward(self,batch):
        x,matrices,groups,regions,n=batch
        h=torch.nn.functional.silu(self.encoder(x))
        for messages,own,norm in zip(self.messages,self.self_layers,self.norms):
            message=own(h)
            for projection,matrix in zip(messages,matrices):
                message=message+projection(torch.sparse.mm(matrix,h))
            h=norm(h+torch.nn.functional.silu(message))
        pools=[]
        for mask in regions:
            g=groups[mask];v=h[mask]
            mean=torch.zeros(n,32).index_add(0,g,v)/torch.bincount(g,minlength=n).clamp_min(1)[:,None]
            maximum=torch.full((n,32),-1e9).scatter_reduce(0,g[:,None].expand(-1,32),v,reduce='amax',include_self=True)
            pools.extend([mean,maximum])
        logits=self.head(torch.cat(pools,dim=1))
        final=QEQ*torch.sigmoid(self.base_final+logits[:,0])
        inc=final[:,None]*torch.softmax(self.base_parts+logits[:,1:],dim=1)
        return torch.cat([torch.zeros(n,1),inc.cumsum(dim=1)],dim=1)


def graph_v2(xyz):
    g=particle_graph(xyz);idx=g['adj'].indices();value=g['adj'].values()
    positions=torch.tensor(xyz/DP,dtype=torch.float32)
    delta=positions[idx[1]]-positions[idx[0]]
    distance=torch.linalg.vector_norm(delta,dim=1)
    # Fixed geometric bases with learned feature projections at every message layer.
    g['matrices']=[g['adj'],torch.sparse_coo_tensor(idx,value*(distance-1),g['adj'].shape).coalesce(),
                   torch.sparse_coo_tensor(idx,value*delta[:,2],g['adj'].shape).coalesce()]
    z=positions[:,2];g['regions']=[z<=2,(z>2)&(z<=5),torch.ones(len(z),dtype=torch.bool)]
    if any(not m.any() for m in g['regions']):raise ValueError('Empty pooling region')
    return g


def pack_v2(graphs):
    offsets=np.cumsum([0]+[len(g['x']) for g in graphs]);n=int(offsets[-1]);mats=[]
    for k in range(3):
        idx=torch.cat([g['matrices'][k].indices()+int(offsets[i]) for i,g in enumerate(graphs)],dim=1)
        val=torch.cat([g['matrices'][k].values() for g in graphs])
        mats.append(torch.sparse_coo_tensor(idx,val,(n,n)).coalesce())
    return (torch.cat([g['x'] for g in graphs]),mats,
            torch.cat([torch.full((len(g['x']),),i,dtype=torch.long) for i,g in enumerate(graphs)]),
            [torch.cat([g['regions'][k] for g in graphs]) for k in range(3)],len(graphs))


def fit(graphs,curves,times,seed,wd,max_epochs,validation=None,diagnostic=False):
    torch.manual_seed(seed)
    mean_knots=np.mean([np.interp(KNOTS,times,c) for c in curves],axis=0)
    model=RegionalGNN(mean_knots);batch=pack_v2(graphs)
    interp=torch.tensor(interpolation_matrix(times),dtype=torch.float32)
    weights=torch.tensor(time_weights(times),dtype=torch.float32)
    target=torch.tensor(curves,dtype=torch.float32);scale=curves[:,-1].mean()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=wd)
    if validation is not None:
        vb=pack_v2(validation[0]);vy=validation[1]
    best=float('inf');best_epoch=0;state=None;history=[];start=perf_counter()
    for epoch in range(1,max_epochs+1):
        model.train();optimizer.zero_grad();pred=model(batch)@interp.T
        loss=(((pred-target)/scale)**2*weights).sum(dim=1).mean();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if epoch%10==0 or epoch==max_epochs:
            model.eval()
            with torch.inference_mode():
                train_pred=(model(batch)@interp.T).numpy()
                train_error=float(np.mean(np.sqrt(np.sum((train_pred-curves)**2*time_weights(times),axis=1))/curves[:,-1]))
                if validation is not None:
                    valpred=(model(vb)@interp.T).numpy()
                    score=float(np.mean(np.sqrt(np.sum((valpred-vy)**2*time_weights(times),axis=1))/vy[:,-1]))
                else:score=train_error
            history.append(dict(epoch=epoch,train_nrmse=train_error,validation_nrmse=score if validation is not None else np.nan))
            if score<best-1e-6:
                best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
            if diagnostic and train_error<.01:break
            if validation is not None and epoch-best_epoch>=100:break
    # With a fixed refit budget, use the final state; validation never touches outer test beds.
    if validation is not None or diagnostic:model.load_state_dict(state)
    return model,best_epoch,best,perf_counter()-start,history


def load_data():
    data=pd.read_csv('co2_effective_diffusivity_closure/closure_dataset.csv').sort_values('seed').reset_index(drop=True)
    graphs=[];curves=[]
    for row in data.itertuples():
        with np.load(f'co2_pore_networks_power22/seed_{row.seed}/pore_network.npz') as net:
            graphs.append(graph_v2(net['particle_xyz']))
        ts=pd.read_csv(f'co2_adsorption_campaign_finite_inlet_5000s/seed_{row.seed}/timeseries.csv')
        if curves:np.testing.assert_allclose(ts.time_s,times)
        else:times=ts.time_s.to_numpy()
        curves.append(ts.mean_loading_mol_kg.to_numpy())
    return data,graphs,np.array(curves),times


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('co2_gnn_v2'))
    p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--max-epochs',type=int,default=1500)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);threadpool_limits(limits=1)
    data,graphs,curves,times=load_data();interp=interpolation_matrix(times);weights=time_weights(times)
    if a.diagnostic_only:
        model,epoch,error,seconds,hist=fit(graphs[:3],curves[:3],times,11,1e-5,a.max_epochs,diagnostic=True)
        pd.DataFrame(hist).to_csv(a.output/'three_bed_training.csv',index=False)
        (a.output/'diagnostic.json').write_text(json.dumps(dict(seeds=data.seed.iloc[:3].tolist(),epoch=epoch,
             train_nrmse=error,seconds=seconds,parameters=sum(p.numel() for p in model.parameters())),indent=2))
        print('Diagnostic:',epoch,error,seconds,flush=True);return
    results=[];outputs=[];selections=[];histories=[];timing=[]
    for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=20260924).split(data)):
        # A deterministic inner holdout entirely within the 16 outer-training beds.
        shuffled=np.random.default_rng(900+fold).permutation(train);inner_val=shuffled[:4];inner_train=shuffled[4:]
        candidates=[]
        for wd in [1e-5,1e-4]:
            model,epoch,score,seconds,hist=fit([graphs[j] for j in inner_train],curves[inner_train],times,11,wd,a.max_epochs,
                    validation=([graphs[j] for j in inner_val],curves[inner_val]))
            candidates.append((score,wd,epoch))
            histories.extend([dict(fold=fold,stage='inner',weight_decay=wd,initialization=11,**h) for h in hist])
            selections.append(dict(fold=fold,weight_decay=wd,selected_epoch=epoch,inner_nrmse=score,seconds=seconds,
                train_seeds=' '.join(map(str,data.seed.iloc[inner_train])),validation_seeds=' '.join(map(str,data.seed.iloc[inner_val])),
                test_seeds=' '.join(map(str,data.seed.iloc[test]))))
            print(f'fold {fold+1}: inner wd {wd}, epoch {epoch}, error {score:.4f}, {seconds:.1f}s',flush=True)
            pd.DataFrame(selections).to_csv(a.output/'selection.csv',index=False)
        _,wd,epochs=min(candidates)
        fold_pred={};models=[]
        for init in [11,29,47]:
            model,_,_,seconds,hist=fit([graphs[j] for j in train],curves[train],times,init,wd,epochs)
            models.append(model)
            with torch.inference_mode():prediction=model(pack_v2([graphs[j] for j in test])).numpy()@interp.T
            fold_pred[f'gnn_v2_init_{init}']=prediction
            histories.extend([dict(fold=fold,stage='refit',weight_decay=wd,initialization=init,**h) for h in hist])
            print(f'fold {fold+1}: refit {init}, {epochs} epochs, {seconds:.1f}s',flush=True)
        fold_pred['gnn_v2_ensemble']=np.mean([fold_pred[f'gnn_v2_init_{s}'] for s in [11,29,47]],axis=0)
        fold_pred['mean_curve']=np.tile(curves[train].mean(axis=0),(len(test),1))
        # Refit physical baseline on exactly the same 16 beds, not historical 19-bed predictions.
        g=data.effective_inlet_to_interior_conductance_m3_s.to_numpy()
        slope,intercept=np.polyfit(np.log(g[train]),np.log(data.effective_diffusivity_m2_s.to_numpy()[train]),1)
        physical=[]
        for j in test:
            bed=bed_model(data.iloc[j]);deff=np.exp(intercept+slope*np.log(g[j]))
            physical.append(bed.inventories(bed.solve(deff,times)).mean_loading_mol_kg.to_numpy())
        fold_pred['G_access_power_1D']=np.array(physical)
        # Original GNN refitted with its frozen settings on the same outer folds.
        old=[]
        for init in [11,29,47]:
            model,_,_=train_model([graphs[j] for j in train],curves[train],times,init,200)
            with torch.inference_mode():old.append(model(pack([graphs[j] for j in test])).numpy()@interp.T)
        fold_pred['gnn_v1_ensemble']=np.mean(old,axis=0)
        for name,prediction in fold_pred.items():
            for k,j in enumerate(test):
                pred=prediction[k];reference=curves[j]
                results.append(dict(seed=int(data.seed.iloc[j]),fold=fold,model=name,
                    curve_nrmse=float(np.sqrt(weights@((pred-reference)**2))/reference[-1]),
                    final_reference=reference[-1],final_prediction=pred[-1],
                    monotonic=bool(np.all(np.diff(pred)>=-1e-8)),bounded=bool(pred.min()>=-1e-8 and pred.max()<=QEQ+1e-8)))
                outputs.append(pd.DataFrame(dict(seed=int(data.seed.iloc[j]),model=name,time_s=times,reference=reference,prediction=pred)))
        for j in test:
            batch=pack_v2([graphs[j]])
            def infer():
                with torch.inference_mode():return np.mean([m(batch).numpy()[0]@interp.T for m in models],axis=0)
            build=timed(lambda:pack_v2([graph_v2(graphs[j]['xyz'])]),5)[0];elapsed=timed(infer,5)[0]
            timing.append(dict(seed=int(data.seed.iloc[j]),graph_build_s=build,ensemble_inference_s=elapsed,total_s=build+elapsed))
        pd.DataFrame(results).to_csv(a.output/'per_bed_metrics.csv',index=False)
        pd.DataFrame(histories).to_csv(a.output/'training_history.csv',index=False)
        pd.DataFrame(timing).to_csv(a.output/'timings.csv',index=False)
        pd.concat(outputs).to_csv(a.output/'held_out_curves.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    r=pd.DataFrame(results);summary=[]
    for name,d in r.groupby('model'):
        summary.append(dict(model=name,mean_curve_nrmse=d.curve_nrmse.mean(),worst_curve_nrmse=d.curve_nrmse.max(),
            final_mape_percent=100*np.mean(abs(d.final_prediction/d.final_reference-1)),
            final_r2=r2_score(d.final_reference,d.final_prediction)))
    s=pd.DataFrame(summary);s.to_csv(a.output/'summary.csv',index=False);print(s.to_string(index=False))


if __name__=='__main__':main()
