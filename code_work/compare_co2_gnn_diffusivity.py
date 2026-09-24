#!/usr/bin/env python3
"""Particle-only GNN prediction of log effective diffusivity, nested bed validation."""
from compare_co2_gnn_v2 import *
import hashlib

class DiffusivityGNN(RegionalGNN):
    def __init__(self,center,scale):
        super().__init__(np.linspace(0,QEQ/2,len(KNOTS)))
        self.head[-1]=nn.Linear(32,1)
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('center',torch.tensor(float(center)))
        self.register_buffer('scale',torch.tensor(float(scale)))

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
        return self.center+self.scale*self.head(torch.cat(pools,dim=1)).squeeze(-1)

def train(graphs,y,seed,wd,epochs,validation=None):
    torch.manual_seed(seed);center=y.mean();scale=max(y.std(),1e-6)
    model=DiffusivityGNN(center,scale);batch=pack_v2(graphs);target=torch.tensor(y,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=wd)
    if validation is not None: vb=pack_v2(validation[0]);vy=validation[1]
    best=np.inf;best_epoch=0;state=None;hist=[]
    for epoch in range(1,epochs+1):
        opt.zero_grad();loss=(((model(batch)-target)/scale)**2).mean();loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if epoch%10==0 or epoch==epochs:
            with torch.inference_mode():
                tr=float(np.sqrt(np.mean((model(batch).numpy()-y)**2)))
                score=float(np.sqrt(np.mean((model(vb).numpy()-vy)**2))) if validation is not None else tr
            hist.append(dict(epoch=epoch,train_log_rmse=tr,validation_log_rmse=score if validation is not None else np.nan))
            if score<best-1e-6:best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
            if validation is not None and epoch-best_epoch>=100:break
    if validation is not None:model.load_state_dict(state)
    return model,best_epoch,best,hist

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);threadpool_limits(limits=1)
    torch.use_deterministic_algorithms(True)
    out=Path('co2_gnn_diffusivity');out.mkdir(exist_ok=True)
    data=pd.read_csv('co2_effective_diffusivity_closure/closure_dataset.csv').sort_values('seed').reset_index(drop=True)
    assert len(data)==data.seed.nunique()==20
    y=np.log(data.effective_diffusivity_m2_s.to_numpy());g=data.effective_inlet_to_interior_conductance_m3_s.to_numpy()
    graphs=[];nets=[];hashes={}
    for seed in data.seed:
        p=Path(f'co2_pore_networks_power22/seed_{seed}/pore_network.npz')
        with np.load(p) as z:nets.append({k:z[k] for k in z.files})
        graphs.append(graph_v2(nets[-1]['particle_xyz']));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[];selections=[];history=[];timings=[]
    for fold,(train_idx,test) in enumerate(KFold(5,shuffle=True,random_state=20260924).split(data)):
        shuffled=np.random.default_rng(900+fold).permutation(train_idx);iv=shuffled[:4];it=shuffled[4:];candidates=[]
        for wd in [1e-5,1e-4]:
            model,epoch,score,h=train([graphs[j] for j in it],y[it],11,wd,1500,([graphs[j] for j in iv],y[iv]))
            candidates.append((score,wd,epoch));selections.append(dict(fold=fold,weight_decay=wd,epoch=epoch,log_rmse=score,train_seeds=' '.join(map(str,data.seed.iloc[it])),validation_seeds=' '.join(map(str,data.seed.iloc[iv])),test_seeds=' '.join(map(str,data.seed.iloc[test]))))
            history.extend(dict(fold=fold,stage='inner',weight_decay=wd,initialization=11,**x) for x in h)
            print(f'fold {fold+1}: wd={wd}, epoch={epoch}, validation log RMSE={score:.4f}',flush=True)
        _,wd,epochs=min(candidates);models=[];pred=[]
        for init in [11,29,47]:
            model,_,_,h=train([graphs[j] for j in train_idx],y[train_idx],init,wd,epochs);models.append(model)
            with torch.inference_mode():pred.append(model(pack_v2([graphs[j] for j in test])).numpy())
            history.extend(dict(fold=fold,stage='refit',weight_decay=wd,initialization=init,**x) for x in h)
        slope,intercept=np.polyfit(np.log(g[train_idx]),y[train_idx],1)
        predictions={'gnn':np.exp(np.mean(pred,axis=0)),'G_access':np.exp(intercept+slope*np.log(g[test])),
                     'mean_diffusivity':np.full(len(test),np.exp(y[train_idx]).mean()),'geometric_mean':np.full(len(test),np.exp(y[train_idx].mean()))}
        for name,values in predictions.items():
            for j,v in zip(test,values):rows.append(dict(seed=int(data.seed.iloc[j]),fold=fold,model=name,reference=np.exp(y[j]),prediction=float(v)))
        for j in test:
            batch=pack_v2([graphs[j]])
            def infer():
                with torch.inference_mode():return np.exp(np.mean([m(batch).item() for m in models]))
            np.testing.assert_allclose(access_solve(nets[j]),g[j],rtol=1e-8)
            build=timed(lambda:pack_v2([graph_v2(nets[j]['particle_xyz'])]),5)[0]
            inf=timed(infer,5)[0];physical=timed(lambda:np.exp(intercept+slope*np.log(access_solve(nets[j]))),5)[0]
            timings.append(dict(seed=int(data.seed.iloc[j]),graph_s=build,inference_s=inf,gnn_total_s=build+inf,G_access_s=physical))
        pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);pd.DataFrame(selections).to_csv(out/'selection.csv',index=False)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False);pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    metrics=[]
    for name,d in pd.DataFrame(rows).groupby('model'):
        metrics.append(dict(model=name,mape_percent=100*np.mean(abs(d.prediction/d.reference-1)),r2=r2_score(d.reference,d.prediction),log_rmse=np.sqrt(np.mean(np.log(d.prediction/d.reference)**2))))
    pd.DataFrame(metrics).to_csv(out/'summary.csv',index=False)
    hashes['co2_effective_diffusivity_closure/closure_dataset.csv']=hashlib.sha256(Path('co2_effective_diffusivity_closure/closure_dataset.csv').read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(torch=str(torch.__version__),numpy=np.__version__,inputs=hashes,parameters=sum(p.numel() for p in models[0].parameters())),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':main()
