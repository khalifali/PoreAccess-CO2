#!/usr/bin/env python3
"""Audit production networks without modifying them. Run from code_work."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from solve_co2_pore_adsorption import load_network, incidence_weights, graph_laplacian


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--network-root',type=Path,default=Path('co2_pore_networks_power22'))
    p.add_argument('--output',type=Path,default=Path('pnm_review/network_audit.csv'))
    a=p.parse_args(); rows=[]
    for path in sorted(a.network_root.glob('seed_*')):
        n,ip,iv=load_network(path); qa=json.loads((path/'network_qa.json').read_text())
        v=n['pore_volume']; xyz=n['pore_xyz']; edges=n['throat_conns']; area=n['throat_area']; length=n['throat_length']
        W=incidence_weights(len(n['particle_id']),len(v),ip,iv,v)
        g=1.5e-5*area/length; L=graph_laplacian(len(v),edges,g)
        t=pd.read_csv(path/'throats.csv'); f=t[['x_m','y_m','z_m']].to_numpy()
        start=xyz[edges[:,0]]; delta=xyz[edges[:,1]]-start
        u=np.sum((f-start)*delta,axis=1)/np.sum(delta**2,axis=1)
        outside=(u < -1e-8)|(u>1+1e-8)
        # Sample centreline clearance as a diagnostic, not an exact throat-area model.
        tree=cKDTree(n['particle_xyz']); minclear=np.full(len(edges),np.inf)
        for fraction in np.linspace(0,1,17):
            points=start+fraction*delta
            d,idx=tree.query(points)
            minclear=np.minimum(minclear,d-n['particle_radius'][idx])
        row=dict(seed=qa['seed'],pores=len(v),throats=len(edges),
                 qa_pass=qa['network_ready_for_transport'],
                 volume_relative_error=abs(v.sum()-qa['void_volume_m3'])/qa['void_volume_m3'],
                 incidence_row_error=np.max(abs(np.asarray(W.sum(axis=1)).ravel()-1)),
                 laplacian_relative_row_error=np.max(abs(L@np.ones(len(v))))/np.max(L.diagonal()),
                 face_centre_outside_segment_count=int(outside.sum()),
                 face_centre_outside_segment_fraction=outside.mean(),
                 sampled_solid_crossing_count=int(np.sum(minclear < -2e-6)),
                 sampled_clearance_below_assigned_radius_count=int(np.sum(minclear < n['throat_radius']-2e-6)),
                 lowest_pore_z_m=xyz[:,2].min(),inlet_pore_max_z_m=xyz[n['pore_inlet'],2].max(),
                 inlet_pores=int(n['pore_inlet'].sum()))
        rows.append(row)
    if not rows: raise SystemExit('No networks found')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows).sort_values('seed'); df.to_csv(a.output,index=False)
    print(df.to_string(index=False))
    if not df.qa_pass.all() or df.volume_relative_error.max()>1e-12 or df.incidence_row_error.max()>1e-12:
        raise SystemExit('Network QA failed')

if __name__=='__main__': main()
