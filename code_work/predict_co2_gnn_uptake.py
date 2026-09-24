#!/usr/bin/env python3
"""Predict a 0–5000 s uptake curve directly from a DEM particle snapshot.

Use the comparison's example checkpoint only as a held-out demonstration.
It was trained on 19 beds, at one fixed set of physical/operating conditions.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from compare_co2_gnn_uptake import CurveGNN,particle_graph,pack,KNOTS,DP
from extract_co2_pore_network import read_last_snapshot,particle_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--particle-dump',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('co2_gnn_curve_comparison/example_held_out_model.pt'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    np.testing.assert_allclose(checkpoint['knots_s'],KNOTS)
    columns,values,_=read_last_snapshot(a.particle_dump)
    _,xyz,radii=particle_data(columns,values)
    if not np.allclose(radii,DP/2,rtol=1e-6,atol=0):
        raise ValueError('Checkpoint only supports the campaign bead diameter')
    model=CurveGNN(checkpoint['mean_knots']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.inference_mode():knots=model(pack([particle_graph(xyz)])).numpy()[0]
    times=np.linspace(0,5000,301)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(time_s=times,predicted_uptake_mol_kg=np.interp(times,KNOTS,knots))).to_csv(a.output,index=False)
    print(a.output)


if __name__=='__main__':main()
