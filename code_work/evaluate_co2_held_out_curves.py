#!/usr/bin/env python3
"""Propagate stored leave-one-bed-out power-law D predictions through the 1D model.
This evaluates a fixed, previously selected model; it is not nested model selection.
Run from code_work; original adsorption and fit outputs are never overwritten.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from solve_co2_homogeneous_1d import HomogeneousBed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('pnm_review/held_out_curves'))
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    predictions=pd.read_csv('co2_effective_diffusivity_closure/closure_loo_predictions.csv')
    predictions=predictions[predictions.model=='accessibility_power_law']
    rows=[]
    for row in predictions.itertuples():
        seed=int(row.seed)
        qa=json.loads(Path(f'co2_pore_networks_power22/seed_{seed}/network_qa.json').read_text())
        ts=pd.read_csv(f'co2_adsorption_campaign_finite_inlet_5000s/seed_{seed}/timeseries.csv')
        params=SimpleNamespace(cells=100,bed_height=qa['bed_top_m'],porosity=qa['analytic_porosity'],
            tube_radius=.008,particle_density=1600.,temperature=298.15,qsat=5.332,b_pa=5.093e-5,
            toth_exponent=1.,k_ldf=8.55e-3,inlet_mass_transfer_coefficient=None,
            boundary_diffusivity=1.5e-5,inlet_transfer_length=.0008,inlet_concentration=6.05,
            initial_concentration=0.,initial_loading=0.,t_end=5000.,rtol=1e-6,atol=1e-10,max_step=np.inf)
        model=HomogeneousBed(params); sol=model.solve(row.loo_predicted_D_eff_m2_s,ts.time_s.to_numpy())
        inv=model.inventories(sol); predicted=inv.mean_loading_mol_kg.to_numpy(); observed=ts.mean_loading_mol_kg.to_numpy()
        t=ts.time_s.to_numpy(); w=np.r_[(t[1]-t[0])/2,(t[2:]-t[:-2])/2,(t[-1]-t[-2])/2]; w/=w.sum()
        error=predicted-observed
        rows.append(dict(seed=seed,predicted_diffusivity_m2_s=row.loo_predicted_D_eff_m2_s,
            time_weighted_nrmse=np.sqrt(np.sum(w*error**2))/observed[-1],
            final_relative_error=error[-1]/observed[-1],
            max_absolute_mass_residual_mol=abs(inv.mass_balance_residual_mol).max()))
        pd.DataFrame(dict(time_s=t,reference_loading=observed,predicted_loading=predicted)).to_csv(a.output/f'seed_{seed}.csv',index=False)
        pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
        print(seed,rows[-1]['time_weighted_nrmse'],flush=True)
    df=pd.DataFrame(rows)
    print('Mean/min/max curve NRMSE:',df.time_weighted_nrmse.mean(),df.time_weighted_nrmse.min(),df.time_weighted_nrmse.max())

if __name__=='__main__': main()
