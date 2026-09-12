#!/usr/bin/env python3
"""Compare conductance closures and propagate held-out D_eff into uptake histories.

Run from code_work:
    python3 analyze_co2_conductance_closure.py

Inputs are existing steady conductances, fitted diffusivities, and PNM histories.
At each depth, leave one bed out and fit proportional, affine, and power-law
relationships using G_access. Porosity is a depth-independent baseline. At the
primary depth, solve the existing 1D model with each held-out prediction; no
refitting to the omitted bed's uptake curve occurs. Original outputs are kept.
This is internal LOOCV, not nested model/depth selection or external validation.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from analyze_co2_effective_diffusivity_closure import (
    GCOL, YCOL, read_homogeneous, read_accessibility, fit_predict, loo_predictions, metrics,
)
from solve_co2_homogeneous_1d import HomogeneousBed

MODELS = [('conductance_proportional', GCOL, 'origin'),
          ('conductance_linear', GCOL, 'linear'),
          ('conductance_power', GCOL, 'power'),
          ('porosity_linear', 'porosity', 'linear')]


def bed_parameters(meta):
    """Use the same physical and numerical settings as the stored per-bed fit."""
    return SimpleNamespace(cells=meta['cells'], bed_height=meta['bed_height_m'],
        porosity=meta['porosity'], tube_radius=meta['tube_radius_m'],
        particle_density=meta['particle_density_kg_m3'], temperature=meta['temperature_K'],
        qsat=meta['qsat_mol_kg'], b_pa=meta['b_Pa_inverse'],
        toth_exponent=meta['toth_exponent'], k_ldf=meta['k_ldf_s_inverse'],
        inlet_mass_transfer_coefficient=meta['inlet_mass_transfer_coefficient_m_s'],
        inlet_concentration=meta['inlet_concentration_mol_m3'],
        initial_concentration=0., initial_loading=0., t_end=meta['t_end_s'],
        rtol=meta['rtol'], atol=meta['atol'], max_step=np.inf)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('co2_conductance_closure'))
    p.add_argument('--primary-depth', type=float, default=5.)
    p.add_argument('--homogeneous-root', type=Path, default=Path('co2_homogeneous_campaign_100cells'))
    p.add_argument('--adsorption-root', type=Path, default=Path('co2_adsorption_campaign_finite_inlet_5000s'))
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    hom = read_homogeneous(a.homogeneous_root/'homogeneous_campaign_summary.csv')
    access = read_accessibility([Path(f'co2_inlet_accessibility_{d}dp.csv') for d in [3,5,7,10]])
    merged = access.merge(hom, on='seed', how='inner', validate='many_to_one')
    metric_rows, prediction_rows, correlations = [], [], []
    for depth, group in merged.groupby('depth_dp'):
        group = group.sort_values('seed').copy()
        if len(group) != len(hom) or not np.isfinite(group[[GCOL,YCOL,'porosity']]).all().all():
            raise ValueError(f'Incomplete inputs at depth {depth}')
        x,y = group[GCOL].to_numpy(),group[YCOL].to_numpy()
        correlations.append(dict(depth_dp=depth, pearson_r=pearsonr(x,y).statistic,
                                 spearman_rho=spearmanr(x,y).statistic))
        for model,xcol,kind in MODELS:
            pred=loo_predictions(kind,group[xcol].to_numpy(),y)
            if np.any(pred<=0):
                raise ValueError(f'{model} predicts non-positive D at depth {depth}; do not clip')
            metric_rows.append(dict(depth_dp=depth,model=model,**metrics(y,pred)))
            for seed,obs,val in zip(group.seed,y,pred):
                prediction_rows.append(dict(depth_dp=depth,model=model,seed=int(seed),
                    observed_D_eff_m2_s=obs,loo_predicted_D_eff_m2_s=val))
    pd.DataFrame(metric_rows).to_csv(a.output/'depth_model_metrics.csv',index=False)
    pd.DataFrame(correlations).to_csv(a.output/'depth_correlations.csv',index=False)
    predictions=pd.DataFrame(prediction_rows)
    predictions.to_csv(a.output/'loo_diffusivity_predictions.csv',index=False)
    primary=merged[np.isclose(merged.depth_dp,a.primary_depth)].sort_values('seed')
    if primary.empty: raise ValueError('Primary depth not available')
    primary[['seed',GCOL,YCOL,'porosity','bed_height_m']].to_csv(a.output/'primary_dataset.csv',index=False)
    coefficients={}
    for model,xcol,kind in MODELS:
        coefficients[model]=fit_predict(kind,primary[xcol],primary[YCOL],primary[xcol])[1]
    (a.output/'coefficients.json').write_text(json.dumps(coefficients,indent=2)+'\n')
    curves,rows=[],[]
    for row in predictions[np.isclose(predictions.depth_dp,a.primary_depth)].itertuples():
        seed=int(row.seed)
        meta=json.loads((a.homogeneous_root/f'seed_{seed}/run_metadata.json').read_text())
        ts=pd.read_csv(a.adsorption_root/f'seed_{seed}/timeseries.csv')
        t=ts.time_s.to_numpy();observed=ts.mean_loading_mol_kg.to_numpy()
        model=HomogeneousBed(bed_parameters(meta))
        sol=model.solve(row.loo_predicted_D_eff_m2_s,t)
        if not sol.success: raise RuntimeError(sol.message)
        inv=model.inventories(sol); predicted=inv.mean_loading_mol_kg.to_numpy()
        if np.any(np.diff(t)<=0) or observed[-1]<=0: raise ValueError('Invalid reference times/history')
        w=np.r_[(t[1]-t[0])/2,(t[2:]-t[:-2])/2,(t[-1]-t[-2])/2];w/=w.sum()
        err=predicted-observed
        rows.append(dict(model=row.model,seed=seed,predicted_diffusivity_m2_s=row.loo_predicted_D_eff_m2_s,
            time_weighted_nrmse=float(np.sqrt(np.sum(w*err**2))/observed[-1]),
            final_relative_error=float(err[-1]/observed[-1]),
            max_absolute_mass_residual_mol=float(abs(inv.mass_balance_residual_mol).max())))
        curves.append(pd.DataFrame(dict(model=row.model,seed=seed,time_s=t,
            reference_loading_mol_kg=observed,predicted_loading_mol_kg=predicted)))
        print(row.model,seed,f"curve NRMSE={rows[-1]['time_weighted_nrmse']:.5f}",flush=True)
    df=pd.DataFrame(rows);df.to_csv(a.output/'curve_metrics.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(a.output/'held_out_curves.csv.gz',index=False,
        compression={'method':'gzip','mtime':0})
    summary=df.groupby('model').agg(mean_curve_nrmse=('time_weighted_nrmse','mean'),
        min_curve_nrmse=('time_weighted_nrmse','min'),max_curve_nrmse=('time_weighted_nrmse','max'),
        max_mass_residual_mol=('max_absolute_mass_residual_mol','max')).reset_index()
    metric_df=pd.DataFrame(metric_rows)
    summary=summary.merge(metric_df[np.isclose(metric_df.depth_dp,a.primary_depth)][['model','loo_r2']],on='model')
    summary.to_csv(a.output/'model_summary.csv',index=False)
    (a.output/'run_metadata.json').write_text(json.dumps(dict(primary_depth_dp=a.primary_depth,
        n_beds=len(primary),models=MODELS,curve_error='trapezoidal time-weighted RMSE / final PNM uptake',
        selection='model/depth compared within same ensemble; not nested validation',
        initial_gas_concentration=0,initial_solid_loading=0),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
