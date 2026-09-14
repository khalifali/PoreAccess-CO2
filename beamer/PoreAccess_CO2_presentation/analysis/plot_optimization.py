#!/usr/bin/env python3
"""Reproduce slide 15's actual optimization trials. Run: python beamer/PoreAccess_CO2_presentation/analysis/plot_optimization.py (from repo root)."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code_work'))
from solve_co2_homogeneous_1d import HomogeneousBed
case = ROOT / 'code_work/co2_homogeneous_campaign_100cells/seed_18427'
m = json.loads((case/'run_metadata.json').read_text())
a = SimpleNamespace(cells=m['cells'], bed_height=m['bed_height_m'], porosity=m['porosity'], tube_radius=m['tube_radius_m'], particle_density=m['particle_density_kg_m3'], inlet_mass_transfer_coefficient=m['inlet_mass_transfer_coefficient_m_s'], temperature=m['temperature_K'], b_pa=m['b_Pa_inverse'], toth_exponent=m['toth_exponent'], qsat=m['qsat_mol_kg'], k_ldf=m['k_ldf_s_inverse'], inlet_concentration=m['inlet_concentration_mol_m3'], initial_concentration=0., initial_loading=0., t_end=m['t_end_s'], rtol=m['rtol'], atol=m['atol'], max_step=np.inf)
model = HomogeneousBed(a)
ref = pd.read_csv(case/'fit_comparison.csv')
trace = pd.read_csv(case/'fit_trace.csv')
t = ref.time_s.to_numpy()
y = ref.reference_loading_mol_kg.to_numpy()
w = np.r_[.5*(t[1]-t[0]), .5*(t[2:]-t[:-2]), .5*(t[-1]-t[-2])]; w /= w.sum()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(6.4,4.5))
ax.plot(t, y, color='black', lw=2.6, label='PNM reference', zorder=5)
out = pd.DataFrame({'time_s':t, 'pnm_loading_mol_kg':y})
for idx, color, style in [(0,'#969696','--'),(1,'#e38a20',':'),(3,'#a85a91','-.'),(7,'#0065bd','--')]:
 d = trace.iloc[idx].effective_diffusivity_m2_s
 sol = model.solve(d,t)
 q = sol.y[model.n:2*model.n].mean(axis=0)
 error = np.sqrt(np.sum(w*(q-y)**2))/y[-1]
 assert abs(error-trace.iloc[idx].normalized_rmse)<1e-5
 out[f'trial_{idx+1}_loading_mol_kg'] = q
 name = 'Best fit (8)' if idx==7 else f'Trial {idx+1}'
 ax.plot(t,q,color=color,ls=style,lw=2.3,label=f'{name}: D={d/1e-7:.2f}, error={error:.1%}',zorder=6 if idx==7 else 3)
ax.set(xlabel='Time [s]',ylabel=r'Mean uptake $\overline{q}$ [mol kg$^{-1}$]',xlim=(0,5000),ylim=(0,.34))
ax.grid(alpha=.15)
ax.legend(loc='upper left',frameon=False,fontsize=8.8,title=r'$D$ in $10^{-7}$ m$^2$ s$^{-1}$; error = NRMSE',title_fontsize=9)
fig.tight_layout(pad=.6)
folder=ROOT/'beamer/PoreAccess_CO2_presentation/figures'
fig.savefig(folder/'fig_optimization_trials.pdf')
fig.savefig(folder/'fig_optimization_trials.png',dpi=180)
out.to_csv(Path(__file__).parent/'optimization_trial_curves.csv',index=False)
print('Verified four re-solved trial errors against the recorded optimization trace.')
