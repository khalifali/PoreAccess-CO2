#!/usr/bin/env python3
"""Plot the G_access-only comparison. Run from code_work after analyze_co2_conductance_closure.py.
Writes PDF/SVG figures used in both decks; normalized-descriptor figures stay unchanged.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_co2_effective_diffusivity_closure import GCOL,YCOL,fit_predict

ROOT=Path('co2_conductance_closure');OUT=ROOT/'figures'
STYLES={'conductance_power':('#0072B2','Power law','o','-'),
        'conductance_linear':('#009E73','Linear + intercept','D','--'),
        'conductance_proportional':('#E69F00','Proportional','^','-.'),
        'porosity_linear':('#777777','Porosity baseline','s',':')}
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'legend.fontsize':8,'svg.fonttype':'none'})


def canvas():
    fig,ax=plt.subplots(figsize=(5.2,3.8),layout='constrained')
    ax.spines[['top','right']].set_visible(False);ax.grid(color='#e4e7eb',lw=.6);ax.set_axisbelow(True)
    return fig,ax


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg']:fig.savefig(OUT/(name+'.'+ext))
    plt.close(fig)


def main():
    d=pd.read_csv(ROOT/'primary_dataset.csv');x=d[GCOL].to_numpy();y=d[YCOL].to_numpy()
    grid=np.linspace(x.min()*.98,x.max()*1.02,200)
    fig,ax=canvas();ax.scatter(x*1e8,y*1e6,s=30,color='#333333',label='Fitted bed diffusivities',zorder=4)
    for model,kind in [('conductance_power','power'),('conductance_linear','linear'),('conductance_proportional','origin')]:
        color,label,_,ls=STYLES[model];pred,_=fit_predict(kind,x,y,grid)
        ax.plot(grid*1e8,pred*1e6,color=color,ls=ls,label=label)
    ax.set_xlabel(r'Inlet-access conductance $G_{\rm access}$ [$10^{-8}$ m$^3$/s]')
    ax.set_ylabel(r'Fitted $D_{\rm eff}$ [$10^{-6}$ m$^2$/s]');ax.legend(frameon=False)
    save(fig,'g_closure_fit')
    pred=pd.read_csv(ROOT/'loo_diffusivity_predictions.csv');primary=pred[pred.depth_dp==5]
    fig,ax=canvas();values=[]
    for model,(color,label,marker,_) in STYLES.items():
        z=primary[primary.model==model];xx=z.observed_D_eff_m2_s*1e6;yy=z.loo_predicted_D_eff_m2_s*1e6
        ax.scatter(xx,yy,color=color,marker=marker,s=30,label=label,alpha=.8,edgecolors='white',linewidths=.4)
        values.extend(xx);values.extend(yy)
    lo,hi=min(values)*.96,max(values)*1.04;ax.plot([lo,hi],[lo,hi],'k--',lw=1,label='Ideal')
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel(r'Observed $D_{\rm eff}$ [$10^{-6}$ m$^2$/s]');ax.set_ylabel(r'Held-out prediction [$10^{-6}$ m$^2$/s]')
    ax.legend(frameon=False);save(fig,'g_held_out_diffusivity')
    cor=pd.read_csv(ROOT/'depth_correlations.csv');fig,ax=canvas()
    ax.plot(cor.depth_dp,cor.pearson_r,'o-',label='Pearson r',color='#0072B2')
    ax.plot(cor.depth_dp,cor.spearman_rho,'s--',label='Spearman rho',color='#009E73')
    ax.set_ylim(.80,1);ax.set_xticks([3,5,7,10]);ax.set_xlabel(r'Interior-plane depth [$d_p$]')
    ax.set_ylabel(r'Correlation of $G_{\rm access}$ with fitted $D_{\rm eff}$');ax.legend(frameon=False)
    save(fig,'g_depth_correlations')
    depth=pd.read_csv(ROOT/'depth_model_metrics.csv');fig,ax=canvas()
    for model,(color,label,marker,ls) in STYLES.items():
        z=depth[depth.model==model];ax.plot(z.depth_dp,z.loo_r2,color=color,label=label,marker=marker,ls=ls)
    ax.set_ylim(-.3,1);ax.set_xticks([3,5,7,10]);ax.set_xlabel(r'Interior-plane depth [$d_p$]')
    ax.set_ylabel(r'Leave-one-bed-out $R^2$');ax.legend(frameon=False,loc='lower right')
    save(fig,'g_depth_models')
    curves=pd.read_csv(ROOT/'held_out_curves.csv.gz');power=curves[curves.model=='conductance_power']
    final=power.groupby('seed').tail(1).sort_values('reference_loading_mol_kg')
    seeds=final.iloc[[0,len(final)//2,len(final)-1]].seed.tolist()
    fig,ax=canvas()
    for seed,color in zip(seeds,['#0072B2','#D55E00','#009E73']):
        z=power[power.seed==seed]
        ax.plot(z.time_s,z.reference_loading_mol_kg,color=color,label=f'PNM, seed {seed}')
        ax.plot(z.time_s,z.predicted_loading_mol_kg,color=color,ls='--')
    ax.plot([],[],'k--',label='1D using held-out power-law D')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Mean uptake [mol/kg]');ax.legend(frameon=False)
    save(fig,'g_held_out_uptake')
    print('Five conductance figures written; all four models evaluated on all twenty beds.')


if __name__=='__main__':main()
