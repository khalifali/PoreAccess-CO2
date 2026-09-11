#!/usr/bin/env python3
"""Implicit pore-network CO2 diffusion with optional particle LDF adsorption.

Gas concentration c_p [mol/m3] is stored on pore nodes. Particle loading q_i
[mol/kg] is stored on zeolite beads. The conservative equations are

  V_p dc_p/dt = sum_j D A_pj/L_pj (c_j-c_p)
                - sum_i W_ip m_i dq_i/dt

  dq_i/dt = k_LDF (q*_i - q_i)

The displayed fixed-weight gas sink describes desorption. During positive
uptake the conservative sink weights are W_ip*max(c_p,0)/(W*max(c,0))_i,
so an empty pore cannot supply adsorption to its neighbour. W also maps pore
concentration to particle concentration. W rows sum to one. The Toth isotherm
is

  q* = q_s bP / [1 + (bP)^t]^(1/t),  P = c_particle R T.

Set t=1 for Langmuir. Two inlet formulations are available. ``fixed`` keeps
the original implementation in which bottom-labelled pores are fixed at the
inlet concentration. ``finite`` connects those pores to an external reservoir
through finite diffusive conductances. In finite mode the assigned boundary
areas always sum to the physical tube cross-section, so adding inlet pores does
not add boundary area or create extra infinite-strength sources. Wall and top
labels are no-flux. An extra integrated state records the reservoir input,
enabling an independent inventory-conservation check.

Run diffusion first, then adsorption. See --help and the generated metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csr_matrix, eye, bmat, diags

RGAS = 8.31446261815324  # J mol-1 K-1


def load_network(directory: Path):
    npz_path = directory / "pore_network.npz"
    incidence_path = directory / "particle_pore_incidence.csv"
    if not npz_path.exists() or not incidence_path.exists():
        raise FileNotFoundError(f"Missing network NPZ or incidence CSV below {directory}")
    z = np.load(npz_path)
    required = ("pore_xyz","pore_volume","pore_inlet","throat_conns",
                "throat_area","throat_length","particle_id","particle_radius")
    missing = [k for k in required if k not in z]
    if missing: raise ValueError(f"NPZ lacks {missing}")
    with incidence_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    inc_pore = np.asarray([int(r["pore_id"]) for r in rows], dtype=int)
    inc_particle = np.asarray([int(r["particle_index"]) for r in rows], dtype=int)
    return {k: np.asarray(z[k]) for k in z.files}, inc_particle, inc_pore


def incidence_weights(npart, npore, inc_particle, inc_pore, pore_volume):
    """Particle-normalized weights proportional to adjacent pore volume."""
    raw = pore_volume[inc_pore]
    denom = np.bincount(inc_particle, weights=raw, minlength=npart)
    if np.any(denom <= 0):
        bad = np.flatnonzero(denom <= 0)
        raise ValueError(f"Particles without positive incident pore volume: {bad[:10]}")
    values = raw / denom[inc_particle]
    # W shape: particles x pores; each row sums to one.
    W = coo_matrix((values, (inc_particle, inc_pore)), shape=(npart, npore)).tocsr()
    err = np.max(np.abs(np.asarray(W.sum(axis=1)).ravel()-1))
    if err > 1e-12: raise ValueError(f"Incidence weights do not normalize: {err}")
    return W


def graph_laplacian(npore, conns, conductance):
    """Return L with transport vector -L*c [mol/s]."""
    a, b = conns[:,0].astype(int), conns[:,1].astype(int)
    rows = np.r_[a,b,a,b]
    cols = np.r_[a,b,b,a]
    vals = np.r_[conductance,conductance,-conductance,-conductance]
    return coo_matrix((vals,(rows,cols)),shape=(npore,npore)).tocsr()


def toth_equilibrium(c_particle, temperature, qsat, b_pa, exponent):
    pressure = np.maximum(c_particle, 0)*RGAS*temperature
    bp = np.maximum(b_pa*pressure, 0)
    return qsat*bp/np.power(1+np.power(bp, exponent), 1/exponent)


def toth_derivative(c_particle, temperature, qsat, b_pa, exponent):
    """Analytic dq*/dc for the Toth equation [m3/kg]."""
    positive = c_particle >= 0
    x = np.maximum(b_pa*RGAS*temperature*c_particle, 0)
    derivative = qsat*b_pa*RGAS*temperature*np.power(
        1+np.power(x, exponent), -1/exponent-1
    )
    return np.where(positive, derivative, 0.0)


def output_times(t_end, count, logarithmic):
    if logarithmic:
        positive = np.geomspace(max(t_end*1e-8,1e-10),t_end,max(count-1,1))
        return np.unique(np.r_[0.,positive])
    return np.linspace(0,t_end,count)


def jacobian_pattern(L, free, W, adsorption):
    Lff = L[free][:,free].astype(bool).astype(int) + eye(len(free),format="csr")
    if not adsorption:
        # Final state is cumulative inlet; it depends on all free c.
        return bmat([[Lff, csr_matrix((len(free),1))],
                     [csr_matrix(np.ones((1,len(free)))),csr_matrix((1,1))]],format="csr")
    Wf = W[:,free].astype(bool).astype(int)
    # Gas adsorption coupling is Wf.T*Wf in c and Wf.T in q.
    cc = (Lff + Wf.T@Wf).astype(bool).astype(int)
    cq = Wf.T
    qc = Wf
    qq = eye(W.shape[0],format="csr")
    zc = csr_matrix((len(free),1)); zq = csr_matrix((W.shape[0],1))
    last_c = csr_matrix(np.ones((1,len(free))))
    last_q = csr_matrix(np.ones((1,W.shape[0])))
    return bmat([[cc,cq,zc],[qc,qq,zq],[last_c,last_q,csr_matrix((1,1))]],format="csr")


def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--network",type=Path,required=True,help="seed_* network directory")
    p.add_argument("--output",type=Path,default=Path("co2_transport_result"))
    p.add_argument("--mode",choices=("diffusion","adsorption"),default="diffusion")
    p.add_argument("--diffusivity",type=float,required=True,help="throat diffusivity [m2/s]")
    p.add_argument("--inlet-concentration",type=float,required=True,help="fixed bottom concentration [mol/m3]")
    p.add_argument("--initial-concentration",type=float,default=0.,help="initial internal gas concentration [mol/m3]")
    p.add_argument("--temperature",type=float,default=298.15,help="isothermal temperature [K]")
    p.add_argument("--particle-density",type=float,default=1600.,help="apparent bead density [kg/m3]")
    p.add_argument("--qsat",type=float,help="Toth saturation loading [mol/kg]")
    p.add_argument("--b-pa",type=float,help="Toth affinity [1/Pa]")
    p.add_argument("--toth-exponent",type=float,default=1.,help="Toth exponent; 1=Langmuir")
    p.add_argument("--k-ldf",type=float,help="LDF coefficient [1/s]")
    p.add_argument("--initial-loading",type=float,default=0.,help="uniform initial q [mol/kg]")
    p.add_argument("--inlet-boundary",choices=("fixed","finite"),default="fixed",
                   help="fixed pore concentration (legacy) or finite reservoir conductance")
    p.add_argument("--tube-radius",type=float,default=8e-3,
                   help="tube radius used to define total inlet area [m]")
    p.add_argument("--bed-bottom",type=float,default=0.,help="physical inlet plane z [m]")
    p.add_argument("--inlet-min-distance-dp",type=float,default=.5,
                   help="minimum reservoir-to-pore length divided by particle diameter")
    p.add_argument("--inlet-area-weighting",choices=("equal","pore-radius-squared"),default="equal",
                   help="distribution of the fixed total inlet area among labelled pores")
    p.add_argument("--t-end",type=float,default=500.,help="simulation duration [s]")
    p.add_argument("--outputs",type=int,default=201)
    p.add_argument("--linear-output-times",action="store_true")
    p.add_argument("--rtol",type=float,default=1e-6); p.add_argument("--atol",type=float,default=1e-10)
    p.add_argument("--max-step",type=float,default=np.inf)
    a=p.parse_args()
    adsorption=a.mode=="adsorption"
    if a.diffusivity<=0 or a.inlet_concentration<0 or a.t_end<=0: p.error("invalid transport input")
    if a.tube_radius<=0 or a.inlet_min_distance_dp<=0: p.error("invalid finite-inlet geometry")
    if (not all(np.isfinite(x) for x in (a.diffusivity, a.inlet_concentration,
            a.initial_concentration, a.initial_loading, a.temperature, a.particle_density,
            a.t_end, a.rtol, a.atol)) or a.initial_concentration < 0 or a.initial_loading < 0
            or a.temperature <= 0 or a.particle_density <= 0 or a.outputs < 2
            or a.rtol <= 0 or a.atol <= 0):
        p.error("Invalid initial state, physical parameters, output count, or tolerances")
    if adsorption:
        absent=[x for x in ("qsat","b_pa","k_ldf") if getattr(a,x) is None]
        if absent: p.error("adsorption requires --qsat, --b-pa and --k-ldf")
        if a.qsat<=0 or a.b_pa<=0 or a.k_ldf<=0 or a.toth_exponent<=0: p.error("isotherm/LDF parameters must be positive")

    net, inc_particle, inc_pore=load_network(a.network)
    vol=net["pore_volume"].astype(float); inlet=net["pore_inlet"].astype(bool)
    conns=net["throat_conns"].astype(int); area=net["throat_area"].astype(float); length=net["throat_length"].astype(float)
    if not inlet.any(): sys.exit("Network contains no inlet pores")
    if any(not np.all(np.isfinite(x)) or np.any(x<=0) for x in (vol,area,length)):
        sys.exit("Network geometry must be finite and positive")
    npore=len(vol); npart=len(net["particle_id"])
    finite_inlet=a.inlet_boundary=="finite"
    free=np.arange(npore,dtype=int) if finite_inlet else np.flatnonzero(~inlet)
    fixed=np.empty(0,dtype=int) if finite_inlet else np.flatnonzero(inlet)
    conductance=a.diffusivity*area/length
    L=graph_laplacian(npore,conns,conductance)
    W=incidence_weights(npart,npore,inc_particle,inc_pore,vol)
    radii=net["particle_radius"].astype(float)
    mass=a.particle_density*(4/3)*math.pi*radii**3

    # Finite reservoir-to-pore conductances.  The individual boundary areas
    # always sum to the physical tube cross-section, independently of how many
    # pores carry the inlet label.
    boundary_g=np.zeros(npore)
    boundary_area=np.zeros(npore)
    boundary_distance=np.zeros(npore)
    if finite_inlet:
        inlet_ids=np.flatnonzero(inlet)
        total_inlet_area=math.pi*a.tube_radius**2
        if a.inlet_area_weighting=="equal":
            weights=np.ones(len(inlet_ids))
        else:
            if "pore_inscribed_radius" not in net:
                p.error("pore-radius-squared weighting requires pore_inscribed_radius")
            weights=np.maximum(net["pore_inscribed_radius"][inlet_ids].astype(float),0.)**2
            if not np.any(weights>0):
                p.error("all inlet pore radii are zero")
        boundary_area[inlet_ids]=total_inlet_area*weights/weights.sum()
        dp=2.*float(np.median(radii))
        minimum_distance=a.inlet_min_distance_dp*dp
        if "pore_xyz" not in net:
            p.error("finite inlet requires pore_xyz")
        axial_distance=net["pore_xyz"][inlet_ids,2].astype(float)-a.bed_bottom
        boundary_distance[inlet_ids]=np.maximum(axial_distance,minimum_distance)
        boundary_g[inlet_ids]=a.diffusivity*boundary_area[inlet_ids]/boundary_distance[inlet_ids]
        if not np.all(np.isfinite(boundary_g[inlet_ids])) or np.any(boundary_g[inlet_ids]<=0):
            p.error("invalid finite inlet conductance")
    else:
        total_inlet_area=math.pi*a.tube_radius**2

    c0=np.full(npore,a.initial_concentration)
    if not finite_inlet:
        c0[inlet]=a.inlet_concentration
    q0=np.full(npart,a.initial_loading)
    inventory0=float(np.dot(vol,c0)+(np.dot(mass,q0) if adsorption else 0))
    if adsorption: y0=np.r_[c0[free],q0,0.]
    else: y0=np.r_[c0[free],0.]

    def unpack(y):
        c=np.empty(npore)
        if not finite_inlet: c[inlet]=a.inlet_concentration
        c[free]=y[:len(free)]
        if adsorption: return c,y[len(free):len(free)+npart]
        return c,None

    def adsorption_terms(c,q,with_jacobian=False):
        """Return conservative, positivity-preserving particle uptake terms.

        Positive uptake from a particle is drawn from its incident pores in
        proportion to W_ip*max(c_p,0). Desorption uses the fixed W_ip weights.
        Thus an empty pore cannot be depleted by a neighbouring particle merely
        because another incident pore has positive concentration.
        """
        cpos=np.maximum(c,0.0)
        active=(c>=0).astype(float)
        cp=W@cpos
        qeq=toth_equilibrium(cp,a.temperature,a.qsat,a.b_pa,a.toth_exponent)
        dq=a.k_ldf*(qeq-q)
        up=(dq>0)&(cp>0)
        down=~up
        h=np.zeros(npart)
        h[up]=mass[up]*dq[up]/cp[up]
        sink=cpos*(W.T@h)+W.T@(mass*dq*down)
        if not with_jacobian:
            return dq,np.asarray(sink).ravel()

        fprime=toth_derivative(cp,a.temperature,a.qsat,a.b_pa,a.toth_exponent)
        Wfd=Wf@diags(active[free])
        hprime=np.zeros(npart)
        hprime[up]=mass[up]*a.k_ldf*(fprime[up]*cp[up]-(qeq[up]-q[up]))/(cp[up]**2)
        selection=coo_matrix((active[free],(free,np.arange(len(free)))),shape=(npore,len(free))).tocsr()
        js_c=(diags(np.asarray(W.T@h).ravel())@selection
              +diags(cpos)@W.T@diags(hprime)@Wfd
              +W.T@diags(mass*a.k_ldf*fprime*down)@Wfd).tocsr()
        qcoef=np.zeros(npart)
        qcoef[up]=-mass[up]*a.k_ldf/cp[up]
        js_q=(diags(cpos)@W.T@diags(qcoef)
              +W.T@diags(-mass*a.k_ldf*down)).tocsr()
        jq_c=(diags(a.k_ldf*fprime)@Wfd).tocsr()
        return dq,np.asarray(sink).ravel(),js_c,js_q,jq_c

    def rhs(t,y):
        c,q=unpack(y)
        transport=-(L@c)  # mol/s per pore
        if finite_inlet:
            boundary_flux=boundary_g*(a.inlet_concentration-c)
            transport=transport+boundary_flux
        if adsorption:
            dq,sink=adsorption_terms(c,q)
        else:
            dq=None; sink=np.zeros(npore)
        dc_free=(transport[free]-sink[free])/vol[free]
        if finite_inlet:
            # Positive when material enters the simulated pore volume.
            reservoir_rate=float(np.sum(boundary_flux))
        else:
            # Reservoir source needed to hold inlet-node concentration fixed.
            reservoir_rate=float(np.sum(sink[fixed]-transport[fixed]))
        return np.r_[dc_free,dq,reservoir_rate] if adsorption else np.r_[dc_free,reservoir_rate]

    # Precompute fixed sparse blocks used by the exact Jacobian. Supplying the
    # analytic Jacobian avoids overflow in SciPy's finite-difference scaling for
    # tiny pore control volumes.
    Lff=L[free][:,free].tocsr()
    Wf=W[:,free].tocsr()
    reservoir_transport_c=(
        -boundary_g[free]
        if finite_inlet
        else np.asarray(L[fixed][:,free].sum(axis=0)).ravel()
    )
    boundary_jac=diags(boundary_g[free]) if finite_inlet else csr_matrix(Lff.shape)
    inv_v=diags(1.0/vol[free])

    def jac(t,y):
        c,q=unpack(y)
        if not adsorption:
            jcc=inv_v@(-Lff-boundary_jac)
            last=csr_matrix(reservoir_transport_c.reshape(1,-1))
            return bmat([[jcc,csr_matrix((len(free),1))],
                         [last,csr_matrix((1,1))]],format="csr")

        dq,sink,js_c,js_q,jqc=adsorption_terms(c,q,with_jacobian=True)
        jqq=-a.k_ldf*eye(npart,format="csr")
        # gas RHS = V^-1[-L*c - sink(c,q)]
        jcc=inv_v@(-Lff-boundary_jac-js_c[free,:])
        jcq=inv_v@(-js_q[free,:])
        if finite_inlet:
            last_c=reservoir_transport_c
            last_q=np.zeros(npart)
        else:
            last_c=reservoir_transport_c+np.asarray(js_c[fixed,:].sum(axis=0)).ravel()
            last_q=np.asarray(js_q[fixed,:].sum(axis=0)).ravel()
        return bmat([
            [jcc,jcq,csr_matrix((len(free),1))],
            [jqc,jqq,csr_matrix((npart,1))],
            [csr_matrix(last_c.reshape(1,-1)),
             csr_matrix(last_q.reshape(1,-1)),csr_matrix((1,1))]
        ],format="csr")

    times=output_times(a.t_end,a.outputs,not a.linear_output_times)
    sol=solve_ivp(rhs,(0,a.t_end),y0,method="BDF",t_eval=times,rtol=a.rtol,atol=a.atol,
                  max_step=a.max_step,jac=jac)
    if not sol.success: sys.exit(f"Integrator failed: {sol.message}")
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    final_c=final_q=None
    minimum_c=np.inf; minimum_q=np.inf
    for k,t in enumerate(sol.t):
        c,q=unpack(sol.y[:,k]); final_c=c; final_q=q
        minimum_c=min(minimum_c,float(c.min()))
        if adsorption: minimum_q=min(minimum_q,float(q.min()))
        gas=float(np.dot(vol,c)); ads=float(np.dot(mass,q)) if adsorption else 0.
        entered=float(sol.y[-1,k]); change=gas+ads-inventory0
        balance=change-entered
        records.append({"time_s":t,"mean_gas_concentration_mol_m3":gas/vol.sum(),
          "gas_inventory_mol":gas,"adsorbed_inventory_mol":ads,"total_inventory_mol":gas+ads,
          "cumulative_reservoir_input_mol":entered,"inventory_change_mol":change,
          "mass_balance_residual_mol":balance,
          "relative_mass_balance_error":abs(balance)/max(abs(entered),abs(change),1e-30),
          "mean_loading_mol_kg":float(np.dot(mass,q)/mass.sum()) if adsorption else 0.})
    fields=list(records[0])
    with (a.output/"timeseries.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    np.savez_compressed(a.output/"final_state.npz",pore_concentration=final_c,
        particle_loading=(final_q if adsorption else np.empty(0)),time_s=sol.t[-1])
    physical_bounds_ok=bool(minimum_c>=-1e-8 and (not adsorption or minimum_q>=-1e-8))
    metadata={"network":str(a.network),"mode":a.mode,"parameters":vars(a),
      "parameters_units":{"diffusivity":"m2/s","concentration":"mol/m3","qsat":"mol/kg","b_pa":"1/Pa","k_ldf":"1/s"},
      "pore_count":npore,"free_pores":len(free),"fixed_inlet_pores":len(fixed),
      "labelled_inlet_pores":int(inlet.sum()),"particle_count":npart,
      "throat_count":len(conns),"initial_inventory_mol":inventory0,
      "solver_success":sol.success,"solver_message":sol.message,"rhs_evaluations":sol.nfev,
      "final_relative_mass_balance_error":records[-1]["relative_mass_balance_error"],
      "maximum_relative_mass_balance_error":max(r["relative_mass_balance_error"] for r in records[1:]),
      "final_minimum_concentration_mol_m3":float(final_c.min()),
      "final_maximum_concentration_mol_m3":float(final_c.max()),
      "final_minimum_loading_mol_kg":float(final_q.min()) if adsorption else None,
      "physical_bounds_ok":physical_bounds_ok,
      "minimum_output_concentration_mol_m3":minimum_c,
      "minimum_output_loading_mol_kg":minimum_q if adsorption else None,
      "boundary_conditions":{
        "bottom_inlet":("finite diffusive reservoir conductance" if finite_inlet else "fixed concentration"),
        "top":"no external flux","wall":"no external flux"},
      "incidence_weighting":"adjacent pore-volume normalized per particle",
      "uptake_sink_weighting":"incidence times positive pore concentration, normalized per particle",
      "desorption_sink_weighting":"particle-normalized incidence"}
    metadata["inlet_boundary_diagnostics"]={
      "formulation":a.inlet_boundary,
      "labelled_pores":int(inlet.sum()),
      "total_physical_area_m2":total_inlet_area,
      "allocated_area_sum_m2":float(boundary_area.sum()) if finite_inlet else None,
      "total_conductance_m3_s":float(boundary_g.sum()) if finite_inlet else None,
      "minimum_distance_m":float(boundary_distance[inlet].min()) if finite_inlet else None,
      "maximum_distance_m":float(boundary_distance[inlet].max()) if finite_inlet else None,
      "area_weighting":a.inlet_area_weighting if finite_inlet else None}
    metadata["parameters"]={k:str(v) if isinstance(v,Path) else v for k,v in metadata["parameters"].items()}
    (a.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(f"mode={a.mode}, pores={npore}, particles={npart}, t={sol.t[-1]:.6g} s")
    print(f"final gas={records[-1]['gas_inventory_mol']:.9e} mol, adsorbed={records[-1]['adsorbed_inventory_mol']:.9e} mol")
    print(f"final mass-balance error={records[-1]['relative_mass_balance_error']:.3e}; max={metadata['maximum_relative_mass_balance_error']:.3e}")
    print(f"final concentration range=[{final_c.min():.6e}, {final_c.max():.6e}] mol/m3; physical_bounds_ok={physical_bounds_ok}")
    if finite_inlet:
        print(f"finite inlet: labelled pores={int(inlet.sum())}, area={boundary_area.sum():.9e} m2, conductance={boundary_g.sum():.9e} m3/s")
    print(f"wrote {a.output}")


if __name__=="__main__":
    main()
