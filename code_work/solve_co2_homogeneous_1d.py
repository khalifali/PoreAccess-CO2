#!/usr/bin/env python3
"""Conservative one-dimensional homogeneous CO2 adsorption model.

The bed is divided into cell-centered finite volumes. Gas diffusion is
represented by one effective axial diffusivity, while every cell contains a
homogeneous adsorbent phase governed by the same Toth/Langmuir equilibrium and
linear-driving-force (LDF) kinetics as the pore-network model.

The bottom boundary is a finite-transfer (Robin) connection to an external CO2
reservoir; the top boundary is closed.  An extra ODE integrates reservoir input,
allowing a direct gas+solid mass-balance check.

Two modes are available:
  simulate : solve with a supplied effective diffusivity;
  fit      : fit only effective diffusivity to a pore-network mean-loading
             timeseries, keeping all material parameters fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.sparse import lil_matrix

RGAS = 8.31446261815324


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["simulate", "fit"], default="simulate")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reference-timeseries", type=Path,
                   help="Pore-network timeseries.csv; required in fit mode.")
    p.add_argument("--reference-time-column", default="time_s")
    p.add_argument("--reference-loading-column", default="mean_loading_mol_kg")

    p.add_argument("--bed-height", required=True, type=float, help="Homogeneous bed height [m].")
    p.add_argument("--porosity", required=True, type=float, help="Bed void fraction [-].")
    p.add_argument("--tube-radius", type=float, default=0.008, help="Tube radius [m].")
    p.add_argument("--cells", type=int, default=100, help="Axial finite-volume cells.")
    p.add_argument("--effective-diffusivity", type=float, default=1.0e-6,
                   help="Effective axial bed diffusivity [m2/s].")
    p.add_argument("--fit-min-diffusivity", type=float, default=1.0e-8)
    p.add_argument("--fit-max-diffusivity", type=float, default=1.5e-5)
    p.add_argument("--fit-xatol-log10", type=float, default=2.0e-3,
                   help="Fit tolerance in log10(diffusivity).")
    p.add_argument("--fit-weighting", choices=["time", "points"], default="time",
                   help="Time weighting avoids bias from logarithmically dense early outputs.")

    p.add_argument("--inlet-concentration", type=float, default=6.05, help="Reservoir CO2 [mol/m3].")
    p.add_argument("--initial-concentration", type=float, default=0.0, help="Initial gas CO2 [mol/m3].")
    p.add_argument("--temperature", type=float, default=298.15, help="Temperature [K].")
    p.add_argument("--particle-density", type=float, default=1600.0, help="Apparent bead density [kg/m3 solid].")
    p.add_argument("--qsat", type=float, default=5.332, help="Saturation loading [mol/kg].")
    p.add_argument("--b-pa", type=float, default=5.093e-5, help="Toth/Langmuir affinity [1/Pa].")
    p.add_argument("--toth-exponent", type=float, default=1.0)
    p.add_argument("--k-ldf", type=float, default=8.55e-3, help="LDF coefficient [1/s].")
    p.add_argument("--initial-loading", type=float, default=0.0, help="Initial solid loading [mol/kg].")

    p.add_argument("--boundary-diffusivity", type=float, default=1.5e-5,
                   help="Diffusivity used only for external inlet transfer [m2/s].")
    p.add_argument("--inlet-transfer-length", type=float, default=8.0e-4,
                   help="External inlet transfer length [m], normally 0.5 dp.")
    p.add_argument("--inlet-mass-transfer-coefficient", type=float,
                   help="Robin coefficient [m/s]; overrides boundary D/length.")

    p.add_argument("--t-end", type=float, default=5000.0)
    p.add_argument("--outputs", type=int, default=301)
    p.add_argument("--linear-output-times", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--max-step", type=float, default=np.inf)
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    positive = {
        "bed height": a.bed_height, "tube radius": a.tube_radius,
        "effective diffusivity": a.effective_diffusivity,
        "temperature": a.temperature, "particle density": a.particle_density,
        "qsat": a.qsat, "b": a.b_pa, "Toth exponent": a.toth_exponent,
        "LDF coefficient": a.k_ldf, "boundary diffusivity": a.boundary_diffusivity,
        "inlet transfer length": a.inlet_transfer_length, "end time": a.t_end,
    }
    bad = [k for k, v in positive.items() if not np.isfinite(v) or v <= 0]
    if bad:
        raise SystemExit("Parameters must be positive: " + ", ".join(bad))
    if not 0 < a.porosity < 1:
        raise SystemExit("--porosity must lie strictly between zero and one.")
    if a.cells < 3:
        raise SystemExit("--cells must be at least 3.")
    if a.outputs < 2:
        raise SystemExit("--outputs must be at least 2.")
    if a.mode == "fit" and a.reference_timeseries is None:
        raise SystemExit("--reference-timeseries is required in fit mode.")
    if not 0 < a.fit_min_diffusivity < a.fit_max_diffusivity:
        raise SystemExit("Require 0 < fit-min-diffusivity < fit-max-diffusivity.")


class HomogeneousBed:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.n = a.cells
        self.dz = a.bed_height / a.cells
        self.area = math.pi * a.tube_radius**2
        self.cell_volume = self.area * self.dz
        self.solid_capacity = (1.0 - a.porosity) * a.particle_density
        self.hm = (a.inlet_mass_transfer_coefficient if
                   a.inlet_mass_transfer_coefficient is not None else
                   a.boundary_diffusivity / a.inlet_transfer_length)
        self.z = (np.arange(self.n) + 0.5) * self.dz
        self.sparsity = self._jacobian_sparsity()

    def qstar(self, concentration: np.ndarray) -> np.ndarray:
        pressure = np.maximum(concentration, 0.0) * RGAS * self.a.temperature
        bp = self.a.b_pa * pressure
        t = self.a.toth_exponent
        return self.a.qsat * bp / np.power(1.0 + np.power(bp, t), 1.0 / t)

    def _jacobian_sparsity(self):
        n = self.n
        s = lil_matrix((2*n + 1, 2*n + 1), dtype=int)
        for i in range(n):
            s[i, i] = 1
            s[i, n+i] = 1
            if i > 0:
                s[i, i-1] = 1
            if i+1 < n:
                s[i, i+1] = 1
            s[n+i, i] = 1
            s[n+i, n+i] = 1
        s[2*n, 0] = 1
        return s.tocsr()

    def rhs(self, _time: float, y: np.ndarray, deff: float) -> np.ndarray:
        n, a = self.n, self.a
        c = y[:n]
        q = y[n:2*n]
        dq = a.k_ldf * (self.qstar(c) - q)

        # Positive face flux points upward, from cell i to cell i+1.
        face = deff * (c[:-1] - c[1:]) / self.dz
        # The reservoir-to-boundary resistance (1/hm) and diffusion from the
        # boundary face to the first cell centre (dz/(2*Deff)) act in series.
        # Including the half-cell term is required for spatial-grid convergence.
        inlet_cell_coefficient = 1.0 / (1.0/self.hm + 0.5*self.dz/deff)
        inlet_flux = inlet_cell_coefficient * (a.inlet_concentration - c[0])
        net_in = np.empty(n)
        net_in[0] = inlet_flux - face[0]
        if n > 2:
            net_in[1:-1] = face[:-1] - face[1:]
        net_in[-1] = face[-1]  # closed top: outgoing top flux is zero
        dc = net_in / (a.porosity * self.dz) - self.solid_capacity / a.porosity * dq

        dy = np.empty(2*n + 1)
        dy[:n] = dc
        dy[n:2*n] = dq
        dy[-1] = self.area * inlet_flux
        return dy

    def initial_state(self) -> np.ndarray:
        y = np.empty(2*self.n + 1)
        y[:self.n] = self.a.initial_concentration
        y[self.n:2*self.n] = self.a.initial_loading
        y[-1] = 0.0
        return y

    def solve(self, deff: float, times: np.ndarray):
        times = np.asarray(times, float)
        if times[0] < 0 or times[-1] > self.a.t_end * (1 + 1e-12):
            raise ValueError("Requested output times lie outside [0,t_end].")
        result = solve_ivp(
            lambda t, y: self.rhs(t, y, deff), (0.0, self.a.t_end),
            self.initial_state(), method="BDF", t_eval=times,
            rtol=self.a.rtol, atol=self.a.atol, max_step=self.a.max_step,
            jac_sparsity=self.sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def inventories(self, result) -> pd.DataFrame:
        a, n = self.a, self.n
        c = result.y[:n]
        q = result.y[n:2*n]
        gas = a.porosity * self.cell_volume * np.sum(c, axis=0)
        ads = self.solid_capacity * self.cell_volume * np.sum(q, axis=0)
        supplied = result.y[-1]
        total = gas + ads
        initial = total[0]
        residual = total - initial - supplied
        scale = np.maximum.reduce([np.abs(total-initial), np.abs(supplied),
                                   np.full_like(total, 1e-30)])
        return pd.DataFrame({
            "time_s": result.t,
            "mean_gas_concentration_mol_m3": np.mean(c, axis=0),
            "gas_inventory_mol": gas,
            "adsorbed_inventory_mol": ads,
            "total_inventory_mol": total,
            "cumulative_reservoir_input_mol": supplied,
            "mass_balance_residual_mol": residual,
            "relative_mass_balance_error": np.abs(residual) / scale,
            "mean_loading_mol_kg": np.mean(q, axis=0),
            "inlet_cell_concentration_mol_m3": c[0],
            "top_cell_concentration_mol_m3": c[-1],
        })


def standard_times(a: argparse.Namespace) -> np.ndarray:
    if a.linear_output_times:
        return np.linspace(0.0, a.t_end, a.outputs)
    if a.outputs == 2:
        return np.array([0.0, a.t_end])
    first = max(1e-8, min(1e-3, a.t_end * 1e-8))
    return np.r_[0.0, np.geomspace(first, a.t_end, a.outputs-1)]


def reference_data(a: argparse.Namespace) -> pd.DataFrame:
    d = pd.read_csv(a.reference_timeseries)
    needed = [a.reference_time_column, a.reference_loading_column]
    missing = [x for x in needed if x not in d]
    if missing:
        raise ValueError(f"Reference file lacks columns {missing}; available={list(d.columns)}")
    d = d[needed].rename(columns={a.reference_time_column: "time_s",
                                  a.reference_loading_column: "reference_loading_mol_kg"})
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("time_s")
    d = d[(d.time_s >= 0) & (d.time_s <= a.t_end)].drop_duplicates("time_s")
    if len(d) < 5:
        raise ValueError("Fewer than five usable reference points in [0,t_end].")
    if d.time_s.iloc[0] > 0:
        d = pd.concat([pd.DataFrame({"time_s":[0.0],
                                     "reference_loading_mol_kg":[a.initial_loading]}), d],
                      ignore_index=True)
    return d


def fit_diffusivity(model: HomogeneousBed, a: argparse.Namespace):
    ref = reference_data(a)
    times = ref.time_s.to_numpy(float)
    observed = ref.reference_loading_mol_kg.to_numpy(float)
    scale = max(np.ptp(observed), abs(observed[-1]), 1e-12)
    if a.fit_weighting == "time":
        # Trapezoidal integration weights. The objective approximates
        # sqrt(integral(error^2 dt)/duration), so changing output density does
        # not change the fitted parameter.
        weights = np.empty(len(times))
        weights[0] = 0.5 * (times[1] - times[0])
        weights[-1] = 0.5 * (times[-1] - times[-2])
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    else:
        weights = np.ones(len(times))
    weights = weights / weights.sum()
    trace = []

    def objective(log10_deff: float) -> float:
        deff = 10.0**log10_deff
        sol = model.solve(deff, times)
        predicted = np.mean(sol.y[model.n:2*model.n], axis=0)
        nrmse = float(np.sqrt(np.sum(weights * (predicted-observed)**2)) / scale)
        trace.append({"effective_diffusivity_m2_s": deff, "normalized_rmse": nrmse})
        print(f"fit D_eff={deff:.8e} m2/s, normalized RMSE={nrmse:.6e}")
        return nrmse

    opt = minimize_scalar(objective,
                          bounds=(math.log10(a.fit_min_diffusivity),
                                  math.log10(a.fit_max_diffusivity)),
                          method="bounded", options={"xatol": a.fit_xatol_log10})
    if not opt.success:
        raise RuntimeError(f"Diffusivity fit failed: {opt.message}")
    best = 10.0**opt.x
    solution = model.solve(best, times)
    predicted = np.mean(solution.y[model.n:2*model.n], axis=0)
    comparison = ref.copy()
    comparison["homogeneous_loading_mol_kg"] = predicted
    comparison["residual_mol_kg"] = observed - predicted
    return best, solution, comparison, pd.DataFrame(trace), float(opt.fun)


def write_results(model: HomogeneousBed, a: argparse.Namespace, deff: float,
                  solution, fit_comparison=None, fit_trace=None, fit_nrmse=None) -> None:
    a.output.mkdir(parents=True, exist_ok=True)
    series = model.inventories(solution)
    series.to_csv(a.output / "timeseries.csv", index=False)
    c = solution.y[:model.n]
    q = solution.y[model.n:2*model.n]
    profile_rows = []
    for j, time in enumerate(solution.t):
        for i, z in enumerate(model.z):
            profile_rows.append({"time_s": time, "cell": i, "z_m": z,
                                 "concentration_mol_m3": c[i,j],
                                 "loading_mol_kg": q[i,j]})
    pd.DataFrame(profile_rows).to_csv(a.output / "axial_profiles.csv", index=False)
    np.savez_compressed(a.output / "final_state.npz", z_m=model.z,
                        concentration_mol_m3=c[:,-1], loading_mol_kg=q[:,-1])
    if fit_comparison is not None:
        fit_comparison.to_csv(a.output / "fit_comparison.csv", index=False)
        fit_trace.to_csv(a.output / "fit_trace.csv", index=False)

    final = series.iloc[-1]
    metadata = {
        "mode": a.mode, "solver_success": True,
        "effective_diffusivity_m2_s": deff,
        "molecular_to_effective_diffusivity_ratio": a.boundary_diffusivity/deff,
        "fit_normalized_rmse": fit_nrmse,
        "fit_weighting": a.fit_weighting if a.mode == "fit" else None,
        "reference_timeseries": str(a.reference_timeseries) if a.reference_timeseries else None,
        "bed_height_m": a.bed_height, "porosity": a.porosity,
        "tube_radius_m": a.tube_radius, "cross_section_area_m2": model.area,
        "cells": a.cells, "cell_width_m": model.dz,
        "inlet_mass_transfer_coefficient_m_s": model.hm,
        "inlet_first_cell_effective_coefficient_m_s":
            1.0 / (1.0/model.hm + 0.5*model.dz/deff),
        "inlet_concentration_mol_m3": a.inlet_concentration,
        "temperature_K": a.temperature, "particle_density_kg_m3": a.particle_density,
        "qsat_mol_kg": a.qsat, "b_Pa_inverse": a.b_pa,
        "toth_exponent": a.toth_exponent, "k_ldf_s_inverse": a.k_ldf,
        "t_end_s": a.t_end, "rtol": a.rtol, "atol": a.atol,
        "final_mean_loading_mol_kg": float(final.mean_loading_mol_kg),
        "final_adsorbed_inventory_mol": float(final.adsorbed_inventory_mol),
        "maximum_relative_mass_balance_error": float(series.relative_mass_balance_error.max()),
        "minimum_concentration_mol_m3": float(c.min()),
        "maximum_concentration_mol_m3": float(c.max()),
        "minimum_loading_mol_kg": float(q.min()),
        "maximum_loading_mol_kg": float(q.max()),
        "physical_bounds_ok": bool(c.min() >= -1e-8 and c.max() <= a.inlet_concentration+1e-8
                                   and q.min() >= -1e-8 and q.max() <= a.qsat+1e-8),
    }
    (a.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"mode={a.mode}, cells={a.cells}, D_eff={deff:.8e} m2/s, t={solution.t[-1]:g} s")
    print(f"final loading={final.mean_loading_mol_kg:.9e} mol/kg; "
          f"max mass-balance error={metadata['maximum_relative_mass_balance_error']:.3e}")
    if fit_nrmse is not None:
        print(f"fit normalized RMSE={fit_nrmse:.6e}")
    print(f"wrote {a.output}")


def main() -> None:
    a = arguments()
    validate(a)
    model = HomogeneousBed(a)
    if a.mode == "fit":
        deff, solution, comparison, trace, nrmse = fit_diffusivity(model, a)
        write_results(model, a, deff, solution, comparison, trace, nrmse)
    else:
        times = standard_times(a)
        solution = model.solve(a.effective_diffusivity, times)
        write_results(model, a, a.effective_diffusivity, solution)


if __name__ == "__main__":
    main()
