#!/usr/bin/env python3
"""Robust porosity analysis for cylindrical LAMMPS packed beds.

Reads the last snapshot from each particles_final*.dump below co2_13x_seed_*
directories. Sphere/bin intersections are evaluated by Gauss-Legendre
quadrature, so particles crossing axial, radial, or interior boundaries are
not assigned merely by their centres.

Outputs (under --output-dir):
  packing_porosity_summary.csv
  axial_porosity_profiles.csv
  radial_porosity_profiles.csv
  packing_porosity_summary.json
  porosity_profiles.png             (unless --no-plots)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def last_atom_snapshot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    found = None
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i + 1]); i += 2
        if not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Expected NUMBER OF ATOMS in {path}")
        n = int(lines[i + 1]); i += 2
        if lines[i].startswith("ITEM: BOX BOUNDS"):
            i += 4
        if not lines[i].startswith("ITEM: ATOMS"):
            raise ValueError(f"Expected ATOMS header in {path}")
        cols = lines[i].split()[2:]; i += 1
        rows = np.asarray([[float(x) for x in lines[i+j].split()] for j in range(n)])
        i += n
        found = (step, cols, rows)
    if found is None:
        raise ValueError(f"No atom snapshot in {path}")
    return found[1], found[2]


def circle_intersection_area(r1: np.ndarray, r2: float, d: np.ndarray) -> np.ndarray:
    """Intersection of circles with radii r1, r2 and centre separation d."""
    r1, d = np.broadcast_arrays(np.maximum(r1, 0.0), np.maximum(d, 0.0))
    out = np.zeros_like(r1)
    contained = d <= np.abs(r2-r1)
    out[contained] = math.pi*np.minimum(r1[contained], r2)**2
    partial = (~contained) & (d < r1+r2) & (r1 > 0)
    a, dd = r1[partial], d[partial]
    c1 = np.clip((dd*dd+a*a-r2*r2)/(2*dd*a), -1, 1)
    c2 = np.clip((dd*dd+r2*r2-a*a)/(2*dd*r2), -1, 1)
    rad = np.maximum((-dd+a+r2)*(dd+a-r2)*(dd-a+r2)*(dd+a+r2), 0)
    out[partial] = a*a*np.arccos(c1)+r2*r2*np.arccos(c2)-0.5*np.sqrt(rad)
    return out


class VolumeIntegrator:
    def __init__(self, order: int):
        self.nodes, self.weights = leggauss(order)

    def sphere_in_cyl_shell_z(self, x, y, z, radius, rlo, rhi, zlo, zhi):
        """Solid volume in coaxial annular cylinder and axial interval."""
        a, b = max(z-radius, zlo), min(z+radius, zhi)
        if b <= a or rhi <= rlo:
            return 0.0
        zz = 0.5*(b-a)*self.nodes + 0.5*(a+b)
        cross_r = np.sqrt(np.maximum(radius*radius-(zz-z)**2, 0))
        d = np.full_like(cross_r, math.hypot(x, y))
        area_hi = circle_intersection_area(cross_r, rhi, d)
        area_lo = circle_intersection_area(cross_r, rlo, d) if rlo > 0 else 0.0
        return float(0.5*(b-a)*np.dot(self.weights, area_hi-area_lo))


def particle_columns(cols, rows):
    def col(*names):
        for name in names:
            if name in cols:
                return rows[:, cols.index(name)]
        raise ValueError(f"Need one of {names}; found {cols}")
    return col("x", "xu"), col("y", "yu"), col("z", "zu"), col("radius")


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--pattern", default="particles_final*.dump")
    p.add_argument("--tube-radius", type=float, default=0.008)
    p.add_argument("--bottom", type=float, default=0.0)
    p.add_argument("--axial-bins", type=int, default=30)
    p.add_argument("--radial-bins", type=int, default=10)
    p.add_argument("--top-percentile", type=float, default=99.0,
                   help="percentile of particle-top elevations defining robust bed top")
    p.add_argument("--interior-wall-exclusion-dp", type=float, default=1.0)
    p.add_argument("--interior-bottom-exclusion-dp", type=float, default=1.0)
    p.add_argument("--interior-top-exclusion-dp", type=float, default=1.0)
    p.add_argument("--quadrature-order", type=int, default=48)
    p.add_argument("--output-dir", type=Path, default=Path("co2_porosity_analysis"))
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    if not 0 < a.top_percentile <= 100: p.error("top percentile must be in (0,100]")
    cases = sorted(a.root.glob(f"co2_13x_seed_*/{a.pattern}"))
    # Permit invocation from inside a single seed directory.
    if not cases: cases = sorted(a.root.glob(a.pattern))
    if not cases: p.error(f"No {a.pattern} found below {a.root}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    integ = VolumeIntegrator(a.quadrature_order)
    summary, axial_rows, radial_rows = [], [], []

    for dump in cases:
        seed_match = re.search(r"co2_13x_seed_(\d+)", str(dump))
        seed = int(seed_match.group(1)) if seed_match else dump.parent.name
        cols, rows = last_atom_snapshot(dump)
        x, y, z, rad = particle_columns(cols, rows)
        dp = 2*float(np.median(rad))
        tops = z+rad
        ztops = {q: float(np.percentile(tops, q)) for q in (95, 98, 99, 100)}
        bed_top = float(np.percentile(tops, a.top_percentile))
        full_solid = sum(integ.sphere_in_cyl_shell_z(*v, 0, a.tube_radius, a.bottom, bed_top)
                         for v in zip(x,y,z,rad))
        bed_vol = math.pi*a.tube_radius**2*(bed_top-a.bottom)
        bulk_eps = 1-full_solid/bed_vol

        zi0 = a.bottom+a.interior_bottom_exclusion_dp*dp
        zi1 = bed_top-a.interior_top_exclusion_dp*dp
        ri = a.tube_radius-a.interior_wall_exclusion_dp*dp
        if zi1 <= zi0 or ri <= 0: raise ValueError("Interior exclusions leave no analysis volume")
        int_solid = sum(integ.sphere_in_cyl_shell_z(*v, 0, ri, zi0, zi1)
                        for v in zip(x,y,z,rad))
        int_vol = math.pi*ri**2*(zi1-zi0)
        int_eps = 1-int_solid/int_vol

        zedges = np.linspace(a.bottom, bed_top, a.axial_bins+1)
        for k, (lo, hi) in enumerate(zip(zedges[:-1], zedges[1:])):
            solid = sum(integ.sphere_in_cyl_shell_z(*v, 0, a.tube_radius, lo, hi)
                        for v in zip(x,y,z,rad))
            vol = math.pi*a.tube_radius**2*(hi-lo)
            axial_rows.append(dict(seed=seed, bin=k, z_low_m=lo, z_high_m=hi,
                z_center_m=.5*(lo+hi), solid_fraction=solid/vol, porosity=1-solid/vol))
        redges = np.linspace(0, a.tube_radius, a.radial_bins+1)
        for k, (lo, hi) in enumerate(zip(redges[:-1], redges[1:])):
            solid = sum(integ.sphere_in_cyl_shell_z(*v, lo, hi, a.bottom, bed_top)
                        for v in zip(x,y,z,rad))
            vol = math.pi*(hi*hi-lo*lo)*(bed_top-a.bottom)
            radial_rows.append(dict(seed=seed, bin=k, r_low_m=lo, r_high_m=hi,
                r_center_m=.5*(lo+hi), r_over_R=.5*(lo+hi)/a.tube_radius,
                solid_fraction=solid/vol, porosity=1-solid/vol))
        summary.append(dict(seed=seed, dump=str(dump), particles=len(rows), diameter_m=dp,
            top_definition_percentile=a.top_percentile, bed_top_m=bed_top,
            bed_height_m=bed_top-a.bottom, z95_top_m=ztops[95], z98_top_m=ztops[98],
            z99_top_m=ztops[99], zmax_top_m=ztops[100], bulk_porosity=bulk_eps,
            interior_porosity=int_eps, interior_radius_m=ri, interior_z_low_m=zi0,
            interior_z_high_m=zi1, interior_wall_exclusion_dp=a.interior_wall_exclusion_dp,
            interior_bottom_exclusion_dp=a.interior_bottom_exclusion_dp,
            interior_top_exclusion_dp=a.interior_top_exclusion_dp))
        print(f"seed {seed}: bulk eps={bulk_eps:.6f}, interior eps={int_eps:.6f}, top={bed_top:.6g} m")

    write_csv(a.output_dir/"packing_porosity_summary.csv", summary)
    write_csv(a.output_dir/"axial_porosity_profiles.csv", axial_rows)
    write_csv(a.output_dir/"radial_porosity_profiles.csv", radial_rows)
    payload = {"settings": vars(a), "cases": summary}
    payload["settings"] = {k: str(v) if isinstance(v, Path) else v for k,v in payload["settings"].items()}
    (a.output_dir/"packing_porosity_summary.json").write_text(json.dumps(payload, indent=2)+"\n")

    if not a.no_plots:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
            for seed in [r["seed"] for r in summary]:
                az=[r for r in axial_rows if r["seed"]==seed]
                rr=[r for r in radial_rows if r["seed"]==seed]
                axes[0].plot([(r["z_center_m"]-a.bottom)/(next(s["bed_height_m"] for s in summary if s["seed"]==seed)) for r in az], [r["porosity"] for r in az], alpha=.35)
                axes[1].plot([r["r_over_R"] for r in rr], [r["porosity"] for r in rr], alpha=.35)
            axes[0].set(xlabel="z / robust bed height", ylabel="local porosity", title="Axial profiles")
            axes[1].set(xlabel="r / tube radius", ylabel="local porosity", title="Radial profiles")
            for ax in axes: ax.grid(alpha=.25)
            fig.savefig(a.output_dir/"porosity_profiles.png", dpi=200)
            plt.close(fig)
        except ImportError:
            print("matplotlib unavailable: CSV/JSON written, plot skipped")
    print(f"Wrote results to {a.output_dir}")


if __name__ == "__main__":
    main()
