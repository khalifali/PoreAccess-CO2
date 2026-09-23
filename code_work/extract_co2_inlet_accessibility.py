#!/usr/bin/env python3
"""Extract physically interpretable inlet-accessibility descriptors.

Reads every ``seed_*/pore_network.npz`` below a pore-network campaign root.
The resulting CSV can be merged directly with the adsorption campaign dataset
using ``seed``.  Conductances use the same diffusive throat expression as the
transport solver, g = D A / L.

The source is the saved ``pore_inlet`` mask, not every low-lying pore. In the
power22 campaign, a retained pore adjoins an exposed Delaunay hull face for which
at least one face bead satisfies z_bead - radius <= physical_bottom + 1.25 dp.
This is a bead-surface criterion, not a pore-centre cutoff. Changing it to a full
bottom-1dp source band changes the supply problem and loses the strong uptake
association across the tested 3, 4, 5, 7 and 10 dp sink depths. See
``analyze_co2_translated_slabs.py`` for the source/depth control and label audit.

The steady conductance to an interior plane is a geometry diagnostic, not a new
transient adsorption simulation. Inlet pores are fixed at c=1 and pores at or
beyond the requested interior depth are fixed at c=0; all remaining pore
concentrations are obtained from the sparse network Laplacian.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import pdist


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--network-root", required=True, type=Path,
                   help="Root containing seed_*/pore_network.npz directories.")
    p.add_argument("--output", type=Path, default=Path("co2_inlet_accessibility.csv"))
    p.add_argument("--pattern", default="seed_*/pore_network.npz",
                   help="Glob relative to --network-root.")
    p.add_argument("--diffusivity", type=float, default=1.5e-5, help="CO2 diffusivity [m2/s].")
    p.add_argument("--tube-radius", type=float, default=0.008, help="Tube radius [m].")
    p.add_argument("--particle-diameter", type=float, default=0.0016, help="Bead diameter [m].")
    p.add_argument("--interior-depth-dp", type=float, default=5.0,
                   help="Interior sink plane depth above the lowest pore, in particle diameters.")
    p.add_argument("--bed-bottom", type=float, default=None,
                   help="Reference plane z [m]; omitted preserves lowest-pore convention.")
    p.add_argument("--radial-bins", type=int, default=5)
    p.add_argument("--angular-sectors", type=int, default=12)
    p.add_argument("--campaign-dataset", type=Path,
                   help="Optional adsorption dataset to merge by seed.")
    p.add_argument("--merged-output", type=Path,
                   help="Output for optional merged adsorption+accessibility CSV.")
    return p.parse_args()


def seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        m = re.search(r"seed[_-]?(\d+)", part)
        if m:
            return int(m.group(1))
    qa = path.parent / "network_qa.json"
    if qa.exists():
        return int(json.loads(qa.read_text())["seed"])
    raise ValueError(f"Could not determine seed from {path}")


def require_array(z: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in z.files:
        raise KeyError(f"{name!r} absent; available fields: {z.files}")
    return np.asarray(z[name])


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    if not len(values) or not np.any(weights > 0):
        return np.nan, np.nan, np.nan
    w = weights / weights.sum()
    mean = float(np.sum(w * values))
    std = float(np.sqrt(np.sum(w * (values - mean) ** 2)))
    cv = std / mean if mean != 0 else np.nan
    return mean, std, cv


def spatial_descriptors(xy: np.ndarray, tube_radius: float,
                        radial_bins: int, angular_sectors: int) -> dict[str, float]:
    if len(xy) == 0:
        return {k: np.nan for k in (
            "inlet_radial_coverage", "inlet_polar_cell_coverage",
            "inlet_radial_mean_over_R", "inlet_radial_std_over_R",
            "inlet_angular_resultant", "inlet_centroid_offset_over_R",
            "inlet_mean_pair_distance_over_D", "inlet_min_pair_distance_over_D")}
    r = np.linalg.norm(xy, axis=1)
    rn = np.clip(r / tube_radius, 0, np.nextafter(1.0, 0.0))
    theta = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
    rb = np.minimum((rn * radial_bins).astype(int), radial_bins - 1)
    ab = np.minimum((theta / (2 * np.pi) * angular_sectors).astype(int), angular_sectors - 1)
    polar_cells = np.unique(rb * angular_sectors + ab).size
    resultant = abs(np.mean(np.exp(1j * theta))) if len(theta) else np.nan
    pair = pdist(xy) / (2 * tube_radius) if len(xy) > 1 else np.array([])
    return {
        "inlet_radial_coverage": np.unique(rb).size / radial_bins,
        "inlet_polar_cell_coverage": polar_cells / (radial_bins * angular_sectors),
        "inlet_radial_mean_over_R": float(np.mean(rn)),
        "inlet_radial_std_over_R": float(np.std(rn, ddof=1)) if len(rn) > 1 else 0.0,
        # 0 = angularly balanced; 1 = concentrated in one direction.
        "inlet_angular_resultant": float(resultant),
        "inlet_centroid_offset_over_R": float(np.linalg.norm(np.mean(xy, axis=0)) / tube_radius),
        "inlet_mean_pair_distance_over_D": float(np.mean(pair)) if len(pair) else np.nan,
        "inlet_min_pair_distance_over_D": float(np.min(pair)) if len(pair) else np.nan,
    }


def effective_conductance(n: int, conns: np.ndarray, conductance: np.ndarray,
                          source: np.ndarray, sink: np.ndarray) -> tuple[float, int, bool]:
    """Return total source flux for unit concentration difference."""
    source = np.asarray(source, bool).copy()
    sink = np.asarray(sink, bool).copy() & ~source
    if not source.any() or not sink.any():
        return np.nan, 0, False
    rows = np.r_[conns[:, 0], conns[:, 1]]
    cols = np.r_[conns[:, 1], conns[:, 0]]
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    active_components = np.intersect1d(np.unique(labels[source]), np.unique(labels[sink]))
    active = np.isin(labels, active_components)
    source &= active
    sink &= active
    if not source.any() or not sink.any():
        return 0.0, int(active.sum()), False

    diag = np.bincount(conns.ravel(), weights=np.repeat(conductance, 2), minlength=n)
    rr = np.r_[conns[:, 0], conns[:, 1], np.arange(n)]
    cc = np.r_[conns[:, 1], conns[:, 0], np.arange(n)]
    vv = np.r_[-conductance, -conductance, diag]
    lap = coo_matrix((vv, (rr, cc)), shape=(n, n)).tocsr()
    fixed = source | sink | ~active
    free = np.where(~fixed)[0]
    c = np.zeros(n)
    c[source] = 1.0
    if len(free):
        rhs = -lap[free][:, source] @ np.ones(source.sum())
        c[free] = spsolve(lap[free][:, free], rhs)
    a, b = conns[:, 0], conns[:, 1]
    flux = conductance * (c[a] - c[b])
    source_flux = float(np.sum(flux[source[a]]) - np.sum(flux[source[b]]))
    return max(source_flux, 0.0), int(active.sum()), True


def extract(path: Path, args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    with np.load(path) as z:
        xyz = require_array(z, "pore_xyz").astype(float)
        volume = require_array(z, "pore_volume").astype(float)
        inlet = require_array(z, "pore_inlet").astype(bool)
        outlet = require_array(z, "pore_outlet").astype(bool)
        conns = require_array(z, "throat_conns").astype(int)
        length = require_array(z, "throat_length").astype(float)
        area = require_array(z, "throat_area").astype(float)
    if conns.ndim != 2 or conns.shape[1] != 2:
        raise ValueError(f"Invalid throat_conns shape in {path}: {conns.shape}")
    if np.any(length <= 0) or np.any(area <= 0):
        raise ValueError(f"Non-positive throat geometry in {path}")
    g = args.diffusivity * area / length
    a, b = conns[:, 0], conns[:, 1]
    crossing = inlet[a] ^ inlet[b]
    crossing_ids = np.where(crossing)[0]
    second = np.unique(np.where(inlet[a[crossing]], b[crossing], a[crossing]))
    inlet_internal_g = float(g[crossing].sum())
    _, _, inlet_g_cv = weighted_stats(g[crossing], np.ones(crossing.sum()))

    z0 = float(xyz[:, 2].min()) if args.bed_bottom is None else args.bed_bottom
    interior_z = z0 + args.interior_depth_dp * args.particle_diameter
    interior = xyz[:, 2] >= interior_z
    g_depth, active_depth, depth_spans = effective_conductance(
        len(xyz), conns, g, inlet, interior)
    g_outlet, active_outlet, outlet_spans = effective_conductance(
        len(xyz), conns, g, inlet, outlet)

    row: dict[str, float | int | str | bool] = {
        "seed": seed_from_path(path), "network_file": str(path),
        "inlet_pore_count": int(inlet.sum()),
        "inlet_pore_fraction": float(inlet.mean()),
        "inlet_total_pore_volume_m3": float(volume[inlet].sum()),
        "inlet_mean_pore_volume_m3": float(volume[inlet].mean()),
        "inlet_pore_volume_fraction": float(volume[inlet].sum() / volume.sum()),
        "inlet_to_internal_throat_count": int(crossing.sum()),
        "inlet_to_internal_unique_second_layer_pores": int(len(second)),
        "inlet_to_internal_throats_per_inlet_pore": float(crossing.sum() / inlet.sum()),
        "inlet_internal_conductance_m3_s": inlet_internal_g,
        "inlet_internal_conductance_per_inlet_pore_m3_s": float(inlet_internal_g / inlet.sum()),
        "inlet_internal_conductance_cv": inlet_g_cv,
        "interior_depth_dp": args.interior_depth_dp,
        "interior_plane_z_m": interior_z,
        "depth_reference_z_m": z0,
        "depth_reference": "lowest_pore" if args.bed_bottom is None else "bed_bottom",
        "effective_inlet_to_interior_conductance_m3_s": g_depth,
        "effective_inlet_to_interior_active_pores": active_depth,
        "effective_inlet_to_interior_spanning": depth_spans,
        "effective_inlet_to_outlet_conductance_m3_s": g_outlet,
        "effective_inlet_to_outlet_active_pores": active_outlet,
        "effective_inlet_to_outlet_spanning": outlet_spans,
    }
    row.update(spatial_descriptors(xyz[inlet, :2], args.tube_radius,
                                   args.radial_bins, args.angular_sectors))
    return row


def main() -> None:
    args = arguments()
    if args.diffusivity <= 0 or args.tube_radius <= 0 or args.particle_diameter <= 0:
        raise SystemExit("Diffusivity and geometric dimensions must be positive.")
    paths = sorted(args.network_root.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No network files matched {args.network_root / args.pattern}")
    rows, failures = [], []
    for path in paths:
        try:
            row = extract(path, args)
            rows.append(row)
            print(f"seed {row['seed']}: inlet={row['inlet_pore_count']}, "
                  f"G_in-int={row['inlet_internal_conductance_m3_s']:.6e}, "
                  f"G_to_{args.interior_depth_dp:g}dp={row['effective_inlet_to_interior_conductance_m3_s']:.6e}")
        except Exception as exc:
            failures.append({"network_file": str(path), "error": str(exc)})
            print(f"FAILED {path}: {exc}")
    if not rows:
        raise SystemExit("All network extractions failed.")
    result = pd.DataFrame(rows).sort_values("seed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.output.with_name(args.output.stem + "_failures.csv"), index=False)

    if args.campaign_dataset:
        campaign = pd.read_csv(args.campaign_dataset)
        if "seed" not in campaign:
            raise SystemExit("Campaign dataset has no seed column.")
        merged = campaign.merge(result, on="seed", how="left", validate="one_to_one")
        merged_path = args.merged_output or args.output.with_name(
            args.output.stem + "_merged_with_adsorption.csv")
        merged.to_csv(merged_path, index=False)
        missing = int(merged["inlet_pore_count"].isna().sum())
        print(f"Wrote merged dataset {merged_path}; unmatched campaign rows={missing}")
    print(f"Wrote {args.output}: {len(result)}/{len(paths)} networks succeeded")
    if failures:
        raise SystemExit(f"{len(failures)} network(s) failed; see failures CSV")


if __name__ == "__main__":
    main()
