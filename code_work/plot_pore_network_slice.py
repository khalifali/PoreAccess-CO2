#!/usr/bin/env python3
"""Create a clean 2-D thin-slice view of one packed bed and its pore network.

The script reads the existing `pore_network.npz` written by
`extract_co2_pore_network.py`.  The left panel shows the true circular
cross-sections of particles intersecting a thin y-slab.  The right panel shows
pore centres and throats whose pore centres lie inside the same slab.

Example
-------
python3 plot_pore_network_slice.py \
    --network co2_pore_networks_power22/seed_18427/pore_network.npz \
    --output pore_network_slice_seed18427.pdf

A PNG with the same basename is also written unless --pdf-only is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--network", type=Path,
                   default=Path("co2_pore_networks_power22/seed_18427/pore_network.npz"))
    p.add_argument("--output", type=Path,
                   default=Path("pore_network_slice_seed18427.pdf"))
    p.add_argument("--slice-center-y", type=float, default=0.0,
                   help="Centre of the y-slice [m].")
    p.add_argument("--slice-half-thickness-dp", type=float, default=0.55,
                   help="Half-thickness of the visualized slab in particle diameters.")
    p.add_argument("--tube-radius", type=float, default=0.008,
                   help="Tube radius [m], used only for plot limits.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--pdf-only", action="store_true")
    return p.parse_args()


def main() -> int:
    a = arguments()
    if not a.network.is_file():
        raise SystemExit(f"Network file not found: {a.network}")

    z = np.load(a.network)
    required = ("particle_xyz", "particle_radius", "pore_xyz", "throat_conns")
    missing = [k for k in required if k not in z]
    if missing:
        raise SystemExit(f"{a.network} lacks required arrays: {missing}")

    particle_xyz = np.asarray(z["particle_xyz"], dtype=float)
    particle_radius = np.asarray(z["particle_radius"], dtype=float)
    pore_xyz = np.asarray(z["pore_xyz"], dtype=float)
    throat_conns = np.asarray(z["throat_conns"], dtype=int)

    pore_radius = (np.asarray(z["pore_inscribed_radius"], dtype=float)
                   if "pore_inscribed_radius" in z else np.ones(len(pore_xyz)))
    throat_radius = (np.asarray(z["throat_radius"], dtype=float)
                     if "throat_radius" in z else np.ones(len(throat_conns)))

    dp = 2.0 * float(np.median(particle_radius))
    half = a.slice_half_thickness_dp * dp
    y0 = a.slice_center_y

    # Particle cross-sections through the central plane y=y0.  A sphere whose
    # centre is offset by dy intersects the plane with radius sqrt(r^2-dy^2).
    dy_particle = np.abs(particle_xyz[:, 1] - y0)
    particle_mask = dy_particle <= particle_radius
    cross_radius = np.sqrt(
        np.maximum(particle_radius[particle_mask] ** 2 - dy_particle[particle_mask] ** 2, 0.0)
    )

    # For the pore network, use a thin slab rather than an exact plane because
    # pores and throats are point/line objects in 3-D.
    pore_mask = np.abs(pore_xyz[:, 1] - y0) <= half
    kept_pores = np.flatnonzero(pore_mask)
    pore_local = np.full(len(pore_xyz), -1, dtype=int)
    pore_local[kept_pores] = np.arange(len(kept_pores))

    edge_mask = pore_mask[throat_conns[:, 0]] & pore_mask[throat_conns[:, 1]]
    kept_edges = throat_conns[edge_mask]

    # x-z coordinates in mm for presentation-friendly axes.
    mm = 1e3
    pxyz = particle_xyz[particle_mask]
    qxyz = pore_xyz[kept_pores]
    segments = np.stack(
        [pore_xyz[kept_edges[:, 0]][:, [0, 2]],
         pore_xyz[kept_edges[:, 1]][:, [0, 2]]],
        axis=1
    ) * mm if len(kept_edges) else np.empty((0, 2, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), constrained_layout=True)

    ax = axes[0]
    for (x, _, zz), rr in zip(pxyz, cross_radius):
        ax.add_patch(Circle((x * mm, zz * mm), rr * mm,
                            facecolor="0.86", edgecolor="0.25", linewidth=0.55))
    ax.set_title("Particle packing: central cross-section")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-a.tube_radius * mm, a.tube_radius * mm)

    ax = axes[1]
    if len(segments):
        tr = throat_radius[edge_mask]
        if np.ptp(tr) > 0:
            lw = 0.35 + 1.6 * (tr - tr.min()) / np.ptp(tr)
        else:
            lw = np.full(len(tr), 0.8)
        ax.add_collection(LineCollection(segments, linewidths=lw, alpha=0.7))

    if len(qxyz):
        pr = pore_radius[kept_pores]
        if np.max(pr) > 0:
            sizes = 7.0 + 34.0 * (pr / np.max(pr)) ** 1.2
        else:
            sizes = np.full(len(pr), 12.0)
        ax.scatter(qxyz[:, 0] * mm, qxyz[:, 2] * mm, s=sizes,
                   edgecolors="none", alpha=0.9)

    ax.set_title("Pore network: thin central slab")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-a.tube_radius * mm, a.tube_radius * mm)

    zmin = min(np.min(particle_xyz[:, 2]), np.min(pore_xyz[:, 2])) * mm
    zmax = max(np.max(particle_xyz[:, 2]), np.max(pore_xyz[:, 2])) * mm
    pad = 0.5
    for ax in axes:
        ax.set_ylim(zmin - pad, zmax + pad)
        ax.grid(False)

    fig.suptitle(
        f"{a.network.parent.name}: packing and corresponding pore network\n"
        f"network slab: |y - {y0*mm:.2f} mm| <= {half*mm:.2f} mm "
        f"({a.slice_half_thickness_dp:.2f} d_p)"
    )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, bbox_inches="tight")
    if not a.pdf_only:
        png = a.output.with_suffix(".png")
        fig.savefig(png, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    print(
        f"particles intersecting plane={particle_mask.sum()}, "
        f"pores in slab={pore_mask.sum()}, throats in slab={edge_mask.sum()}"
    )
    print(f"wrote {a.output}")
    if not a.pdf_only:
        print(f"wrote {a.output.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
