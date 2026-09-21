#!/usr/bin/env python3
"""Create slide-ready 2-D inlet-region views of a packed bed and pore network.

The script reads the existing `pore_network.npz` written by
`extract_co2_pore_network.py` and writes two matching figures:

1. particles only;
2. the identical particle slice with pore centres and throats overlaid.

By default the view is a thin central slab and the bottom 8 particle diameters,
which is useful for explaining inlet accessibility.

Example
-------
python3 plot_pore_network_slice.py \
    --network co2_pore_networks_power22/seed_18427/pore_network.npz \
    --output-prefix seed18427_inlet_slice

This writes PDF and PNG versions of:
  seed18427_inlet_slice_particles
  seed18427_inlet_slice_particles_network
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
    p.add_argument("--output-prefix", type=Path,
                   default=Path("seed18427_inlet_slice"))
    p.add_argument("--slice-center-y", type=float,
                   help="Centre of the y-slice [m]. If omitted, choose it automatically to include as many inlet pores as possible.")
    p.add_argument("--slice-half-thickness-dp", type=float, default=0.50,
                   help="Half-thickness of the pore-network slab in particle diameters.")
    p.add_argument("--height-dp", type=float, default=8.0,
                   help="Axial extent above the bed bottom in particle diameters.")
    p.add_argument("--z-min", type=float, default=0.0,
                   help="Lower axial bound of the displayed region [m].")
    p.add_argument("--tube-radius", type=float, default=0.008,
                   help="Tube radius [m], used for plot limits.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--pdf-only", action="store_true")
    return p.parse_args()


def draw_particles(ax, xyz, radius, y0, zmin, zmax):
    """Draw true sphere/plane cross-sections for the plane y=y0."""
    dy = np.abs(xyz[:, 1] - y0)
    plane_mask = dy <= radius
    cross_radius = np.sqrt(np.maximum(radius[plane_mask] ** 2 - dy[plane_mask] ** 2, 0.0))
    pxyz = xyz[plane_mask]

    # Keep particles whose 2-D cross-section intersects the requested z-window.
    zmask = (pxyz[:, 2] + cross_radius >= zmin) & (pxyz[:, 2] - cross_radius <= zmax)
    pxyz = pxyz[zmask]
    cross_radius = cross_radius[zmask]

    mm = 1e3
    for (x, _, z), rr in zip(pxyz, cross_radius):
        ax.add_patch(Circle((x * mm, z * mm), rr * mm,
                            facecolor="0.86", edgecolor="0.25", linewidth=0.55))
    return len(pxyz)


def configure_axis(ax, tube_radius, zmin, zmax):
    mm = 1e3
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-tube_radius * mm, tube_radius * mm)
    ax.set_ylim(zmin * mm, zmax * mm)
    ax.grid(False)


def save_figure(fig, base: Path, dpi: int, pdf_only: bool):
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {pdf}")
    if not pdf_only:
        png = base.with_suffix(".png")
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        print(f"wrote {png}")


def main() -> int:
    a = arguments()
    if not a.network.is_file():
        raise SystemExit(f"Network file not found: {a.network}")

    z = np.load(a.network)
    required = ("particle_xyz", "particle_radius", "pore_xyz", "throat_conns", "pore_inlet")
    missing = [k for k in required if k not in z]
    if missing:
        raise SystemExit(f"{a.network} lacks required arrays: {missing}")

    particle_xyz = np.asarray(z["particle_xyz"], dtype=float)
    particle_radius = np.asarray(z["particle_radius"], dtype=float)
    pore_xyz = np.asarray(z["pore_xyz"], dtype=float)
    throat_conns = np.asarray(z["throat_conns"], dtype=int)
    pore_inlet = np.asarray(z["pore_inlet"], dtype=bool)

    pore_radius = (np.asarray(z["pore_inscribed_radius"], dtype=float)
                   if "pore_inscribed_radius" in z else np.ones(len(pore_xyz)))
    throat_radius = (np.asarray(z["throat_radius"], dtype=float)
                     if "throat_radius" in z else np.ones(len(throat_conns)))

    dp = 2.0 * float(np.median(particle_radius))
    slab_half = a.slice_half_thickness_dp * dp
    zmin = a.z_min
    zmax = zmin + a.height_dp * dp
    mm = 1e3

    # Choose a thin slice that actually intersects inlet-connected pores.
    if a.slice_center_y is None:
        inlet_ids = np.flatnonzero(pore_inlet)
        if len(inlet_ids) == 0:
            raise SystemExit("Network contains no inlet-labelled pores.")
        inlet_y = pore_xyz[inlet_ids, 1]
        counts = np.asarray([
            np.count_nonzero(np.abs(inlet_y - yc) <= slab_half)
            for yc in inlet_y
        ])
        best = np.flatnonzero(counts == counts.max())
        pick = best[np.argmin(np.abs(inlet_y[best]))]
        y0 = float(inlet_y[pick])
    else:
        y0 = float(a.slice_center_y)

    # Pore/throat selection: same thin y-slab and same z-window.
    pore_mask = (
        (np.abs(pore_xyz[:, 1] - y0) <= slab_half)
        & (pore_xyz[:, 2] >= zmin)
        & (pore_xyz[:, 2] <= zmax)
    )
    kept_pores = np.flatnonzero(pore_mask)
    inlet_overlay_mask = pore_mask & pore_inlet
    kept_inlet_pores = np.flatnonzero(inlet_overlay_mask)

    edge_mask = pore_mask[throat_conns[:, 0]] & pore_mask[throat_conns[:, 1]]
    kept_edges = throat_conns[edge_mask]

    segments = (
        np.stack(
            [pore_xyz[kept_edges[:, 0]][:, [0, 2]],
             pore_xyz[kept_edges[:, 1]][:, [0, 2]]],
            axis=1,
        ) * mm
        if len(kept_edges)
        else np.empty((0, 2, 2))
    )

    # ------------------------------------------------------------------
    # Figure 1: particles only
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 5.5), constrained_layout=True)
    nparticles = draw_particles(ax, particle_xyz, particle_radius, y0, zmin, zmax)
    configure_axis(ax, a.tube_radius, zmin, zmax)
    ax.set_title(f"Particle packing: bottom {a.height_dp:g} $d_p$")
    save_figure(fig, Path(str(a.output_prefix) + "_particles"), a.dpi, a.pdf_only)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: same particles with network directly overlaid
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 5.5), constrained_layout=True)
    draw_particles(ax, particle_xyz, particle_radius, y0, zmin, zmax)

    if len(segments):
        tr = throat_radius[edge_mask]
        if np.ptp(tr) > 0:
            widths = 0.45 + 1.45 * (tr - tr.min()) / np.ptp(tr)
        else:
            widths = np.full(len(tr), 0.8)
        ax.add_collection(LineCollection(segments, linewidths=widths, alpha=0.75))

    # External reservoir and reservoir-to-inlet connections.
    ax.axhspan(zmin * mm, (zmin + 0.18 * dp) * mm, alpha=0.12, zorder=0)
    for pid in kept_inlet_pores:
        xi = pore_xyz[pid, 0] * mm
        zi = pore_xyz[pid, 2] * mm
        ax.plot([xi, xi], [zmin * mm, zi], linewidth=1.3, alpha=0.95, zorder=4)

    if len(kept_pores):
        pr = pore_radius[kept_pores]
        if np.max(pr) > 0:
            sizes = 9.0 + 34.0 * (pr / np.max(pr)) ** 1.2
        else:
            sizes = np.full(len(pr), 12.0)
        ax.scatter(
            pore_xyz[kept_pores, 0] * mm,
            pore_xyz[kept_pores, 2] * mm,
            s=sizes,
            edgecolors="none",
            alpha=0.95,
            zorder=5,
        )

    configure_axis(ax, a.tube_radius, zmin, zmax)
    ax.set_title(f"Same packing with inlet-connected pore network: bottom {a.height_dp:g} $d_p$")
    save_figure(fig, Path(str(a.output_prefix) + "_particles_network"), a.dpi, a.pdf_only)
    plt.close(fig)

    print(
        f"region: z={zmin:.6g}..{zmax:.6g} m ({a.height_dp:g} dp), "
        f"|y-{y0:.6g}|<={slab_half:.6g} m"
    )
    print(
        f"particle cross-sections={nparticles}, "
        f"pores in slab/window={pore_mask.sum()}, "
        f"inlet pores in slab/window={inlet_overlay_mask.sum()}, "
        f"throats in slab/window={edge_mask.sum()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
