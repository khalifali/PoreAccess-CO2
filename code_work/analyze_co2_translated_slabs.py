#!/usr/bin/env python3
"""Tortuosity and matched translated-slab controls, with reproducible figures.

Slabs are induced subgraphs: delete all edges leaving the window. Source/sink
bands lie INSIDE each window with identical thickness at every position. This
boundary treatment differs from the original hull-labelled inlet diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree
from extract_co2_pore_network import FACES, circumcenter_tetra
from scipy.stats import spearmanr
from extract_co2_inlet_accessibility import effective_conductance, seed_from_path

TARGETS = {'final_uptake_mol_kg': r'$q_{5000}$ [mol kg$^{-1}$]',
           't50_final_uptake_s': r'$t_{50}$ [s]',
           'underutilized_particle_fraction': r'$f_{\mathrm{under}}$'}


def reconstruct_inlet_labels(pores, particles, radii, bottom=0., boundary_layer_dp=1.25):
    """Audit the original geometric labels, not a pore-centre height cutoff.

    A retained pore must adjoin a convex-hull face (no Delaunay neighbour).
    At least one of that face's three beads must satisfy
    z_bead - radius <= physical_bottom + 1.25 * particle_diameter.
    The face need not intersect the bottom plane or have a downward normal.
    """
    tri = Delaunay(particles, qhull_options='Qbb Qc Qz Q12')
    tree = cKDTree(pores)
    labels = np.zeros(len(pores), dtype=bool)
    dp = 2*float(np.median(radii))
    for si, fi in zip(*np.where(tri.neighbors < 0)):
        face = tri.simplices[si][list(FACES[fi])]
        if np.min(particles[face, 2]-radii[face]) <= bottom+boundary_layer_dp*dp:
            try:
                center, _ = circumcenter_tetra(particles[tri.simplices[si]])
            except np.linalg.LinAlgError:
                continue
            distance, index = tree.query(center)
            if distance < 1e-10:
                labels[index] = True
    return labels


def slab_conductance(xyz, conns, g, lower, upper, band):
    """Return unit-drop conductance; a disconnected slab has zero conductance."""
    if not (upper > lower and 0 < band < (upper-lower)/2):
        raise ValueError('Require positive, non-overlapping boundary bands')
    keep = (xyz[:, 2] >= lower) & (xyz[:, 2] <= upper)
    ids = np.flatnonzero(keep)
    index = np.full(len(xyz), -1, dtype=int)
    index[ids] = np.arange(len(ids))
    edges = keep[conns].all(axis=1)
    z = xyz[ids, 2]
    source, sink = z <= lower+band, z >= upper-band
    if not source.any() or not sink.any():
        raise ValueError('Slab has an empty source or sink band')
    value, active, spans = effective_conductance(len(ids), index[conns[edges]], g[edges], source, sink)
    return dict(conductance_m3_s=value, pore_count=len(ids), throat_count=int(edges.sum()),
                source_count=int(source.sum()), sink_count=int(sink.sum()), active_pores=active, spanning=spans)


def tortuosity(xyz, conns, length, inlet, outlet):
    graph = coo_matrix((np.r_[length, length],
                       (np.r_[conns[:, 0], conns[:, 1]], np.r_[conns[:, 1], conns[:, 0]])),
                      shape=(len(xyz), len(xyz))).tocsr()
    distance = dijkstra(graph, directed=False, indices=np.flatnonzero(inlet), min_only=True)
    return float(distance[outlet].min()/(xyz[outlet, 2].max()-xyz[inlet, 2].min()))


def save_figure(fig, out, name):
    for extension in ('pdf', 'png'):
        fig.savefig(out/f'{name}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--network-root', type=Path, default=Path('co2_pore_networks_power22'))
    p.add_argument('--campaign-dataset', type=Path, default=Path('co2_adsorption_finite_inlet_analysis/co2_adsorption_campaign_dataset.csv'))
    p.add_argument('--output', type=Path, default=Path('co2_translated_slabs'))
    p.add_argument('--window-dp', type=float, default=5)
    p.add_argument('--max-start-dp', type=float, default=20)
    p.add_argument('--boundary-bands-dp', type=float, nargs='+', default=[0.5, 1, 1.5])
    p.add_argument('--particle-diameter', type=float, default=.0016)
    p.add_argument('--diffusivity', type=float, default=1.5e-5)
    p.add_argument('--sink-depths-dp', type=float, nargs='+', default=[3, 4, 5, 7, 10],
                   help='Compare labelled inlet and full bottom-1dp source at identical sink depths.')
    p.add_argument('--label-bottom', type=float, default=0., help='Physical bottom used to audit inlet labels [m].')
    p.add_argument('--label-boundary-layer-dp', type=float, default=1.25)
    p.add_argument('--expected-beds', type=int, default=20)
    args = p.parse_args()
    if (args.window_dp <= 0 or args.max_start_dp < 0 or args.particle_diameter <= 0
            or args.diffusivity <= 0 or any(not 0 < b < args.window_dp/2 for b in args.boundary_bands_dp)
            or 1.0 not in args.boundary_bands_dp or not args.sink_depths_dp
            or any(not np.isfinite(d) or d <= 1 for d in args.sink_depths_dp)
            or args.label_boundary_layer_dp <= 0):
        p.error('Positive geometry required; bands must include 1 dp and be smaller than half the window')
    data = pd.read_csv(args.campaign_dataset).sort_values('seed')
    paths = sorted(args.network_root.glob('seed_*/pore_network.npz'))
    if len(data) != args.expected_beds or data.seed.duplicated().any():
        raise ValueError('Unexpected or duplicate campaign seeds')
    if len(paths) != args.expected_beds or set(map(seed_from_path, paths)) != set(data.seed):
        raise ValueError('Network and campaign seed sets must match exactly')
    if not np.isfinite(data[list(TARGETS)]).all().all():
        raise ValueError('Missing/nonfinite adsorption outcomes')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, descriptors, skipped, provenance, bridge, audits = [], [], [], [], [], []
    for path in paths:
        seed = seed_from_path(path)
        with np.load(path) as net:
            xyz, conns = net['pore_xyz'], net['throat_conns']
            length, area = net['throat_length'], net['throat_area']
            inlet, outlet = net['pore_inlet'].astype(bool), net['pore_outlet'].astype(bool)
            particles, radii = net['particle_xyz'], net['particle_radius']
        if not all(np.isfinite(a).all() for a in (xyz, length, area)) or np.any(length <= 0) or np.any(area <= 0):
            raise ValueError(f'Invalid geometry: {path}')
        g = args.diffusivity*area/length
        z0, zmax = xyz[:, 2].min(), xyz[:, 2].max()
        access, _, spans = effective_conductance(len(xyz), conns, g, inlet, xyz[:, 2] >= z0+5*args.particle_diameter)
        if not spans:
            raise ValueError(f'Nonspanning original access: {seed}')
        descriptors.append(dict(seed=seed, tortuosity_raw=tortuosity(xyz, conns, length, inlet, outlet), original_G_access_m3_s=access))
        rebuilt = reconstruct_inlet_labels(xyz, particles, radii, args.label_bottom, args.label_boundary_layer_dp)
        if not np.array_equal(rebuilt, inlet):
            raise ValueError(f'Inlet labels do not match the stated geometric rule: {seed}')
        band_source = xyz[:, 2] <= z0+args.particle_diameter
        audits.append(dict(seed=seed, label_rule_matches=True, inlet_count=int(inlet.sum()),
                           full_1dp_count=int(band_source.sum()), shared_count=int((inlet & band_source).sum()),
                           inlet_min_z_m=float(xyz[inlet, 2].min()), inlet_max_z_m=float(xyz[inlet, 2].max()),
                           band_upper_z_m=z0+args.particle_diameter))
        # Keep the source fixed while moving the sink, then replace ONLY the source.
        # Full bottom band means z <= lowest-pore z + dp, not pores exactly on a plane.
        for source_name, source in [('hull_inlet', inlet), ('full_1dp_band', xyz[:, 2] <= z0+args.particle_diameter)]:
            for sink_dp in sorted(set(args.sink_depths_dp)):
                sink = xyz[:, 2] >= z0+sink_dp*args.particle_diameter
                if not sink.any() or np.any(source & sink):
                    raise ValueError(f'Missing sink or overlapping source/sink: seed {seed}, depth {sink_dp}')
                value, _, spans = effective_conductance(len(xyz), conns, g, source,
                                                        xyz[:, 2] >= z0+sink_dp*args.particle_diameter)
                bridge.append(dict(seed=seed, source=source_name, sink_start_dp=sink_dp,
                                   conductance_m3_s=value, spanning=spans, source_count=int(source.sum())))
        provenance.append(dict(seed=seed, file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for start in np.arange(0, args.max_start_dp+1e-9, args.window_dp):
            lower, upper = z0+start*args.particle_diameter, z0+(start+args.window_dp)*args.particle_diameter
            if upper > zmax:
                skipped.append(dict(seed=seed, start_dp=start, end_dp=start+args.window_dp,
                                    reason='Full window exceeds highest retained pore', available_height_dp=(zmax-z0)/args.particle_diameter))
                continue
            for band in args.boundary_bands_dp:
                result = slab_conductance(xyz, conns, g, lower, upper, band*args.particle_diameter)
                rows.append(dict(seed=seed, start_dp=start, end_dp=start+args.window_dp,
                                 band_dp=band, lower_z_m=lower, upper_z_m=upper, **result))
        print(f'Completed seed {seed}', flush=True)
    desc = data.merge(pd.DataFrame(descriptors), on='seed', validate='one_to_one')
    np.testing.assert_allclose(desc.tortuosity_raw, desc.network_geometric_tortuosity, rtol=1e-10)
    slabs = pd.DataFrame(rows).merge(desc[['seed', *TARGETS]], on='seed', validate='many_to_one')
    correlations = []
    def correlate(frame, col, name, start=np.nan, band=np.nan):
        for target in TARGETS:
            stat = spearmanr(frame[col], frame[target])
            correlations.append(dict(descriptor=name, start_dp=start, band_dp=band, target=target,
                                     n=len(frame), spearman_rho=stat.statistic, p_unadjusted=stat.pvalue))
    for name in ['tortuosity_raw', 'original_G_access_m3_s']:
        correlate(desc, name, name)
    for (start, band), group in slabs.groupby(['start_dp', 'band_dp']):
        if len(group) != args.expected_beds:
            raise ValueError('Incomplete matched slab ensemble')
        correlate(group, 'conductance_m3_s', 'translated_slab', start, band)
    bridge = pd.DataFrame(bridge).merge(desc[['seed', *TARGETS]], on='seed', validate='many_to_one')
    bridge_corr = []
    for (source_name, sink_dp), group in bridge.groupby(['source', 'sink_start_dp']):
        for target in TARGETS:
            bridge_corr.append(dict(source=source_name, sink_start_dp=sink_dp, target=target,
                                    n=len(group), spearman_rho=spearmanr(group.conductance_m3_s, group[target]).statistic))
    bridge.to_csv(args.output/'boundary_bridge_conductances.csv', index=False)
    bridge_corr = pd.DataFrame(bridge_corr)
    bridge_corr.to_csv(args.output/'boundary_bridge_correlations.csv', index=False)
    audit = pd.DataFrame(audits)
    audit.to_csv(args.output/'inlet_selection_audit.csv', index=False)
    corr = pd.DataFrame(correlations)
    desc[['seed', 'tortuosity_raw', 'network_geometric_tortuosity', 'original_G_access_m3_s', *TARGETS]].to_csv(args.output/'tortuosity_and_outcomes.csv', index=False)
    slabs.to_csv(args.output/'slab_conductances.csv', index=False)
    corr.to_csv(args.output/'correlations.csv', index=False)
    pd.DataFrame(skipped).to_csv(args.output/'skipped_windows.csv', index=False)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        campaign_sha256=hashlib.sha256(args.campaign_dataset.read_bytes()).hexdigest(), networks=provenance), indent=2)+'\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for ax, (target, label) in zip(axes, TARGETS.items()):
        for source, color, display in [('hull_inlet', '#0065BD', 'Labelled inlet pores'),
                                        ('full_1dp_band', '#F37D20', 'All pores in bottom 1 dp')]:
            c = bridge_corr[(bridge_corr.source == source) & (bridge_corr.target == target)]
            ax.plot(c.sink_start_dp, c.spearman_rho, 'o-', color=color, label=display)
        ax.set(xlabel=r'Interior sink depth [$d_p$]', ylabel=r'Spearman $\rho_s$', title=label,
               ylim=(-1.06, 1.06), xticks=sorted(set(args.sink_depths_dp)))
    axes[0].legend(fontsize=8, loc='best')
    save_figure(fig, args.output, 'fig_inlet_selection_depth')
    summary = ['# Inlet selection versus interior-plane depth', '',
               'All correlations use the same adsorption outcomes and networks. Only the steady diagnostic changes.', '',
               '| Sink depth [dp] | Labelled inlet: uptake rho | Bottom 1dp band: uptake rho |',
               '|---|---:|---:|']
    uptake = bridge_corr[bridge_corr.target == 'final_uptake_mol_kg'].pivot(index='sink_start_dp', columns='source', values='spearman_rho')
    for depth, row in uptake.iterrows():
        summary.append(f"| {depth:g} | {row.hull_inlet:+.3f} | {row.full_1dp_band:+.3f} |")
    summary += ['', f'Geometric inlet rule exactly reproduced for {len(audit)} beds. Labelled inlet counts: {audit.inlet_count.min()}–{audit.inlet_count.max()}; full-band counts: {audit.full_1dp_count.min()}–{audit.full_1dp_count.max()}.', '',
                'Interpret depth robustness only over the tested range. The full-band source is a different boundary condition, not a translated copy of the original inlet.']
    (args.output/'inlet_selection_summary.md').write_text('\n'.join(summary)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for ax, (target, label) in zip(axes, TARGETS.items()):
        rho = spearmanr(desc.tortuosity_raw, desc[target]).statistic
        ax.scatter(desc.tortuosity_raw, desc[target], c='#0065BD', s=32)
        ax.set(xlabel=r'Global geometric tortuosity $\tau_g$', ylabel=label, title=fr'$\rho_s={rho:+.3f}$  ($n=20$)')
    save_figure(fig, args.output, 'fig_global_tortuosity')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for ax, (target, label) in zip(axes, TARGETS.items()):
        for band, color in zip(args.boundary_bands_dp, ['#789000', '#0065BD', '#F37D20']*10):
            c = corr[(corr.descriptor=='translated_slab') & (corr.target==target) & (corr.band_dp==band)]
            ax.plot(c.start_dp, c.spearman_rho, 'o-', color=color, label=f'{band:g}'+r'$d_p$ band')
        baseline = corr[(corr.descriptor=='original_G_access_m3_s') & (corr.target==target)].spearman_rho.iloc[0]
        ax.axhline(baseline, color='0.35', ls='--', label=r'Original $G_{\rm access}$')
        starts=sorted(slabs.start_dp.unique())
        ax.set(ylim=(-1.06,1.06), ylabel=r'Spearman $\rho_s$', title=label, xticks=starts,
               xticklabels=[f'{s:g}–{s+args.window_dp:g}' for s in starts], xlabel=r'Slab position [$d_p$]')
    axes[0].legend(fontsize=8, loc='best')
    save_figure(fig, args.output, 'fig_translated_slab_correlations')
    print(corr.to_string(index=False))


if __name__ == '__main__':
    main()
