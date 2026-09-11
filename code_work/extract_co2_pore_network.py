#!/usr/bin/env python3
"""Extract a Delaunay-dual pore network from DEM zeolite bead packings.

This is a geometry/QA stage, not the CO2 transport solver.  For each final
LAMMPS particle snapshot it creates pore nodes from valid tetrahedral
circumcentres and throats from shared Delaunay faces.  It also labels boundary
pores, builds the particle--pore incidence relation, allocates the analytically
computed void volume conservatively, writes ParaView files, and reports
connectivity/geometry checks.

Monodisperse note: ordinary and radical Delaunay have identical topology for
equal radii.  A future polydisperse campaign requires a regular (weighted)
triangulation; this script deliberately stops when radii are not equal.

Dependencies: numpy, scipy.  matplotlib is not required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import qmc


FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


def read_last_snapshot(path: Path) -> tuple[list[str], np.ndarray, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = None
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i + 1]); i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Malformed atom dump: {path}")
        n = int(lines[i + 1]); i += 2
        if lines[i].startswith("ITEM: BOX BOUNDS"):
            i += 4
        if not lines[i].startswith("ITEM: ATOMS"):
            raise ValueError(f"Missing ATOMS header: {path}")
        columns = lines[i].split()[2:]; i += 1
        rows = np.asarray([[float(v) for v in lines[i+j].split()] for j in range(n)])
        i += n
        result = columns, rows, step
    if result is None:
        raise ValueError(f"No snapshot found in {path}")
    return result


def particle_data(columns: list[str], rows: np.ndarray):
    def get(*names):
        for name in names:
            if name in columns:
                return rows[:, columns.index(name)]
        raise ValueError(f"Need one of {names}; found {columns}")
    ids = get("id").astype(np.int64)
    xyz = np.column_stack((get("x", "xu"), get("y", "yu"), get("z", "zu")))
    radii = get("radius")
    order = np.argsort(ids)
    return ids[order], xyz[order], radii[order]


def circumcenter_tetra(points: np.ndarray) -> tuple[np.ndarray, float]:
    p0 = points[0]
    a = 2.0*(points[1:]-p0)
    b = np.einsum("ij,ij->i", points[1:], points[1:])-np.dot(p0, p0)
    center = np.linalg.solve(a, b)
    return center, float(np.linalg.norm(center-p0))


def circumcenter_triangle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """3-D triangle circumcenter and circumradius."""
    a, b, c = points
    u, v = b-a, c-a
    n = np.cross(u, v)
    n2 = np.dot(n, n)
    if n2 <= 1e-30:
        raise np.linalg.LinAlgError("degenerate triangle")
    center = a + (np.dot(u,u)*np.cross(v,n) + np.dot(v,v)*np.cross(n,u))/(2*n2)
    return center, float(np.linalg.norm(center-a))


def circle_intersection_area(r1: np.ndarray, r2: float, d: np.ndarray) -> np.ndarray:
    r1, d = np.broadcast_arrays(np.maximum(r1, 0), np.maximum(d, 0))
    out = np.zeros_like(r1)
    contained = d <= np.abs(r2-r1)
    out[contained] = math.pi*np.minimum(r1[contained], r2)**2
    part = (~contained) & (d < r1+r2) & (r1 > 0)
    aa, dd = r1[part], d[part]
    c1 = np.clip((dd*dd+aa*aa-r2*r2)/(2*dd*aa), -1, 1)
    c2 = np.clip((dd*dd+r2*r2-aa*aa)/(2*dd*r2), -1, 1)
    h = np.maximum((-dd+aa+r2)*(dd+aa-r2)*(dd-aa+r2)*(dd+aa+r2), 0)
    out[part] = aa*aa*np.arccos(c1)+r2*r2*np.arccos(c2)-0.5*np.sqrt(h)
    return out


def solid_volume_in_cylinder(xyz, radii, tube_radius, zlo, zhi, order=48):
    nodes, weights = leggauss(order)
    total = 0.0
    for (x, y, z), r in zip(xyz, radii):
        lo, hi = max(z-r, zlo), min(z+r, zhi)
        if hi <= lo: continue
        zz = .5*(hi-lo)*nodes+.5*(hi+lo)
        rs = np.sqrt(np.maximum(r*r-(zz-z)**2, 0))
        area = circle_intersection_area(rs, tube_radius, np.full_like(rs, math.hypot(x,y)))
        total += .5*(hi-lo)*float(np.dot(weights, area))
    return total


def connected_components(n: int, edges: np.ndarray):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    labels = np.full(n, -1, dtype=int)
    sizes = []
    for start in range(n):
        if labels[start] >= 0: continue
        lab, q, count = len(sizes), deque([start]), 0
        labels[start] = lab
        while q:
            u = q.popleft(); count += 1
            for v in adj[u]:
                if labels[v] < 0: labels[v] = lab; q.append(v)
        sizes.append(count)
    return labels, np.asarray(sizes), adj


def conservative_pore_volumes(pores, particle_xyz, radii, R, zlo, zhi,
                              total_void, power, scramble_seed, chunk=200000):
    """Allocate exact total void volume using Sobol estimates as fractions."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=scramble_seed)
    u = sampler.random_base2(power)
    sample = np.column_stack((R*np.sqrt(u[:,0])*np.cos(2*math.pi*u[:,1]),
                              R*np.sqrt(u[:,0])*np.sin(2*math.pi*u[:,1]),
                              zlo+(zhi-zlo)*u[:,2]))
    ptree, vtree = cKDTree(particle_xyz), cKDTree(pores)
    counts = np.zeros(len(pores), dtype=np.int64)
    void_count = 0
    k = min(8, len(particle_xyz))
    for start in range(0, len(sample), chunk):
        s = sample[start:start+chunk]
        dist, idx = ptree.query(s, k=k)
        if k == 1: dist, idx = dist[:,None], idx[:,None]
        void = np.all(dist-radii[idx] > 0, axis=1)
        sv = s[void]
        if len(sv):
            nearest = vtree.query(sv, k=1)[1]
            counts += np.bincount(nearest, minlength=len(pores))
            void_count += len(sv)
    if void_count == 0: raise ValueError("Sobol allocation found no void samples")
    # A half-count stabilizer avoids exactly zero control volumes at tiny pores.
    weights = counts.astype(float)+0.5
    volumes = total_void*weights/weights.sum()
    sampled_eps = void_count/len(sample)
    return volumes, counts, sampled_eps, len(sample)


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def arr(name, values, dtype="Float64", components=1):
    vals = np.asarray(values).reshape(-1)
    text = " ".join(str(int(v)) if dtype.startswith("Int") else f"{v:.12g}" for v in vals)
    return (f'<DataArray type="{dtype}" Name="{escape(name)}" '
            f'NumberOfComponents="{components}" format="ascii">\n{text}\n</DataArray>')


def write_network_vtp(path, pores, edges, pore_volume, pore_radius, comp,
                      inlet, outlet, wall, throat_radius, throat_area, throat_length):
    n, m = len(pores), len(edges)
    offsets = 2*np.arange(1, m+1)
    xml = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData><Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="{m}" NumberOfStrips="0" NumberOfPolys="0">
<PointData>{arr("pore_id",np.arange(n),"Int64")}{arr("pore_volume",pore_volume)}{arr("inscribed_radius",pore_radius)}{arr("component",comp,"Int32")}{arr("inlet",inlet.astype(int),"Int32")}{arr("outlet",outlet.astype(int),"Int32")}{arr("wall",wall.astype(int),"Int32")}</PointData>
<CellData>{arr("throat_id",np.arange(m),"Int64")}{arr("throat_radius",throat_radius)}{arr("throat_area",throat_area)}{arr("throat_length",throat_length)}</CellData>
<Points>{arr("Points",pores,components=3)}</Points>
<Lines>{arr("connectivity",edges,"Int64")}{arr("offsets",offsets,"Int64")}</Lines>
</Piece></PolyData></VTKFile>\n'''
    path.write_text(xml, encoding="utf-8")


def process_case(dump: Path, outdir: Path, args) -> dict:
    cols, rows, step = read_last_snapshot(dump)
    particle_ids, xyz, radii = particle_data(cols, rows)
    rmed = float(np.median(radii)); dp = 2*rmed
    if np.ptp(radii) > args.radius_tolerance*rmed:
        raise ValueError("Polydisperse radii detected: weighted triangulation required")
    top = float(np.percentile(xyz[:,2]+radii, args.bed_top_percentile))
    zlo = args.bottom
    if top <= zlo: raise ValueError("Invalid bed domain")

    tri = Delaunay(xyz, qhull_options="Qbb Qc Qz Q12")
    simplices = tri.simplices
    old_to_new = np.full(len(simplices), -1, dtype=int)
    centers, crad, valid_simplex = [], [], []
    rejected_singular = rejected_domain = rejected_solid = 0
    for si, tet in enumerate(simplices):
        try: c, rc = circumcenter_tetra(xyz[tet])
        except np.linalg.LinAlgError:
            rejected_singular += 1; continue
        in_domain = (math.hypot(c[0],c[1]) <= args.tube_radius+args.domain_tolerance and
                     zlo-args.domain_tolerance <= c[2] <= top+args.domain_tolerance)
        if not in_domain: rejected_domain += 1; continue
        clearance = np.min(np.linalg.norm(xyz[tet]-c,axis=1)-radii[tet])
        if clearance < -args.clearance_tolerance:
            rejected_solid += 1; continue
        old_to_new[si] = len(centers)
        centers.append(c); crad.append(max(clearance, 0)); valid_simplex.append(si)
    pores = np.asarray(centers); pore_radius = np.asarray(crad)
    if len(pores) == 0: raise ValueError("No valid pore centres")

    edge_rows, edge_keys = [], set()
    incidence = set()
    inlet = np.zeros(len(pores), bool); outlet = np.zeros(len(pores), bool); wall = np.zeros(len(pores), bool)
    for new_i, si in enumerate(valid_simplex):
        tet = simplices[si]
        for pid in tet: incidence.add((new_i, int(pid)))
        for local_face, face_idx in enumerate(FACES):
            face = tet[list(face_idx)]
            neigh = tri.neighbors[si, local_face]
            if neigh >= 0 and old_to_new[neigh] >= 0:
                j = int(old_to_new[neigh]); key = tuple(sorted((new_i,j)))
                if new_i == j or key in edge_keys: continue
                try: fc, frc = circumcenter_triangle(xyz[face])
                except np.linalg.LinAlgError: continue
                aperture = float(frc-np.mean(radii[face]))
                # Never turn a closed aperture into an artificial conducting edge.
                if aperture <= args.min_throat_radius:
                    continue
                length = float(np.linalg.norm(pores[new_i]-pores[j]))
                if length <= args.min_throat_length: continue
                edge_keys.add(key)
                edge_rows.append((key[0],key[1],*face.tolist(),*fc.tolist(),length,aperture,math.pi*aperture**2))
            elif neigh < 0:
                fxyz = xyz[face]; fc = fxyz.mean(axis=0)
                # Hull faces near each physical boundary label their adjacent pore.
                if np.min(fxyz[:,2]-radii[face]) <= zlo+args.boundary_layer_dp*dp: inlet[new_i] = True
                if np.max(fxyz[:,2]+radii[face]) >= top-args.boundary_layer_dp*dp: outlet[new_i] = True
                if np.max(np.hypot(fxyz[:,0],fxyz[:,1])+radii[face]) >= args.tube_radius-args.boundary_layer_dp*dp: wall[new_i] = True
    edges = np.asarray([[r[0],r[1]] for r in edge_rows], dtype=int)
    if len(edges) == 0: raise ValueError("No valid throats")
    labels, sizes, adj = connected_components(len(pores), edges)
    largest = int(np.argmax(sizes))
    spanning_components = sorted(set(labels[inlet]) & set(labels[outlet]))

    cylinder_vol = math.pi*args.tube_radius**2*(top-zlo)
    solid_vol = solid_volume_in_cylinder(xyz,radii,args.tube_radius,zlo,top,args.quadrature_order)
    void_vol = cylinder_vol-solid_vol
    if not np.isfinite(void_vol) or not 0 < void_vol < cylinder_vol:
        raise ValueError("Invalid void volume: check particle geometry and bed bounds")
    seed_match = re.search(r"co2_13x_seed_(\d+)", str(dump))
    seed = int(seed_match.group(1)) if seed_match else args.sobol_seed
    pvol, sample_counts, sampled_eps, nsamples = conservative_pore_volumes(
        pores,xyz,radii,args.tube_radius,zlo,top,void_vol,args.sobol_power,seed)

    outdir.mkdir(parents=True, exist_ok=True)
    pore_fields = ["pore_id","x_m","y_m","z_m","volume_m3","inscribed_radius_m",
                   "component","inlet","outlet","wall","sobol_void_samples"]
    write_csv(outdir/"pores.csv",pore_fields,(
        dict(zip(pore_fields,[i,*pores[i],pvol[i],pore_radius[i],labels[i],int(inlet[i]),int(outlet[i]),int(wall[i]),sample_counts[i]]))
        for i in range(len(pores))))
    throat_fields = ["throat_id","pore_1","pore_2","particle_1","particle_2","particle_3",
                     "x_m","y_m","z_m","length_m","radius_m","area_m2"]
    write_csv(outdir/"throats.csv",throat_fields,(
        dict(zip(throat_fields,[i,*r])) for i,r in enumerate(edge_rows)))
    inc_fields=["pore_id","particle_index","particle_id"]
    write_csv(outdir/"particle_pore_incidence.csv",inc_fields,(
        {"pore_id":p,"particle_index":q,"particle_id":int(particle_ids[q])} for p,q in sorted(incidence)))
    throat_radius=np.asarray([r[-2] for r in edge_rows]); throat_area=np.asarray([r[-1] for r in edge_rows]); throat_len=np.asarray([r[-3] for r in edge_rows])
    write_network_vtp(outdir/"pore_network.vtp",pores,edges,pvol,pore_radius,labels,inlet,outlet,wall,throat_radius,throat_area,throat_len)
    np.savez_compressed(outdir/"pore_network.npz",pore_xyz=pores,pore_volume=pvol,
        pore_inscribed_radius=pore_radius,pore_component=labels,pore_inlet=inlet,
        pore_outlet=outlet,pore_wall=wall,throat_conns=edges,
        throat_length=throat_len,throat_radius=throat_radius,throat_area=throat_area,
        particle_id=particle_ids,particle_xyz=xyz,particle_radius=radii)

    qa = {
      "source_dump":str(dump),"source_timestep":step,"seed":seed,"particles":len(xyz),
      "bed_top_percentile":args.bed_top_percentile,"bed_top_m":top,"tube_radius_m":args.tube_radius,
      "delaunay_tetrahedra":len(simplices),"valid_pores":len(pores),"throats":len(edges),
      "rejected_singular":rejected_singular,"rejected_outside_domain":rejected_domain,
      "rejected_inside_solid":rejected_solid,"components":len(sizes),
      "largest_component_pores":int(sizes[largest]),"largest_component_fraction":float(sizes[largest]/len(pores)),
      "inlet_pores":int(inlet.sum()),"outlet_pores":int(outlet.sum()),"wall_pores":int(wall.sum()),
      "spanning_component_ids":[int(v) for v in spanning_components],
      "cylinder_volume_m3":cylinder_vol,"solid_volume_m3":solid_vol,"void_volume_m3":void_vol,
      "analytic_porosity":void_vol/cylinder_vol,"allocated_pore_volume_m3":float(pvol.sum()),
      "volume_relative_error":float(abs(pvol.sum()-void_vol)/void_vol),
      "sobol_power":args.sobol_power,"sobol_samples":nsamples,"sobol_sampled_porosity":sampled_eps,
      "min_pore_volume_m3":float(pvol.min()),"min_throat_radius_m":float(throat_radius.min()),
      "min_throat_length_m":float(throat_len.min()),
    }
    qa["checks"]={
      "finite_geometry":bool(np.all(np.isfinite(pores)) and np.all(np.isfinite(pvol))),
      "positive_pore_volumes":bool(np.all(pvol>0)),"positive_throat_lengths":bool(np.all(throat_len>0)),
      "unique_throats":len(edge_keys)==len(edges),"has_inlet":bool(inlet.any()),"has_outlet":bool(outlet.any()),
      "inlet_outlet_connected":bool(spanning_components),"largest_component_fraction_ok":bool(sizes[largest]/len(pores)>=args.min_largest_component_fraction),
      "volume_conserved":bool(abs(pvol.sum()-void_vol)/void_vol<=1e-12),
    }
    qa["network_ready_for_transport"] = all(qa["checks"].values())
    (outdir/"network_qa.json").write_text(json.dumps(qa,indent=2)+"\n",encoding="utf-8")
    print(f"{dump.parent.name}: pores={len(pores)}, throats={len(edges)}, components={len(sizes)}, "
          f"spanning={bool(spanning_components)}, eps={qa['analytic_porosity']:.6f}, ready={qa['network_ready_for_transport']}")
    return qa


def arguments():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--case",type=Path,help="one co2_13x_seed_* directory or particle dump")
    src.add_argument("--all-cases",action="store_true")
    p.add_argument("--root",type=Path,default=Path(".")); p.add_argument("--pattern",default="particles_final*.dump")
    p.add_argument("--output-root",type=Path,default=Path("co2_pore_networks_power22"))
    p.add_argument("--tube-radius",type=float,default=.008); p.add_argument("--bottom",type=float,default=0.)
    p.add_argument("--bed-top-percentile",type=float,default=99.)
    p.add_argument("--boundary-layer-dp",type=float,default=1.25)
    p.add_argument("--sobol-power",type=int,default=22,help="allocation samples = 2**power")
    p.add_argument("--sobol-seed",type=int,default=18427); p.add_argument("--quadrature-order",type=int,default=48)
    p.add_argument("--radius-tolerance",type=float,default=1e-6)
    p.add_argument("--domain-tolerance",type=float,default=2e-6)
    p.add_argument("--clearance-tolerance",type=float,default=2e-6)
    p.add_argument("--min-throat-radius",type=float,default=1e-9)
    p.add_argument("--min-throat-length",type=float,default=1e-12)
    p.add_argument("--min-largest-component-fraction",type=float,default=.98)
    return p.parse_args()


def main():
    a=arguments()
    if a.case:
        dumps=sorted(a.case.glob(a.pattern)) if a.case.is_dir() else [a.case]
    else:
        dumps=sorted(a.root.glob(f"co2_13x_seed_*/{a.pattern}"))
    if not dumps: sys.exit("No particle dumps found")
    reports=[]
    for dump in dumps:
        match=re.search(r"co2_13x_seed_(\d+)",str(dump)); label=f"seed_{match.group(1)}" if match else dump.parent.name
        try: reports.append(process_case(dump,a.output_root/label,a))
        except Exception as exc:
            print(f"ERROR {dump}: {exc}",file=sys.stderr)
            reports.append({"source_dump":str(dump),"error":str(exc),"network_ready_for_transport":False})
    (a.output_root/"campaign_network_qa.json").parent.mkdir(parents=True,exist_ok=True)
    (a.output_root/"campaign_network_qa.json").write_text(json.dumps(reports,indent=2)+"\n")
    ready=sum(bool(r.get("network_ready_for_transport")) for r in reports)
    print(f"Campaign extraction: {ready}/{len(reports)} networks pass current QA")
    return 0 if ready==len(reports) else 2


if __name__=="__main__":
    raise SystemExit(main())
