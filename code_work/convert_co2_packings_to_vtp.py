#!/usr/bin/env python3
"""Convert final LAMMPS packing/contact dumps to ParaView VTP files.

The script searches recursively below --root for directories named
co2_13x_seed_*.  Every particles_final*.dump is converted.  If a matching
contacts_final*.dump exists, a second VTP containing contact lines is written.

No VTK Python package is required; XML PolyData is written directly.
In ParaView, display particle points with the Glyph filter:
  Glyph Type = Sphere, Scale Array = diameter, Scale Factor = 1.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


def read_lammps_snapshots(path: Path) -> list[tuple[list[str], list[list[float]]]]:
    """Read custom atom or local dump snapshots from a LAMMPS text dump."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    snapshots: list[tuple[list[str], list[list[float]]]] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        i += 2  # TIMESTEP header and value
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF"):
            raise ValueError(f"Malformed dump near line {i + 1}: {path}")
        count = int(lines[i + 1].strip())
        i += 2

        # Skip BOX BOUNDS and its three rows when present.
        if i < len(lines) and lines[i].startswith("ITEM: BOX BOUNDS"):
            i += 4
        if i >= len(lines) or not (
            lines[i].startswith("ITEM: ATOMS") or lines[i].startswith("ITEM: ENTRIES")
        ):
            raise ValueError(f"Missing ATOMS/ENTRIES header near line {i + 1}: {path}")
        columns = lines[i].split()[2:]
        i += 1
        rows: list[list[float]] = []
        for _ in range(count):
            if i >= len(lines):
                raise ValueError(f"Unexpected end of dump: {path}")
            rows.append([float(value) for value in lines[i].split()])
            i += 1
        snapshots.append((columns, rows))
    if not snapshots:
        raise ValueError(f"No LAMMPS snapshots found: {path}")
    return snapshots


def column_index(columns: list[str], name: str) -> int:
    try:
        return columns.index(name)
    except ValueError as exc:
        raise ValueError(f"Required column '{name}' absent; found {columns}") from exc


def numbers(values: Iterable[float | int]) -> str:
    return " ".join(f"{value:.12g}" if isinstance(value, float) else str(value) for value in values)


def data_array(name: str, values: Iterable[float | int], vtk_type: str = "Float64", components: int = 1) -> str:
    safe_name = escape(name, {'"': '&quot;'})
    return (
        f'<DataArray type="{vtk_type}" Name="{safe_name}" '
        f'NumberOfComponents="{components}" format="ascii">\n'
        f'{numbers(values)}\n</DataArray>'
    )


def write_particles_vtp(columns: list[str], rows: list[list[float]], output: Path) -> dict[int, tuple[float, float, float]]:
    idx = {name: column_index(columns, name) for name in ("id", "type", "x", "y", "z", "radius")}
    optional = {name: columns.index(name) for name in ("vx", "vy", "vz") if name in columns}
    rows = sorted(rows, key=lambda row: int(row[idx["id"]]))
    ids = [int(row[idx["id"]]) for row in rows]
    types = [int(row[idx["type"]]) for row in rows]
    radii = [row[idx["radius"]] for row in rows]
    diameters = [2.0 * radius for radius in radii]
    coordinates = [(row[idx["x"]], row[idx["y"]], row[idx["z"]]) for row in rows]
    id_to_xyz = dict(zip(ids, coordinates))
    points = [value for xyz in coordinates for value in xyz]
    speeds = []
    if len(optional) == 3:
        speeds = [math.sqrt(row[optional["vx"]] ** 2 + row[optional["vy"]] ** 2 + row[optional["vz"]] ** 2) for row in rows]

    point_arrays = [
        data_array("particle_id", ids, "Int64"),
        data_array("particle_type", types, "Int32"),
        data_array("radius", radii),
        data_array("diameter", diameters),
    ]
    if speeds:
        point_arrays.append(data_array("speed", speeds))

    n = len(rows)
    xml = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <PointData>{''.join(point_arrays)}</PointData>
      <CellData/>
      <Points>{data_array("Points", points, components=3)}</Points>
      <Verts>
        {data_array("connectivity", range(n), "Int64")}
        {data_array("offsets", range(1, n + 1), "Int64")}
      </Verts>
    </Piece>
  </PolyData>
</VTKFile>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml, encoding="utf-8")
    return id_to_xyz


def write_contacts_vtp(rows: list[list[float]], id_to_xyz: dict[int, tuple[float, float, float]], output: Path) -> int:
    # Output order from the supplied LAMMPS input:
    # index, atom1, atom2, type1, type2, dist, dx, dy, dz, force, fx, fy, fz, p1..p4
    valid: list[list[float]] = []
    for row in rows:
        if len(row) < 13:
            continue
        atom1, atom2 = int(row[1]), int(row[2])
        if atom1 in id_to_xyz and atom2 in id_to_xyz:
            valid.append(row)

    particle_ids = sorted(id_to_xyz)
    point_index = {particle_id: index for index, particle_id in enumerate(particle_ids)}
    points = [value for particle_id in particle_ids for value in id_to_xyz[particle_id]]
    connectivity: list[int] = []
    distances: list[float] = []
    forces: list[float] = []
    pair1: list[int] = []
    pair2: list[int] = []
    for row in valid:
        atom1, atom2 = int(row[1]), int(row[2])
        connectivity.extend((point_index[atom1], point_index[atom2]))
        pair1.append(atom1)
        pair2.append(atom2)
        distances.append(row[5])
        forces.append(row[9])
    offsets = [2 * (i + 1) for i in range(len(valid))]

    xml = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{len(particle_ids)}" NumberOfVerts="0" NumberOfLines="{len(valid)}" NumberOfStrips="0" NumberOfPolys="0">
      <PointData>{data_array("particle_id", particle_ids, "Int64")}</PointData>
      <CellData>
        {data_array("particle_1", pair1, "Int64")}
        {data_array("particle_2", pair2, "Int64")}
        {data_array("center_distance", distances)}
        {data_array("contact_force", forces)}
      </CellData>
      <Points>{data_array("Points", points, components=3)}</Points>
      <Lines>
        {data_array("connectivity", connectivity, "Int64")}
        {data_array("offsets", offsets, "Int64")}
      </Lines>
    </Piece>
  </PolyData>
</VTKFile>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml, encoding="utf-8")
    return len(valid)


def seed_name(directory: Path) -> str:
    match = re.search(r"co2_13x_seed_(\d+)", directory.name)
    return f"seed_{match.group(1)}" if match else directory.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Search root (default: current directory)")
    parser.add_argument("--output", type=Path, default=Path("paraview_co2_packings"), help="Output directory")
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.root.glob("co2_13x_seed_*") if path.is_dir())
    if not case_dirs:
        print("No co2_13x_seed_* directories found.", file=sys.stderr)
        return 1

    converted = 0
    for case_dir in case_dirs:
        particle_dumps = sorted(case_dir.glob("particles_final*.dump"))
        if not particle_dumps:
            print(f"WARNING: no particles_final*.dump in {case_dir}", file=sys.stderr)
            continue
        for particle_dump in particle_dumps:
            columns, rows = read_lammps_snapshots(particle_dump)[-1]
            suffix = particle_dump.stem.removeprefix("particles_")
            case_output = args.output / seed_name(case_dir)
            particle_vtp = case_output / f"particles_{suffix}.vtp"
            id_to_xyz = write_particles_vtp(columns, rows, particle_vtp)

            contact_dump = case_dir / particle_dump.name.replace("particles_", "contacts_", 1)
            contact_count = 0
            if contact_dump.exists():
                _, contact_rows = read_lammps_snapshots(contact_dump)[-1]
                contact_vtp = case_output / f"contacts_{suffix}.vtp"
                contact_count = write_contacts_vtp(contact_rows, id_to_xyz, contact_vtp)
            else:
                print(f"WARNING: matching contact dump absent: {contact_dump}", file=sys.stderr)
            print(f"{case_dir.name}: {len(rows)} particles, {contact_count} contacts -> {case_output}")
            converted += 1

    if converted == 0:
        return 1
    print(f"Converted {converted} final packing dump set(s) into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
