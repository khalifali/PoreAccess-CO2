#!/usr/bin/env python3
"""Check completeness, settling and geometry quality of all CO2 DEM packings.

Outputs, by default:
  co2_packing_campaign_summary.csv
  co2_packing_campaign_summary.json
  co2_packing_campaign_summary.md

Mechanical acceptance and porosity matching are deliberately separate:
`analysis_ready` means that the packing is complete, settled, geometrically
valid and has the required dump files. `porosity_matched` identifies cases
within the requested tolerance of the ensemble median porosity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = [
    18427, 27183, 31991, 40213, 47837,
    52691, 61417, 69371, 74167, 81929,
    90709, 98251, 105019, 113117, 121021,
    129121, 137077, 145093, 153133, 161123,
]


def read_last_snapshot(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    last: tuple[list[str], list[list[float]]] | None = None
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF"):
            raise ValueError(f"Malformed dump near line {i + 1}: {path}")
        count = int(lines[i + 1])
        i += 2
        if i < len(lines) and lines[i].startswith("ITEM: BOX BOUNDS"):
            i += 4
        if i >= len(lines) or not (
            lines[i].startswith("ITEM: ATOMS") or lines[i].startswith("ITEM: ENTRIES")
        ):
            raise ValueError(f"Missing ATOMS/ENTRIES header: {path}")
        columns = lines[i].split()[2:]
        i += 1
        rows = [[float(value) for value in lines[i + j].split()] for j in range(count)]
        i += count
        last = columns, rows
    if last is None:
        raise ValueError(f"No snapshot found: {path}")
    return last


def read_summary(path: Path) -> dict[str, float]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Summary has no data row: {path}")
    header, values = lines[0], lines[-1]
    if len(header) != len(values):
        raise ValueError(f"Summary column mismatch: {path}")
    return {name: float(value) for name, value in zip(header, values)}


def last_matching_file(case_dir: Path, pattern: str) -> Path | None:
    files = sorted(case_dir.glob(pattern))
    if not files:
        return None
    relaxed = [path for path in files if "relaxed" in path.stem]
    return relaxed[-1] if relaxed else files[-1]


def parse_log(log_path: Path) -> tuple[int | None, str | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    dangerous = re.findall(r"Dangerous builds\s*=\s*(\d+)", text)
    wall_times = re.findall(r"Total wall time:\s*(\S+)", text)
    return (int(dangerous[-1]) if dangerous else None, wall_times[-1] if wall_times else None)


def analyse_case(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = args.root / f"co2_13x_seed_{seed}"
    result: dict[str, Any] = {
        "seed": seed,
        "case_directory": str(case_dir),
        "case_exists": case_dir.is_dir(),
        "summary_exists": False,
        "particle_dump_exists": False,
        "contact_dump_exists": False,
        "error": "",
    }
    if not case_dir.is_dir():
        result["error"] = "case directory missing"
        return result

    try:
        summary_path = case_dir / "packing_summary.dat"
        result["summary_exists"] = summary_path.is_file()
        summary = read_summary(summary_path) if summary_path.is_file() else {}

        particle_path = last_matching_file(case_dir, "particles_final*.dump")
        contact_path = last_matching_file(case_dir, "contacts_final*.dump")
        result["particle_dump_exists"] = particle_path is not None
        result["contact_dump_exists"] = contact_path is not None
        if particle_path is None:
            result["error"] = "final particle dump missing"
            return result

        columns, particles = read_last_snapshot(particle_path)
        required = ["id", "x", "y", "z", "radius", "mass", "vx", "vy", "vz"]
        index = {name: columns.index(name) for name in required}
        particle_by_id = {int(row[index["id"]]): row for row in particles}

        speeds = [
            math.sqrt(row[index["vx"]] ** 2 + row[index["vy"]] ** 2 + row[index["vz"]] ** 2)
            for row in particles
        ]
        translational_ke = sum(
            0.5 * row[index["mass"]] * speed * speed
            for row, speed in zip(particles, speeds)
        )
        tops = [row[index["z"]] + row[index["radius"]] for row in particles]
        bottoms = [row[index["z"]] - row[index["radius"]] for row in particles]
        radial_extents = [
            math.hypot(row[index["x"]], row[index["y"]]) + row[index["radius"]]
            for row in particles
        ]

        nfinal = len(particles)
        solid_volume = sum(4.0 * math.pi * row[index["radius"]] ** 3 / 3.0 for row in particles)
        bed_height = max(tops) if tops else math.nan
        bed_volume = math.pi * args.tube_radius ** 2 * bed_height
        porosity = 1.0 - solid_volume / bed_volume
        bottom_penetration = max(0.0, -min(bottoms))
        radial_penetration = max(0.0, max(radial_extents) - args.tube_radius)

        contact_count = 0
        max_overlap = math.nan
        max_overlap_normalized = math.nan
        mean_coordination = math.nan
        if contact_path is not None:
            _, contacts = read_last_snapshot(contact_path)
            overlaps: list[float] = []
            valid_pairs: set[tuple[int, int]] = set()
            for row in contacts:
                if len(row) < 6:
                    continue
                atom1, atom2 = int(row[1]), int(row[2])
                if atom1 not in particle_by_id or atom2 not in particle_by_id:
                    continue
                pair = tuple(sorted((atom1, atom2)))
                if pair in valid_pairs:
                    continue
                valid_pairs.add(pair)
                distance = row[5]
                radius1 = particle_by_id[atom1][index["radius"]]
                radius2 = particle_by_id[atom2][index["radius"]]
                overlaps.append(max(0.0, radius1 + radius2 - distance))
            contact_count = len(valid_pairs)
            mean_coordination = 2.0 * contact_count / nfinal if nfinal else math.nan
            max_overlap = max(overlaps, default=0.0)
            max_overlap_normalized = max_overlap / args.particle_diameter

        summary_ke = summary.get("kineticEnergy_J", translational_ke)
        max_speed = max(speeds, default=math.inf)
        ke_per_particle = summary_ke / nfinal if nfinal else math.inf
        count_ok = nfinal == args.target_particles
        settled = max_speed <= args.max_speed and ke_per_particle <= args.max_ke_per_particle
        overlap_ok = (
            not math.isnan(max_overlap_normalized)
            and max_overlap_normalized <= args.max_normalized_overlap
            and bottom_penetration / args.particle_diameter <= args.max_normalized_overlap
            and radial_penetration / args.particle_diameter <= args.max_normalized_overlap
        )

        log_path = args.root / "batch_logs_co2_packings" / f"seed_{seed}.log"
        dangerous_builds, wall_time = parse_log(log_path)
        neighbor_ok = dangerous_builds in (None, 0)
        outputs_ok = result["summary_exists"] and result["particle_dump_exists"] and result["contact_dump_exists"]
        analysis_ready = count_ok and settled and overlap_ok and neighbor_ok and outputs_ok

        result.update({
            "particle_dump": str(particle_path),
            "contact_dump": str(contact_path) if contact_path else "",
            "nfinal": nfinal,
            "count_ok": count_ok,
            "max_speed_m_per_s": max_speed,
            "kinetic_energy_J": summary_ke,
            "kinetic_energy_per_particle_J": ke_per_particle,
            "settled": settled,
            "bed_height_m": bed_height,
            "solid_volume_m3": solid_volume,
            "porosity": porosity,
            "bottom_penetration_m": bottom_penetration,
            "radial_penetration_m": radial_penetration,
            "contact_count": contact_count,
            "mean_coordination": mean_coordination,
            "max_particle_overlap_m": max_overlap,
            "max_normalized_overlap": max_overlap_normalized,
            "overlap_ok": overlap_ok,
            "dangerous_neighbor_builds": dangerous_builds,
            "neighbor_ok": neighbor_ok,
            "wall_time": wall_time,
            "outputs_ok": outputs_ok,
            "analysis_ready": analysis_ready,
        })
    except Exception as exc:  # preserve all other cases in the campaign report
        result["error"] = str(exc)
        result["analysis_ready"] = False
    return result


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if value is None:
        return ""
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "seed", "case_exists", "nfinal", "count_ok", "max_speed_m_per_s",
        "kinetic_energy_J", "kinetic_energy_per_particle_J", "settled",
        "bed_height_m", "porosity", "porosity_deviation_from_median",
        "porosity_matched", "contact_count", "mean_coordination",
        "max_particle_overlap_m", "max_normalized_overlap",
        "bottom_penetration_m", "radial_penetration_m", "overlap_ok",
        "dangerous_neighbor_builds", "neighbor_ok", "outputs_ok",
        "analysis_ready", "wall_time", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({key: csv_value(result.get(key)) for key in fields})


def fmt(value: Any, scientific: bool = False) -> str:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if scientific and isinstance(value, (int, float)):
        return f"{value:.3e}"
    return str(value)


def write_markdown(path: Path, results: list[dict[str, Any]], campaign: dict[str, Any]) -> None:
    lines = [
        "# CO2 packed-bed campaign summary",
        "",
        f"- Expected cases: {campaign['expected_cases']}",
        f"- Existing cases: {campaign['existing_cases']}",
        f"- Analysis-ready cases: {campaign['analysis_ready_cases']}",
        f"- Settled cases: {campaign['settled_cases']}",
        f"- Median porosity: {fmt(campaign.get('median_porosity'))}",
        f"- Mechanically complete campaign: {fmt(campaign['all_analysis_ready'])}",
        "",
        "`analysis_ready` checks completeness, settling, overlaps, neighbor-list safety and required outputs. "
        "`porosity_matched` is a separate ensemble-selection flag.",
        "",
        "| Seed | N | Settled | Max speed [m/s] | Porosity | Coordination | Max overlap/dp | Porosity matched | Analysis ready | Error |",
        "|---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['seed']} | {fmt(result.get('nfinal'))} | {fmt(result.get('settled'))} | "
            f"{fmt(result.get('max_speed_m_per_s'), True)} | {fmt(result.get('porosity'))} | "
            f"{fmt(result.get('mean_coordination'))} | {fmt(result.get('max_normalized_overlap'), True)} | "
            f"{fmt(result.get('porosity_matched'))} | {fmt(result.get('analysis_ready'))} | "
            f"{result.get('error', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", default="co2_packing_campaign_summary")
    parser.add_argument("--target-particles", type=int, default=1850)
    parser.add_argument("--particle-diameter", type=float, default=1.6e-3)
    parser.add_argument("--tube-radius", type=float, default=8.0e-3)
    parser.add_argument("--max-speed", type=float, default=1.0e-4)
    parser.add_argument("--max-ke-per-particle", type=float, default=1.0e-15)
    parser.add_argument("--max-normalized-overlap", type=float, default=1.0e-3)
    parser.add_argument("--porosity-match-tolerance", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    results = [analyse_case(seed, args) for seed in args.seeds]
    porosities = [result["porosity"] for result in results if isinstance(result.get("porosity"), float)]
    median_porosity = statistics.median(porosities) if porosities else math.nan
    for result in results:
        porosity = result.get("porosity")
        if isinstance(porosity, float) and not math.isnan(median_porosity):
            deviation = porosity - median_porosity
            result["porosity_deviation_from_median"] = deviation
            result["porosity_matched"] = abs(deviation) <= args.porosity_match_tolerance
        else:
            result["porosity_deviation_from_median"] = math.nan
            result["porosity_matched"] = False

    campaign = {
        "expected_cases": len(results),
        "existing_cases": sum(bool(result.get("case_exists")) for result in results),
        "settled_cases": sum(bool(result.get("settled")) for result in results),
        "analysis_ready_cases": sum(bool(result.get("analysis_ready")) for result in results),
        "porosity_matched_cases": sum(bool(result.get("porosity_matched")) for result in results),
        "median_porosity": median_porosity,
        "all_analysis_ready": all(bool(result.get("analysis_ready")) for result in results),
        "thresholds": {
            "target_particles": args.target_particles,
            "max_speed_m_per_s": args.max_speed,
            "max_ke_per_particle_J": args.max_ke_per_particle,
            "max_normalized_overlap": args.max_normalized_overlap,
            "porosity_match_tolerance": args.porosity_match_tolerance,
        },
    }

    prefix = args.root / args.output_prefix
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    write_csv(csv_path, results)
    json_path.write_text(
        json.dumps(json_safe({"campaign": campaign, "cases": results}), indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, results, campaign)

    print(f"Expected cases:       {campaign['expected_cases']}")
    print(f"Existing cases:       {campaign['existing_cases']}")
    print(f"Settled cases:        {campaign['settled_cases']}")
    print(f"Analysis-ready cases: {campaign['analysis_ready_cases']}")
    print(f"Median porosity:      {median_porosity:.6f}" if porosities else "Median porosity:      unavailable")
    print(f"Reports: {csv_path}, {json_path}, {md_path}")
    return 0 if campaign["all_analysis_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
