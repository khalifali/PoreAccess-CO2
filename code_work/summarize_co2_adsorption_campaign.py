#!/usr/bin/env python3
"""Summarize a campaign of CO2 pore-network adsorption simulations.

The script searches below --results-root for result directories containing
``run_metadata.json``, ``timeseries.csv`` and, when available,
``final_state.npz``.  It performs numerical/physical QA and extracts one row
per adsorption run for later statistics or machine learning.

Main outputs below --output-dir
--------------------------------
co2_adsorption_campaign_dataset.csv
    One row per case: QA, solver parameters, kinetic targets and structural
    descriptors.
co2_adsorption_campaign_summary.json
    Machine-readable campaign settings, checks and all case records.
co2_adsorption_campaign_summary.md
    Short human-readable QA report.
co2_adsorption_campaign_statistics.csv
    Campaign statistics for important performance targets.
co2_adsorption_axial_loading_profiles.csv
    Final particle loading in axial bins (written when final state and network
    geometry are available).
co2_adsorption_campaign_overview.png
    QA and structure--performance overview plot (unless --no-plots is used).

Percent times t10, t50, t90 and t95 are relative to the uptake reached at the
end of each simulation, not necessarily to true thermodynamic equilibrium.
The late-uptake fraction reports whether this distinction is important.

Example
-------
python3 summarize_co2_adsorption_campaign.py \
  --results-root co2_adsorption_campaign \
  --output-dir co2_adsorption_campaign_analysis \
  --expected-cases 20 \
  --porosity-summary porosity_analysis/packing_porosity_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_TIMESERIES_COLUMNS = (
    "time_s",
    "mean_gas_concentration_mol_m3",
    "gas_inventory_mol",
    "adsorbed_inventory_mol",
    "total_inventory_mol",
    "cumulative_reservoir_input_mol",
    "mass_balance_residual_mol",
    "relative_mass_balance_error",
    "mean_loading_mol_kg",
)

PERFORMANCE_FIELDS = (
    "final_adsorbed_inventory_mol",
    "final_uptake_mol_kg",
    "t10_final_uptake_s",
    "t50_final_uptake_s",
    "t90_final_uptake_s",
    "t95_final_uptake_s",
    "maximum_adsorption_rate_mol_s",
    "time_of_maximum_rate_s",
    "normalized_uptake_auc",
    "late_uptake_fraction",
    "end_relative_uptake_rate_per_s",
    "particle_loading_cv",
    "underutilized_particle_fraction",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-root", type=Path, default=Path("."),
                   help="root containing the adsorption result directories")
    p.add_argument("--result-pattern", default="**/run_metadata.json",
                   help="glob below results-root used to discover runs")
    p.add_argument("--output-dir", type=Path,
                   default=Path("co2_adsorption_campaign_analysis"))
    p.add_argument("--expected-cases", type=int, default=20)
    p.add_argument("--mass-balance-tolerance", type=float, default=1e-8,
                   help="maximum accepted relative mass-balance error")
    p.add_argument("--concentration-tolerance", type=float, default=1e-8,
                   help="allowed negative concentration/loading roundoff")
    p.add_argument("--completion-time-tolerance", type=float, default=1e-8,
                   help="relative tolerance when comparing final time with t_end")
    p.add_argument("--late-window-fraction", type=float, default=0.10,
                   help="last fraction of simulated time used for convergence checks")
    p.add_argument("--late-uptake-tolerance", type=float, default=0.02,
                   help="maximum accepted uptake increase during the late window")
    p.add_argument("--end-relative-rate-tolerance", type=float, default=1e-5,
                   help="maximum accepted (dn/dt)/n at the end [1/s]")
    p.add_argument("--underutilized-threshold", type=float, default=0.50,
                   help="particle is underutilized below this fraction of inlet q*")
    p.add_argument("--axial-bins", type=int, default=10)
    p.add_argument("--porosity-summary", type=Path,
                   help="optional packing_porosity_summary.csv")
    p.add_argument("--packing-summary", type=Path,
                   help="optional co2_packing_campaign_summary.csv")
    p.add_argument("--network-root", type=Path,
                   help="optional root containing seed_*/pore_network.npz; normally read from metadata")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="return a nonzero code unless the complete campaign passes QA")
    a = p.parse_args()
    if a.expected_cases <= 0 or a.axial_bins <= 0:
        p.error("--expected-cases and --axial-bins must be positive")
    if not 0 < a.late_window_fraction < 1:
        p.error("--late-window-fraction must lie between zero and one")
    if not 0 <= a.underutilized_threshold <= 1:
        p.error("--underutilized-threshold must lie between zero and one")
    return a


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def seed_from_text(text: str) -> int | None:
    matches = re.findall(r"(?:seed[_-]?)(\d+)", text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        missing = set(REQUIRED_TIMESERIES_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"timeseries is missing columns {sorted(missing)}")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row_number, row in enumerate(reader, start=2):
            for name in reader.fieldnames:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"non-numeric value in {path}, row {row_number}, column {name}"
                    ) from exc
    if not columns or len(next(iter(columns.values()))) < 2:
        raise ValueError(f"timeseries must contain at least two rows: {path}")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def crossing_time(time: np.ndarray, values: np.ndarray, fraction: float) -> float:
    """First linearly interpolated crossing of a fraction of final change."""
    start, final = float(values[0]), float(values[-1])
    target = start + fraction * (final - start)
    if final >= start:
        monotone = np.maximum.accumulate(values)
        hit = np.flatnonzero(monotone >= target)
    else:
        monotone = np.minimum.accumulate(values)
        hit = np.flatnonzero(monotone <= target)
    if len(hit) == 0:
        return math.nan
    i = int(hit[0])
    if i == 0:
        return float(time[0])
    y0, y1 = monotone[i - 1], monotone[i]
    if y1 == y0:
        return float(time[i])
    weight = (target - y0) / (y1 - y0)
    return float(time[i - 1] + weight * (time[i] - time[i - 1]))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) <= 0:
        return math.nan
    xc = x - np.mean(x)
    return float(np.dot(xc, y - np.mean(y)) / np.dot(xc, xc))


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_table_by_seed(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"summary file does not exist: {path}")
    result: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                result[int(float(row["seed"]))] = row
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"summary file requires a numeric seed column: {path}")
    return result


def resolve_network_dir(metadata: dict[str, Any], seed: int | None,
                        metadata_path: Path, args: argparse.Namespace) -> Path | None:
    candidates: list[Path] = []
    if args.network_root is not None and seed is not None:
        candidates.append(args.network_root / f"seed_{seed}")
    network_text = metadata.get("network") or metadata.get("parameters", {}).get("network")
    if network_text:
        raw = Path(str(network_text))
        candidates.extend([raw, metadata_path.parent / raw, args.results_root / raw])
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "pore_network.npz").is_file():
            return candidate.resolve()
    return None


def add_external_descriptors(record: dict[str, Any], seed: int | None,
                             porosity: dict[int, dict[str, str]],
                             packing: dict[int, dict[str, str]]) -> None:
    if seed is None:
        return
    prow = porosity.get(seed, {})
    aliases = {
        "bulk_porosity": ("bulk_porosity", "porosity"),
        "interior_porosity": ("interior_porosity",),
        "bed_top_m": ("bed_top_m", "bed_height_m"),
    }
    for output, names in aliases.items():
        for name in names:
            if name in prow and prow[name] != "":
                record[output] = finite_float(prow[name])
                break
    packrow = packing.get(seed, {})
    for output, names in {
        "packing_porosity": ("porosity",),
        "packing_mean_coordination": ("mean_coordination",),
        "packing_contact_count": ("contact_count",),
        "packing_bed_height_m": ("bed_height_m",),
        "packing_analysis_ready": ("analysis_ready",),
    }.items():
        for name in names:
            if name not in packrow or packrow[name] == "":
                continue
            text = packrow[name].strip().lower()
            record[output] = text in ("true", "1", "yes") if output.endswith("ready") else finite_float(packrow[name])
            break


def network_descriptors(network_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    npz_path = network_dir / "pore_network.npz"
    with np.load(npz_path) as z:
        required = ("pore_xyz", "pore_volume", "pore_inlet", "pore_outlet",
                    "throat_conns", "throat_radius", "throat_length",
                    "particle_xyz", "particle_radius")
        missing = [name for name in required if name not in z]
        if missing:
            raise ValueError(f"network NPZ is missing {missing}")
        pore_xyz = np.asarray(z["pore_xyz"], float)
        pvol = np.asarray(z["pore_volume"], float)
        inlet = np.asarray(z["pore_inlet"], bool)
        outlet = np.asarray(z["pore_outlet"], bool)
        conns = np.asarray(z["throat_conns"], int)
        tr = np.asarray(z["throat_radius"], float)
        tl = np.asarray(z["throat_length"], float)
        particle_xyz = np.asarray(z["particle_xyz"], float)
        particle_radius = np.asarray(z["particle_radius"], float)

    degree = np.bincount(conns.ravel(), minlength=len(pvol))
    values: dict[str, Any] = {
        "network_directory": str(network_dir),
        "network_pore_count": len(pvol),
        "network_throat_count": len(conns),
        "network_total_pore_volume_m3": float(pvol.sum()),
        "network_mean_pore_volume_m3": float(pvol.mean()),
        "network_median_pore_volume_m3": float(np.median(pvol)),
        "network_pore_volume_cv": float(pvol.std() / pvol.mean()),
        "network_mean_throat_radius_m": float(tr.mean()),
        "network_median_throat_radius_m": float(np.median(tr)),
        "network_minimum_throat_radius_m": float(tr.min()),
        "network_throat_radius_p10_m": float(np.percentile(tr, 10)),
        "network_mean_throat_length_m": float(tl.mean()),
        "network_mean_degree": float(degree.mean()),
        "network_degree_cv": float(degree.std() / degree.mean()),
        "network_inlet_pores": int(inlet.sum()),
        "network_outlet_pores": int(outlet.sum()),
    }
    qa_path = network_dir / "network_qa.json"
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        values["network_ready_for_transport"] = bool(qa.get("network_ready_for_transport", False))
        values["network_analytic_porosity"] = finite_float(qa.get("analytic_porosity"))
        values["network_largest_component_fraction"] = finite_float(qa.get("largest_component_fraction"))

    # Shortest geometric inlet-to-outlet network path and a simple tortuosity ratio.
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        rows = np.r_[conns[:, 0], conns[:, 1]]
        cols = np.r_[conns[:, 1], conns[:, 0]]
        graph = coo_matrix((np.r_[tl, tl], (rows, cols)), shape=(len(pvol), len(pvol))).tocsr()
        distances = dijkstra(graph, directed=False, indices=np.flatnonzero(inlet), min_only=True)
        path = float(np.min(distances[outlet]))
        axial_span = float(np.max(pore_xyz[outlet, 2]) - np.min(pore_xyz[inlet, 2]))
        values["network_shortest_inlet_outlet_path_m"] = path
        values["network_geometric_tortuosity"] = path / axial_span if axial_span > 0 else math.nan
    except Exception as exc:
        values["network_path_warning"] = str(exc)

    values["_particle_xyz"] = particle_xyz
    values["_particle_radius"] = particle_radius
    return values


def particle_loading_metrics(final_state: Path, network: dict[str, Any],
                             parameters: dict[str, Any], args: argparse.Namespace,
                             seed: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(final_state) as z:
        if "particle_loading" not in z:
            raise ValueError("final_state.npz has no particle_loading")
        loading = np.asarray(z["particle_loading"], float)
    xyz = np.asarray(network.pop("_particle_xyz"))
    radius = np.asarray(network.pop("_particle_radius"))
    if len(loading) != len(xyz):
        raise ValueError("particle loading and network particle counts differ")
    mean = float(np.mean(loading))
    qsat = finite_float(parameters.get("qsat"))
    b_pa = finite_float(parameters.get("b_pa"))
    exponent = finite_float(parameters.get("toth_exponent"), 1.0)
    cin = finite_float(parameters.get("inlet_concentration"))
    temperature = finite_float(parameters.get("temperature"))
    rg = 8.31446261815324
    pressure = max(cin, 0.0) * rg * temperature
    bp = max(b_pa * pressure, 0.0)
    qeq_inlet = qsat * bp / (1.0 + bp ** exponent) ** (1.0 / exponent)
    threshold = args.underutilized_threshold * qeq_inlet
    metrics = {
        "particle_loading_mean_unweighted_mol_kg": mean,
        "particle_loading_std_mol_kg": float(np.std(loading)),
        "particle_loading_cv": float(np.std(loading) / mean) if mean > 0 else math.nan,
        "particle_loading_min_mol_kg": float(np.min(loading)),
        "particle_loading_max_mol_kg": float(np.max(loading)),
        "inlet_equilibrium_loading_mol_kg": qeq_inlet,
        "underutilized_loading_threshold_mol_kg": threshold,
        "underutilized_particle_fraction": float(np.mean(loading < threshold)),
    }

    z = xyz[:, 2]
    edges = np.linspace(float(np.min(z - radius)), float(np.max(z + radius)), args.axial_bins + 1)
    bins = np.clip(np.digitize(z, edges) - 1, 0, args.axial_bins - 1)
    profiles: list[dict[str, Any]] = []
    bin_means = []
    for k in range(args.axial_bins):
        mask = bins == k
        value = float(np.mean(loading[mask])) if np.any(mask) else math.nan
        bin_means.append(value)
        profiles.append({
            "seed": seed,
            "axial_bin": k,
            "z_low_m": edges[k],
            "z_high_m": edges[k + 1],
            "particle_count": int(mask.sum()),
            "mean_loading_mol_kg": value,
            "mean_loading_over_inlet_equilibrium": value / qeq_inlet if qeq_inlet > 0 else math.nan,
        })
    valid = np.asarray([x for x in bin_means if math.isfinite(x)])
    if len(valid) >= 2 and mean > 0:
        metrics["axial_loading_bottom_minus_top_over_mean"] = float((valid[0] - valid[-1]) / mean)
        metrics["axial_loading_bin_cv"] = float(valid.std() / valid.mean()) if valid.mean() > 0 else math.nan
    return metrics, profiles


def analyse_run(metadata_path: Path, args: argparse.Namespace,
                porosity: dict[int, dict[str, str]],
                packing: dict[int, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = metadata_path.parent
    record: dict[str, Any] = {
        "result_directory": str(run_dir),
        "metadata_file": str(metadata_path),
        "error": "",
    }
    profiles: list[dict[str, Any]] = []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parameters = metadata.get("parameters", {})
        seed = seed_from_text(str(metadata.get("network", "")))
        if seed is None:
            seed = seed_from_text(str(run_dir))
        record["seed"] = seed
        record["mode"] = metadata.get("mode")
        record["solver_success"] = bool(metadata.get("solver_success", False))
        record["physical_bounds_ok_metadata"] = bool(metadata.get("physical_bounds_ok", False))
        for name in ("pore_count", "throat_count", "particle_count", "rhs_evaluations"):
            record[name] = metadata.get(name)
        for name in ("diffusivity", "inlet_concentration", "initial_concentration",
                     "temperature", "particle_density", "qsat", "b_pa",
                     "toth_exponent", "k_ldf", "initial_loading", "t_end",
                     "rtol", "atol", "max_step"):
            record[name] = parameters.get(name)

        timeseries_path = run_dir / "timeseries.csv"
        if not timeseries_path.is_file():
            raise FileNotFoundError(f"missing {timeseries_path}")
        ts = read_numeric_csv(timeseries_path)
        time = ts["time_s"]
        uptake = ts["adsorbed_inventory_mol"]
        if not np.all(np.isfinite(np.concatenate(list(ts.values())))):
            raise ValueError("timeseries contains NaN or infinity")
        if np.any(np.diff(time) <= 0):
            raise ValueError("timeseries time must be strictly increasing")

        t_end_requested = finite_float(parameters.get("t_end"))
        record["final_time_s"] = float(time[-1])
        record["completed_to_t_end"] = bool(
            math.isfinite(t_end_requested)
            and abs(time[-1] - t_end_requested) <= args.completion_time_tolerance * max(t_end_requested, 1.0)
        )
        record["final_gas_inventory_mol"] = float(ts["gas_inventory_mol"][-1])
        record["final_adsorbed_inventory_mol"] = float(uptake[-1])
        record["final_total_inventory_mol"] = float(ts["total_inventory_mol"][-1])
        record["final_reservoir_input_mol"] = float(ts["cumulative_reservoir_input_mol"][-1])
        record["final_mean_gas_concentration_mol_m3"] = float(ts["mean_gas_concentration_mol_m3"][-1])
        record["final_uptake_mol_kg"] = float(ts["mean_loading_mol_kg"][-1])
        record["final_relative_mass_balance_error"] = float(ts["relative_mass_balance_error"][-1])
        record["maximum_relative_mass_balance_error"] = float(np.max(np.abs(ts["relative_mass_balance_error"])))
        record["maximum_absolute_mass_balance_residual_mol"] = float(np.max(np.abs(ts["mass_balance_residual_mol"])))
        record["minimum_mean_gas_concentration_mol_m3"] = float(np.min(ts["mean_gas_concentration_mol_m3"]))
        record["minimum_mean_loading_mol_kg"] = float(np.min(ts["mean_loading_mol_kg"]))
        record["mass_balance_ok"] = record["maximum_relative_mass_balance_error"] <= args.mass_balance_tolerance
        record["timeseries_mean_bounds_ok"] = bool(
            record["minimum_mean_gas_concentration_mol_m3"] >= -args.concentration_tolerance
            and record["minimum_mean_loading_mol_kg"] >= -args.concentration_tolerance
        )

        for fraction in (0.10, 0.50, 0.90, 0.95):
            record[f"t{int(100*fraction):02d}_final_uptake_s"] = crossing_time(time, uptake, fraction)
        dt = np.diff(time)
        segment_rate = np.diff(uptake) / dt
        imax = int(np.argmax(segment_rate))
        record["maximum_adsorption_rate_mol_s"] = float(segment_rate[imax])
        record["time_of_maximum_rate_s"] = float(0.5 * (time[imax] + time[imax + 1]))
        record["uptake_monotonic_ok"] = bool(
            np.min(np.diff(uptake)) >= -max(abs(uptake[-1]), 1e-30) * 1e-8
        )
        final_change = float(uptake[-1] - uptake[0])
        record["normalized_uptake_auc"] = (
            float(np.trapezoid(uptake - uptake[0], time) / (final_change * time[-1]))
            if final_change > 0 and time[-1] > 0 else math.nan
        )
        late_start = (1.0 - args.late_window_fraction) * time[-1]
        late_idx = np.flatnonzero(time >= late_start)
        if len(late_idx) < 2:
            late_idx = np.arange(max(0, len(time) - 2), len(time))
        late_initial = float(np.interp(late_start, time, uptake))
        record["late_uptake_fraction"] = (
            float((uptake[-1] - late_initial) / uptake[-1]) if uptake[-1] > 0 else math.nan
        )
        end_slope = linear_slope(time[late_idx], uptake[late_idx])
        record["end_adsorption_rate_mol_s"] = end_slope
        record["end_relative_uptake_rate_per_s"] = end_slope / uptake[-1] if uptake[-1] > 0 else math.nan
        record["practical_equilibrium_reached"] = bool(
            record["late_uptake_fraction"] <= args.late_uptake_tolerance
            and abs(record["end_relative_uptake_rate_per_s"]) <= args.end_relative_rate_tolerance
        )

        add_external_descriptors(record, seed, porosity, packing)
        network_dir = resolve_network_dir(metadata, seed, metadata_path, args)
        if network_dir is not None:
            ndesc = network_descriptors(network_dir, record)
            final_state = run_dir / "final_state.npz"
            if final_state.is_file():
                loading_metrics, profiles = particle_loading_metrics(
                    final_state, ndesc, parameters, args, seed
                )
                record.update(loading_metrics)
            else:
                ndesc.pop("_particle_xyz", None)
                ndesc.pop("_particle_radius", None)
                record["spatial_metrics_warning"] = "final_state.npz missing"
            record.update(ndesc)
        else:
            record["network_warning"] = "network directory could not be resolved"

        record["case_qa_pass"] = bool(
            record.get("mode") == "adsorption"
            and record["solver_success"]
            and record["physical_bounds_ok_metadata"]
            and record["completed_to_t_end"]
            and record["mass_balance_ok"]
            and record["timeseries_mean_bounds_ok"]
            and record["uptake_monotonic_ok"]
            and record.get("network_ready_for_transport", True)
        )
    except Exception as exc:
        record["error"] = str(exc)
        record["case_qa_pass"] = False
    return record, profiles


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        preferred = [
            "seed", "case_qa_pass", "solver_success", "physical_bounds_ok_metadata",
            "completed_to_t_end", "mass_balance_ok", "timeseries_mean_bounds_ok",
            "uptake_monotonic_ok", "practical_equilibrium_reached", "error",
        ]
        all_fields = set().union(*(row.keys() for row in rows))
        fields = [x for x in preferred if x in all_fields] + sorted(all_fields - set(preferred))
    field_list = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if json_safe(row.get(k)) is None else json_safe(row.get(k)) for k in field_list})


def campaign_statistics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for field in PERFORMANCE_FIELDS:
        values = np.asarray([
            finite_float(row.get(field)) for row in records
            if math.isfinite(finite_float(row.get(field)))
        ])
        if len(values) == 0:
            continue
        mean = float(values.mean())
        rows.append({
            "metric": field,
            "count": len(values),
            "mean": mean,
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "coefficient_of_variation": float(values.std(ddof=1) / abs(mean)) if len(values) > 1 and mean != 0 else 0.0,
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        })
    return rows


def parameter_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("diffusivity", "inlet_concentration", "initial_concentration", "temperature",
              "particle_density", "qsat", "b_pa", "toth_exponent", "k_ldf",
              "initial_loading", "t_end", "rtol", "atol")
    report = {}
    for field in fields:
        values = [row.get(field) for row in records if row.get(field) is not None]
        normalized = {str(v) for v in values}
        report[field] = {"consistent": len(normalized) <= 1,
                         "values": sorted(normalized), "missing_cases": len(records) - len(values)}
    return report


def write_markdown(path: Path, records: list[dict[str, Any]], campaign: dict[str, Any]) -> None:
    lines = [
        "# CO2 adsorption campaign summary", "",
        f"- Discovered runs: {campaign['discovered_runs']}",
        f"- Unique seeds: {campaign['unique_seeds']}",
        f"- Expected cases: {campaign['expected_cases']}",
        f"- Cases passing numerical and physical QA: {campaign['qa_pass_cases']}",
        f"- Cases meeting the practical-equilibrium criterion: {campaign['practical_equilibrium_cases']}",
        f"- Duplicate seeds: {campaign['duplicate_seeds'] or 'none'}", "",
        "The percentage times are measured relative to each run's final uptake. "
        "A run can pass numerical QA without having reached practical equilibrium.", "",
        "| Seed | QA | Final uptake [mol/kg] | t50 [s] | t90 [s] | Late uptake [%] | Equilibrium | Error |",
        "|---:|:---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in sorted(records, key=lambda x: (x.get("seed") is None, x.get("seed") or 0)):
        late = finite_float(row.get("late_uptake_fraction"))
        lines.append(
            f"| {row.get('seed', '--')} | {'yes' if row.get('case_qa_pass') else 'no'} | "
            f"{finite_float(row.get('final_uptake_mol_kg')):.6g} | "
            f"{finite_float(row.get('t50_final_uptake_s')):.6g} | "
            f"{finite_float(row.get('t90_final_uptake_s')):.6g} | "
            f"{100*late:.3g} | {'yes' if row.get('practical_equilibrium_reached') else 'no'} | "
            f"{row.get('error', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, records: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; plot was skipped"
    ordered = sorted(
        [row for row in records if row.get("seed") is not None],
        key=lambda row: int(row["seed"]),
    )
    if not ordered:
        return "no seeded records were available; plot was skipped"
    seed_labels = [str(row["seed"]) for row in ordered]
    uptake = [finite_float(row.get("final_uptake_mol_kg")) for row in ordered]
    t90 = [finite_float(row.get("t90_final_uptake_s")) for row in ordered]
    late = [100 * finite_float(row.get("late_uptake_fraction")) for row in ordered]
    porosity = [finite_float(row.get("bulk_porosity", row.get("network_analytic_porosity"))) for row in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(ordered))
    axes[0, 0].bar(x, uptake, color="#2a788e")
    axes[0, 0].set_ylabel("Final uptake [mol kg$^{-1}$]")
    axes[0, 0].set_xticks(x, seed_labels, rotation=60, ha="right", fontsize=8)
    axes[0, 1].bar(x, t90, color="#7ad151")
    axes[0, 1].set_ylabel("$t_{90}$ relative to final uptake [s]")
    axes[0, 1].set_xticks(x, seed_labels, rotation=60, ha="right", fontsize=8)
    colors = ["#2ca25f" if row.get("case_qa_pass") else "#de2d26" for row in ordered]
    axes[1, 0].bar(x, late, color=colors)
    axes[1, 0].set_ylabel("Uptake gained in final 10% [%]")
    axes[1, 0].set_xticks(x, seed_labels, rotation=60, ha="right", fontsize=8)
    mask = np.isfinite(porosity) & np.isfinite(t90)
    axes[1, 1].scatter(np.asarray(porosity)[mask], np.asarray(t90)[mask], color="#414487")
    axes[1, 1].set_xlabel("Bulk/network porosity")
    axes[1, 1].set_ylabel("$t_{90}$ [s]")
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle("CO$_2$ adsorption campaign overview")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return None


def main() -> int:
    args = parse_args()
    metadata_files = sorted(args.results_root.glob(args.result_pattern))
    # Avoid reading summaries generated below the selected output directory.
    metadata_files = [p for p in metadata_files if args.output_dir.resolve() not in p.resolve().parents]
    if not metadata_files:
        sys.exit(f"No run_metadata.json files found below {args.results_root}")

    porosity = read_table_by_seed(args.porosity_summary)
    packing = read_table_by_seed(args.packing_summary)
    records: list[dict[str, Any]] = []
    axial_profiles: list[dict[str, Any]] = []
    for metadata_path in metadata_files:
        record, profiles = analyse_run(metadata_path, args, porosity, packing)
        records.append(record)
        axial_profiles.extend(profiles)
        status = "PASS" if record.get("case_qa_pass") else "FAIL"
        print(f"seed {record.get('seed', '?')}: {status}; "
              f"uptake={finite_float(record.get('final_uptake_mol_kg')):.6g} mol/kg; "
              f"t90={finite_float(record.get('t90_final_uptake_s')):.6g} s")

    seeds = [row.get("seed") for row in records if row.get("seed") is not None]
    counts = Counter(seeds)
    duplicate_seeds = sorted(seed for seed, count in counts.items() if count > 1)
    consistency = parameter_consistency(records)
    inconsistent = sorted(name for name, check in consistency.items() if not check["consistent"])
    campaign = {
        "results_root": str(args.results_root),
        "discovered_runs": len(records),
        "unique_seeds": len(set(seeds)),
        "expected_cases": args.expected_cases,
        "qa_pass_cases": sum(bool(row.get("case_qa_pass")) for row in records),
        "practical_equilibrium_cases": sum(bool(row.get("practical_equilibrium_reached")) for row in records),
        "duplicate_seeds": duplicate_seeds,
        "unidentified_seed_runs": sum(row.get("seed") is None for row in records),
        "inconsistent_parameters": inconsistent,
        "parameter_consistency": consistency,
        "thresholds": {
            "mass_balance_tolerance": args.mass_balance_tolerance,
            "concentration_tolerance": args.concentration_tolerance,
            "late_window_fraction": args.late_window_fraction,
            "late_uptake_tolerance": args.late_uptake_tolerance,
            "end_relative_rate_tolerance_per_s": args.end_relative_rate_tolerance,
            "underutilized_threshold_of_inlet_equilibrium": args.underutilized_threshold,
        },
    }
    campaign["complete_and_qa_pass"] = bool(
        len(records) == args.expected_cases
        and len(set(seeds)) == args.expected_cases
        and not duplicate_seeds
        and not inconsistent
        and all(bool(row.get("case_qa_pass")) for row in records)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "co2_adsorption_campaign_dataset.csv"
    stats_path = args.output_dir / "co2_adsorption_campaign_statistics.csv"
    json_path = args.output_dir / "co2_adsorption_campaign_summary.json"
    md_path = args.output_dir / "co2_adsorption_campaign_summary.md"
    axial_path = args.output_dir / "co2_adsorption_axial_loading_profiles.csv"
    plot_path = args.output_dir / "co2_adsorption_campaign_overview.png"
    write_csv(dataset_path, records)
    write_csv(stats_path, campaign_statistics(records))
    if axial_profiles:
        write_csv(axial_path, axial_profiles)
    json_path.write_text(
        json.dumps(json_safe({"campaign": campaign, "cases": records}), indent=2,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, records, campaign)
    plot_warning = None if args.no_plots else make_plot(plot_path, records)

    print(f"\nDiscovered runs: {len(records)}; unique seeds: {len(set(seeds))}")
    print(f"QA pass: {campaign['qa_pass_cases']}/{len(records)}")
    print(f"Practical equilibrium: {campaign['practical_equilibrium_cases']}/{len(records)}")
    if duplicate_seeds:
        print(f"WARNING duplicate seeds: {duplicate_seeds}")
    if inconsistent:
        print(f"WARNING inconsistent physical/numerical parameters: {', '.join(inconsistent)}")
    if plot_warning:
        print(f"WARNING {plot_warning}")
    print(f"Wrote {args.output_dir}")
    return 2 if args.strict and not campaign["complete_and_qa_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
