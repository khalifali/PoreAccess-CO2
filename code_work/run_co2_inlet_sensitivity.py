#!/usr/bin/env python3
"""Prepare, run and summarize a small CO2 inlet-boundary sensitivity study.

Why this test exists
--------------------
The present adsorption solver fixes every inlet-labelled pore at the reservoir
concentration.  The 20-bed campaign showed that the number of such pores is
strongly correlated with uptake.  This script checks whether conclusions are
sensitive to the thickness used to define the inlet region.

The expensive Delaunay and Sobol calculations are NOT repeated.  For each
selected existing network, this script copies the network arrays and changes
only ``pore_inlet``.  A pore is labelled as inlet when its centre satisfies

    z_pore <= bottom + inlet_layer_dp * particle_diameter.

This pore-centre rule is intentionally simple and reproducible.  It is a
diagnostic boundary test; it is not claimed to be the final physical boundary
model.  Particle--pore incidence and all pore/throat geometry remain unchanged.

Default study: three representative beds x four inlet layers = 12 runs.

Example
-------
python3 run_co2_inlet_sensitivity.py \
  --network-root co2_pore_networks_power22 \
  --solver solve_co2_pore_adsorption.py \
  --output-root co2_inlet_sensitivity \
  --run

The command is restartable: completed matching cases are skipped.  Use
``--force`` only when intentionally replacing a sensitivity result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEEDS = (105019, 18427, 27183)
DEFAULT_LAYERS = (0.75, 1.00, 1.25, 1.50)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--network-root", type=Path,
                   default=Path("co2_pore_networks_power22"))
    p.add_argument("--solver", type=Path,
                   default=Path("solve_co2_pore_adsorption.py"))
    p.add_argument("--output-root", type=Path,
                   default=Path("co2_inlet_sensitivity"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--layers-dp", type=float, nargs="+", default=list(DEFAULT_LAYERS))
    p.add_argument("--bottom", type=float, default=0.0, help="bed bottom [m]")
    p.add_argument("--tube-radius", type=float, default=8e-3, help="physical tube radius [m]")
    p.add_argument("--inlet-min-distance-dp", type=float, default=0.5)
    p.add_argument("--run", action="store_true",
                   help="run adsorption after preparing the network variants")
    p.add_argument("--force", action="store_true",
                   help="replace matching prepared/result files")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable used to launch the adsorption solver")

    # Same documented commercial 13X parameter set as the 20-case campaign.
    p.add_argument("--diffusivity", type=float, default=1.5e-5, help="[m2/s]")
    p.add_argument("--inlet-concentration", type=float, default=6.05, help="[mol/m3]")
    p.add_argument("--initial-concentration", type=float, default=0.0, help="[mol/m3]")
    p.add_argument("--temperature", type=float, default=298.15, help="[K]")
    p.add_argument("--particle-density", type=float, default=1600.0, help="[kg/m3]")
    p.add_argument("--qsat", type=float, default=5.332, help="[mol/kg]")
    p.add_argument("--b-pa", type=float, default=5.093e-5, help="[1/Pa]")
    p.add_argument("--toth-exponent", type=float, default=1.0)
    p.add_argument("--k-ldf", type=float, default=8.55e-3, help="[1/s]")
    p.add_argument("--initial-loading", type=float, default=0.0, help="[mol/kg]")
    p.add_argument("--t-end", type=float, default=5000.0, help="[s]")
    p.add_argument("--outputs", type=int, default=301)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-10)
    a = p.parse_args()
    if any(layer <= 0 for layer in a.layers_dp):
        p.error("all --layers-dp values must be positive")
    if len(set(a.seeds)) != len(a.seeds):
        p.error("--seeds contains duplicates")
    return a


def layer_label(layer: float) -> str:
    return f"layer_{layer:g}dp".replace(".", "p")


def connected_to_inlet(n: int, conns: np.ndarray, inlet: np.ndarray) -> bool:
    """Every labelled inlet must belong to the main graph."""
    degree = np.bincount(conns.ravel(), minlength=n)
    return bool(np.all(degree[inlet] > 0))


def prepare_variant(source: Path, destination: Path, seed: int, layer: float,
                    bottom: float, force: bool) -> dict[str, Any]:
    source_npz = source / "pore_network.npz"
    incidence = source / "particle_pore_incidence.csv"
    if not source_npz.is_file() or not incidence.is_file():
        raise FileNotFoundError(f"network or incidence file missing below {source}")

    with np.load(source_npz) as src:
        arrays = {name: np.asarray(src[name]) for name in src.files}
    required = ("pore_xyz", "pore_inlet", "pore_outlet", "throat_conns",
                "particle_radius")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"{source_npz} lacks {missing}")

    dp = 2.0 * float(np.median(arrays["particle_radius"]))
    cutoff = bottom + layer * dp
    inlet = np.asarray(arrays["pore_xyz"][:, 2] <= cutoff, dtype=bool)
    outlet = np.asarray(arrays["pore_outlet"], dtype=bool)
    # A pore cannot simultaneously be a fixed inlet and a top outlet.
    inlet &= ~outlet
    if not np.any(inlet):
        raise ValueError(f"seed {seed}, layer {layer:g}: no inlet pores")
    if not connected_to_inlet(len(inlet), np.asarray(arrays["throat_conns"], int), inlet):
        raise ValueError(f"seed {seed}, layer {layer:g}: isolated inlet pore detected")
    arrays["pore_inlet"] = inlet

    destination.mkdir(parents=True, exist_ok=True)
    network_out = destination / "pore_network.npz"
    incidence_out = destination / "particle_pore_incidence.csv"
    metadata_out = destination / "inlet_variant_metadata.json"
    if network_out.exists() and not force:
        old = json.loads(metadata_out.read_text(encoding="utf-8")) if metadata_out.is_file() else {}
        if old.get("seed") != seed or old.get("inlet_layer_dp") != layer:
            raise FileExistsError(
                f"existing variant does not match request: {destination}; use --force intentionally"
            )
    else:
        np.savez_compressed(network_out, **arrays)
        shutil.copy2(incidence, incidence_out)

    original_inlet = np.asarray(np.load(source_npz)["pore_inlet"], bool)
    metadata = {
        "seed": seed,
        "source_network": str(source.resolve()),
        "labelling_rule": "pore_z <= bottom + inlet_layer_dp * particle_diameter",
        "bottom_m": bottom,
        "particle_diameter_m": dp,
        "inlet_layer_dp": layer,
        "inlet_cutoff_z_m": cutoff,
        "original_inlet_pores": int(original_inlet.sum()),
        "sensitivity_inlet_pores": int(inlet.sum()),
        "pore_count": len(inlet),
        "topology_and_geometry_reused": True,
    }
    metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def solver_command(args: argparse.Namespace, network: Path, output: Path) -> list[str]:
    return [
        args.python, str(args.solver),
        "--network", str(network),
        "--output", str(output),
        "--mode", "adsorption",
        "--diffusivity", str(args.diffusivity),
        "--inlet-concentration", str(args.inlet_concentration),
        "--initial-concentration", str(args.initial_concentration),
        "--temperature", str(args.temperature),
        "--particle-density", str(args.particle_density),
        "--qsat", str(args.qsat),
        "--b-pa", str(args.b_pa),
        "--toth-exponent", str(args.toth_exponent),
        "--k-ldf", str(args.k_ldf),
        "--initial-loading", str(args.initial_loading),
        "--inlet-boundary", "finite",
        "--tube-radius", str(args.tube_radius),
        "--bed-bottom", str(args.bottom),
        "--inlet-min-distance-dp", str(args.inlet_min_distance_dp),
        "--inlet-area-weighting", "equal",
        "--t-end", str(args.t_end),
        "--outputs", str(args.outputs),
        "--rtol", str(args.rtol),
        "--atol", str(args.atol),
    ]


def result_matches(path: Path, args: argparse.Namespace, network: Path) -> bool:
    metadata_path = path / "run_metadata.json"
    timeseries_path = path / "timeseries.csv"
    if not metadata_path.is_file() or not timeseries_path.is_file():
        return False
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        p = data["parameters"]
        return bool(
            data.get("solver_success")
            and Path(data["network"]).resolve() == network.resolve()
            and p.get("inlet_boundary") == "finite"
            and float(p["t_end"]) == args.t_end
            and float(p["b_pa"]) == args.b_pa
            and float(p["k_ldf"]) == args.k_ldf
            and float(p["qsat"]) == args.qsat
        )
    except Exception:
        return False


def run_case(args: argparse.Namespace, network: Path, result: Path) -> None:
    if result_matches(result, args, network) and not args.force:
        print(f"SKIP completed matching result: {result}")
        return
    result.mkdir(parents=True, exist_ok=True)
    command = solver_command(args, network, result)
    command_file = result / "command.txt"
    command_file.write_text(" ".join(command) + "\n", encoding="utf-8")
    print(f"RUN {result}")
    completed = subprocess.run(command, text=True, capture_output=True)
    (result / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (result / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"solver failed for {result} with code {completed.returncode}; "
            f"see stdout.log and stderr.log"
        )


def read_timeseries(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"insufficient rows in {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def crossing_time(t: np.ndarray, y: np.ndarray, fraction: float) -> float:
    target = y[0] + fraction * (y[-1] - y[0])
    ym = np.maximum.accumulate(y)
    hits = np.flatnonzero(ym >= target)
    if not len(hits):
        return math.nan
    i = int(hits[0])
    if i == 0 or ym[i] == ym[i - 1]:
        return float(t[i])
    f = (target - ym[i - 1]) / (ym[i] - ym[i - 1])
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def summarize_case(seed: int, layer: float, variant: Path, result: Path,
                   args: argparse.Namespace) -> dict[str, Any]:
    vm = json.loads((variant / "inlet_variant_metadata.json").read_text(encoding="utf-8"))
    rm = json.loads((result / "run_metadata.json").read_text(encoding="utf-8"))
    ts = read_timeseries(result / "timeseries.csv")
    t = ts["time_s"]
    ads = ts["adsorbed_inventory_mol"]
    late_start = 0.9 * t[-1]
    late_ads = float(np.interp(late_start, t, ads))
    final_ads = float(ads[-1])
    return {
        "seed": seed,
        "inlet_layer_dp": layer,
        "inlet_cutoff_z_m": vm["inlet_cutoff_z_m"],
        "original_inlet_pores": vm["original_inlet_pores"],
        "sensitivity_inlet_pores": vm["sensitivity_inlet_pores"],
        "solver_success": bool(rm.get("solver_success")),
        "physical_bounds_ok": bool(rm.get("physical_bounds_ok")),
        "maximum_relative_mass_balance_error": rm.get("maximum_relative_mass_balance_error"),
        "final_adsorbed_inventory_mol": final_ads,
        "final_uptake_mol_kg": float(ts["mean_loading_mol_kg"][-1]),
        "t50_final_uptake_s": crossing_time(t, ads, 0.50),
        "t90_final_uptake_s": crossing_time(t, ads, 0.90),
        "late_uptake_fraction": (final_ads - late_ads) / final_ads,
        "final_mean_gas_concentration_mol_m3": float(ts["mean_gas_concentration_mol_m3"][-1]),
        "variant_network": str(variant),
        "result_directory": str(result),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING matplotlib unavailable; sensitivity plot skipped")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = sorted((row for row in rows if row["seed"] == seed),
                          key=lambda row: row["inlet_layer_dp"])
        layer = [row["inlet_layer_dp"] for row in selected]
        axes[0].plot(layer, [row["sensitivity_inlet_pores"] for row in selected], "o-", label=str(seed))
        axes[1].plot(layer, [row["final_uptake_mol_kg"] for row in selected], "o-")
        axes[2].plot(layer, [row["t50_final_uptake_s"] for row in selected], "o-")
    axes[0].set_ylabel("Inlet pore count")
    axes[1].set_ylabel("Uptake at 5000 s [mol/kg]")
    axes[2].set_ylabel("$t_{50}$ relative to final uptake [s]")
    for ax in axes:
        ax.set_xlabel("Inlet-layer thickness [$d_p$]")
        ax.grid(alpha=0.25)
    axes[0].legend(title="Seed")
    fig.suptitle("Inlet-boundary sensitivity")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = arguments()
    if args.run and not args.solver.is_file():
        sys.exit(f"adsorption solver not found: {args.solver}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    failures: list[str] = []
    for seed in args.seeds:
        source = args.network_root / f"seed_{seed}"
        for layer in args.layers_dp:
            label = layer_label(layer)
            variant = args.output_root / "networks" / f"seed_{seed}" / label
            result = args.output_root / "results" / f"seed_{seed}" / label
            try:
                meta = prepare_variant(source, variant, seed, layer, args.bottom, args.force)
                manifest.append(meta | {"variant_network": str(variant), "result_directory": str(result)})
                print(f"PREP seed {seed}, {layer:g} dp: {meta['sensitivity_inlet_pores']} inlet pores")
                if args.run:
                    run_case(args, variant, result)
            except Exception as exc:
                message = f"seed {seed}, layer {layer:g}: {exc}"
                failures.append(message)
                print(f"ERROR {message}", file=sys.stderr)

    (args.output_root / "sensitivity_manifest.json").write_text(
        json.dumps({"cases": manifest, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.run:
        rows = []
        for seed in args.seeds:
            for layer in args.layers_dp:
                label = layer_label(layer)
                variant = args.output_root / "networks" / f"seed_{seed}" / label
                result = args.output_root / "results" / f"seed_{seed}" / label
                if (result / "run_metadata.json").is_file():
                    rows.append(summarize_case(seed, layer, variant, result, args))
        if rows:
            write_summary(args.output_root / "inlet_sensitivity_summary.csv", rows)
            plot_summary(args.output_root / "inlet_sensitivity_overview.png", rows)
            print(f"Wrote sensitivity summary for {len(rows)} cases")
    print(f"Study directory: {args.output_root}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
