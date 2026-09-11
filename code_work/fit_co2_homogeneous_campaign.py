#!/usr/bin/env python3
"""Fit the verified homogeneous 1D model to every accepted CO2 bed.

For each seed, this runner reads bed height and porosity from ``network_qa.json``,
uses the matching pore-network adsorption ``timeseries.csv`` as reference, and
launches ``solve_co2_homogeneous_1d.py`` with the frozen production settings.

The campaign is sequential and restart-safe:
* completed cases with matching parameters and valid metadata are skipped;
* incomplete directories are resumed by rerunning the deterministic fit;
* completed results with conflicting parameters are never overwritten unless
  ``--force-conflicts`` is supplied;
* summary CSV and JSON files are refreshed after every case.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = [
    18427, 27183, 31991, 40213, 47837, 52691, 61417, 69371, 74167, 81929,
    90709, 98251, 105019, 113117, 121021, 129121, 137077, 145093, 153133,
    161123,
]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--network-root", type=Path, default=Path("co2_pore_networks_power22"))
    p.add_argument("--adsorption-root", type=Path,
                   default=Path("co2_adsorption_campaign_finite_inlet_5000s"))
    p.add_argument("--output-root", type=Path,
                   default=Path("co2_homogeneous_campaign_100cells"))
    p.add_argument("--solver", type=Path,
                   default=Path(__file__).with_name("solve_co2_homogeneous_1d.py"))
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--bed-bottom", type=float, default=0.0)
    p.add_argument("--cells", type=int, default=100)
    p.add_argument("--tube-radius", type=float, default=0.008)
    p.add_argument("--t-end", type=float, default=5000.0)
    p.add_argument("--fit-min-diffusivity", type=float, default=1e-8)
    p.add_argument("--fit-max-diffusivity", type=float, default=1.5e-5)
    p.add_argument("--fit-weighting", choices=["time", "points"], default="time")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--force-conflicts", action="store_true",
                   help="Allow rerun when existing completed metadata uses different settings.")
    p.add_argument("--rerun-completed", action="store_true",
                   help="Rerun even valid matching cases.")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-fit-nrmse", type=float, default=0.05)
    p.add_argument("--max-final-loading-relative-error", type=float, default=0.10)
    p.add_argument("--max-absolute-mass-residual", type=float, default=1e-10)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, rtol=1e-10, atol=1e-14) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


def expected_settings(args, height, porosity) -> dict:
    return {
        "cells": args.cells, "fit_weighting": args.fit_weighting,
        "t_end_s": args.t_end, "bed_height_m": height, "porosity": porosity,
        "tube_radius_m": args.tube_radius, "temperature_K": 298.15,
        "particle_density_kg_m3": 1600.0, "qsat_mol_kg": 5.332,
        "b_Pa_inverse": 5.093e-5, "toth_exponent": 1.0,
        "k_ldf_s_inverse": 8.55e-3,
    }


def metadata_matches(meta: dict, expected: dict) -> tuple[bool, list[str]]:
    differences = []
    for key, value in expected.items():
        actual = meta.get(key)
        if isinstance(value, str):
            ok = actual == value
        elif isinstance(value, int):
            ok = actual == value
        else:
            ok = close(actual, value)
        if not ok:
            differences.append(f"{key}: existing={actual!r}, expected={value!r}")
    return not differences, differences


def build_command(args, reference: Path, output: Path, height: float,
                  porosity: float) -> list[str]:
    return [
        sys.executable, str(args.solver), "--mode", "fit",
        "--reference-timeseries", str(reference), "--output", str(output),
        "--bed-height", repr(height), "--porosity", repr(porosity),
        "--tube-radius", repr(args.tube_radius), "--cells", str(args.cells),
        "--inlet-concentration", "6.05", "--initial-concentration", "0",
        "--temperature", "298.15", "--particle-density", "1600",
        "--qsat", "5.332", "--b-pa", "5.093e-5", "--toth-exponent", "1",
        "--k-ldf", "8.55e-3", "--initial-loading", "0",
        "--boundary-diffusivity", "1.5e-5", "--inlet-transfer-length", "8.0e-4",
        "--fit-min-diffusivity", repr(args.fit_min_diffusivity),
        "--fit-max-diffusivity", repr(args.fit_max_diffusivity),
        "--fit-weighting", args.fit_weighting, "--t-end", repr(args.t_end),
        "--rtol", repr(args.rtol), "--atol", repr(args.atol),
    ]


def run_and_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def summarize_case(seed: int, case_dir: Path, qa: dict, status: str,
                   error: str = "") -> dict:
    row = {
        "seed": seed, "status": status, "error": error,
        "bed_height_m": qa.get("bed_top_m"),
        "porosity": qa.get("analytic_porosity"),
        "network_ready_for_transport": qa.get("network_ready_for_transport"),
        "result_directory": str(case_dir),
    }
    metadata_path = case_dir / "run_metadata.json"
    if not metadata_path.exists():
        row["case_qa_pass"] = False
        return row
    try:
        m = load_json(metadata_path)
        row.update({
            "solver_success": m.get("solver_success"),
            "physical_bounds_ok": m.get("physical_bounds_ok"),
            "cells": m.get("cells"), "fit_weighting": m.get("fit_weighting"),
            "effective_diffusivity_m2_s": m.get("effective_diffusivity_m2_s"),
            "effective_over_molecular_diffusivity":
                (1.0 / m["molecular_to_effective_diffusivity_ratio"]
                 if m.get("molecular_to_effective_diffusivity_ratio") else np.nan),
            "fit_normalized_rmse": m.get("fit_normalized_rmse"),
            "final_homogeneous_loading_mol_kg": m.get("final_mean_loading_mol_kg"),
            "maximum_relative_mass_balance_error":
                m.get("maximum_relative_mass_balance_error"),
        })
        comparison = pd.read_csv(case_dir / "fit_comparison.csv")
        ref_final = float(comparison.reference_loading_mol_kg.iloc[-1])
        pred_final = float(comparison.homogeneous_loading_mol_kg.iloc[-1])
        row["final_reference_loading_mol_kg"] = ref_final
        row["final_loading_error_mol_kg"] = pred_final - ref_final
        row["final_loading_relative_error"] = ((pred_final-ref_final)/ref_final
                                                if ref_final != 0 else np.nan)
        series = pd.read_csv(case_dir / "timeseries.csv")
        row["maximum_absolute_mass_balance_residual_mol"] = float(
            series.mass_balance_residual_mol.abs().max())
        row["final_relative_mass_balance_error"] = float(
            series.relative_mass_balance_error.iloc[-1])
        deff = float(row["effective_diffusivity_m2_s"])
        at_bound = (deff <= 1.01*ARGS.fit_min_diffusivity or
                    deff >= 0.99*ARGS.fit_max_diffusivity)
        row["fit_at_bound"] = at_bound
        row["case_qa_pass"] = bool(
            m.get("solver_success") and m.get("physical_bounds_ok") and
            not at_bound and np.isfinite(row["fit_normalized_rmse"]) and
            row["fit_normalized_rmse"] <= ARGS.max_fit_nrmse and
            abs(row["final_loading_relative_error"]) <=
                ARGS.max_final_loading_relative_error and
            row["maximum_absolute_mass_balance_residual_mol"] <=
                ARGS.max_absolute_mass_residual)
    except Exception as exc:
        row["case_qa_pass"] = False
        row["error"] = (row["error"] + "; " if row["error"] else "") + str(exc)
    return row


def write_summary(rows: list[dict], output_root: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(output_root / "homogeneous_campaign_summary.csv", index=False)
    payload = {
        "cases_recorded": len(frame),
        "successful_or_skipped": int(frame.status.isin(["succeeded", "skipped"]).sum()),
        "qa_pass_cases": int(frame.get("case_qa_pass", pd.Series(dtype=bool)).fillna(False).sum()),
        "all_recorded_cases_pass": bool(len(frame) and frame.get(
            "case_qa_pass", pd.Series(False, index=frame.index)).fillna(False).all()),
        "cases": frame.replace({np.nan: None}).to_dict("records"),
    }
    (output_root / "homogeneous_campaign_summary.json").write_text(
        json.dumps(payload, indent=2)+"\n", encoding="utf-8")


def main() -> None:
    global ARGS
    ARGS = arguments()
    if not ARGS.solver.exists():
        raise SystemExit(f"Solver not found: {ARGS.solver}")
    ARGS.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, seed in enumerate(ARGS.seeds, 1):
        network_dir = ARGS.network_root / f"seed_{seed}"
        qa_path = network_dir / "network_qa.json"
        reference = ARGS.adsorption_root / f"seed_{seed}" / "timeseries.csv"
        case_dir = ARGS.output_root / f"seed_{seed}"
        print(f"\n[{number}/{len(ARGS.seeds)}] seed {seed}")
        if not qa_path.exists() or not reference.exists():
            error = f"missing {'network QA' if not qa_path.exists() else 'reference timeseries'}"
            print("FAILED:", error)
            rows.append({"seed":seed, "status":"failed", "error":error,
                         "case_qa_pass":False})
            write_summary(rows, ARGS.output_root)
            if not ARGS.continue_on_error:
                raise SystemExit(error)
            continue
        qa = load_json(qa_path)
        if not qa.get("network_ready_for_transport", False):
            error = "network_ready_for_transport is false"
            rows.append(summarize_case(seed, case_dir, qa, "failed", error))
            write_summary(rows, ARGS.output_root)
            if not ARGS.continue_on_error:
                raise SystemExit(error)
            continue
        height = float(qa["bed_top_m"]) - ARGS.bed_bottom
        porosity = float(qa["analytic_porosity"])
        expected = expected_settings(ARGS, height, porosity)
        meta_path = case_dir / "run_metadata.json"
        if meta_path.exists():
            existing = load_json(meta_path)
            matches, differences = metadata_matches(existing, expected)
            valid = bool(existing.get("solver_success") and
                         existing.get("physical_bounds_ok") and matches)
            if valid and not ARGS.rerun_completed:
                print("SKIP: completed matching result")
                rows.append(summarize_case(seed, case_dir, qa, "skipped"))
                write_summary(rows, ARGS.output_root)
                continue
            if not matches and not ARGS.force_conflicts:
                error = "existing completed metadata conflicts: " + "; ".join(differences)
                print("FAILED:", error)
                rows.append(summarize_case(seed, case_dir, qa, "failed", error))
                write_summary(rows, ARGS.output_root)
                if not ARGS.continue_on_error:
                    raise SystemExit(error + " (use --force-conflicts to replace)")
                continue
        command = build_command(ARGS, reference, case_dir, height, porosity)
        if ARGS.dry_run:
            print("DRY RUN:", " ".join(command))
            rows.append({"seed":seed, "status":"dry_run", "error":"",
                         "bed_height_m":height, "porosity":porosity,
                         "case_qa_pass":False})
            write_summary(rows, ARGS.output_root)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        code = run_and_tee(command, case_dir / "fit.log")
        if code != 0:
            error = f"solver exit code {code}; see {case_dir/'fit.log'}"
            rows.append(summarize_case(seed, case_dir, qa, "failed", error))
            write_summary(rows, ARGS.output_root)
            if not ARGS.continue_on_error:
                raise SystemExit(error)
            continue
        row = summarize_case(seed, case_dir, qa, "succeeded")
        rows.append(row)
        write_summary(rows, ARGS.output_root)
        print(f"seed {seed}: D_eff={row.get('effective_diffusivity_m2_s',np.nan):.8e}, "
              f"NRMSE={row.get('fit_normalized_rmse',np.nan):.4%}, "
              f"QA={row.get('case_qa_pass')}")
    write_summary(rows, ARGS.output_root)
    if ARGS.dry_run:
        print(f"\nDry run complete: {len(rows)} commands checked")
        return
    passed = sum(bool(r.get("case_qa_pass")) for r in rows)
    print(f"\nCampaign complete: {passed}/{len(rows)} recorded cases pass QA")
    if passed != len(ARGS.seeds):
        raise SystemExit(2)


if __name__ == "__main__":
    ARGS = None
    main()
