#!/usr/bin/env python3
"""Generate all non-ParaView figures for the CO2 packed-bed manuscript.

The program creates eleven standalone figures in vector PDF/SVG and 600 dpi PNG:

1. campaign_variability     -- matched porosity and 20 uptake curves
2. accessibility_responses  -- structure-response correlations
3. homogeneous_fits         -- detailed network versus homogeneous model
4. diffusivity_closure      -- fitted closure and leave-one-bed-out predictions
5. accessibility_depth      -- robustness to the interior-plane definition

The script never changes simulation results.  It also writes a JSON manifest
recording every source file used and a CSV listing representative seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#777777",
    "light": "#D7DCE2",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate non-ParaView figures for the CO2 packed-bed paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--campaign-dataset", type=Path,
                   default=Path("co2_adsorption_finite_inlet_analysis/co2_adsorption_campaign_dataset.csv"))
    p.add_argument("--adsorption-root", type=Path,
                   default=Path("co2_adsorption_campaign_finite_inlet_5000s"))
    p.add_argument("--homogeneous-summary", type=Path,
                   default=Path("co2_homogeneous_campaign_100cells/homogeneous_campaign_summary.csv"))
    p.add_argument("--homogeneous-root", type=Path,
                   default=Path("co2_homogeneous_campaign_100cells"))
    p.add_argument("--closure-dir", type=Path,
                   default=Path("co2_effective_diffusivity_closure"))
    p.add_argument("--closure-dataset", type=Path, default=None,
                   help="Override closure-dir/closure_dataset.csv.")
    p.add_argument("--closure-predictions", type=Path, default=None,
                   help="Override closure-dir/closure_loo_predictions.csv.")
    p.add_argument("--depth-sensitivity", type=Path, default=None,
                   help="Override closure-dir/closure_depth_sensitivity.csv.")
    p.add_argument("--molecular-diffusivity", type=float, default=1.5e-5,
                   help="Reference molecular diffusivity [m2/s].")
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="Bed bootstrap samples for the power-closure band.")
    p.add_argument("--random-seed", type=int, default=20260824)
    p.add_argument("--output", type=Path, default=Path("co2_publication_figures"))
    p.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"],
                   choices=["pdf", "png", "svg"])
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--only", nargs="+",
                   choices=["campaign", "accessibility", "homogeneous", "closure", "depth"],
                   help="Generate only selected figures.")
    return p.parse_args()


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 11.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def read_csv(path: Path, label: str) -> pd.DataFrame:
    return pd.read_csv(require(path, label))


def normalize_seed(values):
    extracted = values.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def prepare_seed(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "seed" not in df:
        raise ValueError(f"{source} has no 'seed' column")
    df = df.copy()
    df["seed"] = normalize_seed(df["seed"])
    if df.seed.isna().any():
        raise ValueError(f"Could not parse every seed in {source}")
    df["seed"] = df["seed"].astype(int)
    return df


def require_columns(df: pd.DataFrame, columns, source: str):
    missing = [c for c in columns if c not in df]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def qa_filter(df: pd.DataFrame) -> pd.DataFrame:
    for name in ["case_qa_pass", "qa_pass"]:
        if name in df:
            accepted = df[name].astype(str).str.lower().isin(["true", "1", "yes"])
            if not accepted.any():
                raise ValueError("No cases passed QA")
            return df[accepted].copy()
    return df.copy()


def single_panels(count):
    pairs = [plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
             for _ in range(count)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def panel_label(ax, label):
    # Standalone figures intentionally have no panel letters.
    pass


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E4E7EB", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, output: Path, stem: str, formats, dpi, manifest):
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, list):
        names = {
            "fig_campaign_variability": ["fig_porosity_variability", "fig_uptake_curves"],
            "fig_accessibility_responses": ["fig_porosity_uptake", "fig_accessibility_uptake", "fig_accessibility_utilization"],
            "fig_homogeneous_fits": ["fig_homogeneous_uptake", "fig_homogeneous_residuals"],
            "fig_diffusivity_closure": ["fig_closure_power_law", "fig_closure_held_out"],
            "fig_accessibility_depth": ["fig_depth_correlations", "fig_depth_prediction"],
        }
        for figure, name in zip(fig, names[stem]):
            save_figure(figure, output, name, formats, dpi, manifest)
        return
    written = []
    for fmt in formats:
        path = output / f"{stem}.{fmt}"
        kwargs = {"dpi": dpi} if fmt == "png" else {}
        fig.savefig(path, **kwargs)
        if fmt == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        written.append(str(path))
    plt.close(fig)
    manifest["figures"][stem] = written
    print(f"wrote {', '.join(written)}")


def representative_seeds(df, value_column):
    work = df[["seed", value_column]].dropna().sort_values(value_column)
    if len(work) < 3:
        raise ValueError(f"Need at least three cases for {value_column}")
    values = work[value_column].to_numpy()
    median = np.median(values)
    middle = int(np.argmin(np.abs(values - median)))
    rows = [work.iloc[0], work.iloc[middle], work.iloc[-1]]
    labels = ["low", "median", "high"]
    return {label: int(row.seed) for label, row in zip(labels, rows)}


def find_seed_file(root: Path, seed: int, filename: str) -> Path:
    direct = [
        root / f"seed_{seed}" / filename,
        root / f"co2_13x_seed_{seed}" / filename,
        root / str(seed) / filename,
    ]
    for path in direct:
        if path.is_file():
            return path
    matches = [p for p in root.rglob(filename)
               if re.search(rf"(?<!\d){seed}(?!\d)", str(p.parent))]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {filename} for seed {seed} below {root}")
    raise RuntimeError(f"Multiple {filename} files for seed {seed}: {matches}")


def get_time_loading(df: pd.DataFrame, source: Path):
    time_candidates = ["time_s", "time"]
    loading_candidates = [
        "mean_loading_mol_kg", "final_uptake_mol_kg",
        "reference_loading_mol_kg", "homogeneous_loading_mol_kg",
    ]
    t = next((c for c in time_candidates if c in df), None)
    q = next((c for c in loading_candidates if c in df), None)
    if t is None or q is None:
        raise ValueError(f"Cannot find time/loading columns in {source}; columns={list(df)}")
    return df[t].to_numpy(float), df[q].to_numpy(float)


def figure_campaign(args, campaign, closure, manifest, seed_records):
    require_columns(campaign, ["seed", "final_uptake_mol_kg"], "campaign dataset")
    require_columns(closure, ["seed", "porosity"], "closure dataset")
    data = campaign.merge(closure[["seed", "porosity"]], on="seed", how="inner")
    seeds = representative_seeds(data, "final_uptake_mol_kg")
    seed_records["campaign_uptake"] = seeds

    fig, axes = single_panels(2)
    ax = axes[0]
    order = data.sort_values("seed")
    ax.scatter(np.arange(len(order)), order.porosity, color=COLORS["blue"],
               s=26, edgecolor="white", linewidth=.45, zorder=3)
    ax.axhline(order.porosity.mean(), color=COLORS["red"], ls="--", lw=1.2,
               label=f"mean = {order.porosity.mean():.4f}")
    ax.set_xlabel("Packing realization (ordered by seed)")
    ax.set_ylabel(r"Robust bulk porosity, $\varepsilon_b$")
    ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    ax.legend(frameon=False)
    clean_axes(ax); panel_label(ax, "a")

    ax = axes[1]
    selected_colors = {"low": COLORS["red"], "median": COLORS["blue"],
                       "high": COLORS["green"]}
    selected_by_seed = {v: k for k, v in seeds.items()}
    used = []
    for seed in sorted(data.seed):
        path = find_seed_file(args.adsorption_root, int(seed), "timeseries.csv")
        ts = pd.read_csv(path)
        t, q = get_time_loading(ts, path)
        used.append(str(path))
        if seed in selected_by_seed:
            level = selected_by_seed[seed]
            ax.plot(t, q, color=selected_colors[level], lw=2.0,
                    label=f"{level}: seed {seed}", zorder=3)
        else:
            ax.plot(t, q, color=COLORS["light"], lw=.8, alpha=.85, zorder=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean loading [mol kg$^{-1}$]")
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax); panel_label(ax, "b")
    manifest["sources"]["campaign_variability"] = used
    save_figure(fig, args.output, "fig_campaign_variability", args.formats, args.dpi, manifest)


def bootstrap_regression_band(x, y, xgrid, rng, nboot=5000):
    n = len(x)
    predictions = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0:
            continue
        slope, intercept = np.polyfit(x[idx], y[idx], 1)
        predictions.append(intercept + slope * xgrid)
    pred = np.asarray(predictions)
    return np.quantile(pred, .025, axis=0), np.quantile(pred, .975, axis=0)


def scatter_with_fit(ax, x, y, xlabel, ylabel, color, rng, annotate=True):
    ax.scatter(x, y, color=color, s=32, edgecolor="white", linewidth=.5, zorder=3)
    xgrid = np.linspace(np.min(x), np.max(x), 200)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(xgrid, intercept + slope*xgrid, color=color, lw=1.7)
    lo, hi = bootstrap_regression_band(np.asarray(x), np.asarray(y), xgrid, rng)
    ax.fill_between(xgrid, lo, hi, color=color, alpha=.15, linewidth=0)
    rho, p = spearmanr(x, y)
    if annotate:
        ax.text(.04, .95, rf"$\rho_s={rho:.3f}$" + "\n" + rf"$p={p:.1e}$",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=.85))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); clean_axes(ax)


def figure_accessibility(args, campaign, closure, manifest):
    columns = ["seed", "porosity", "effective_inlet_to_interior_conductance_m3_s"]
    require_columns(closure, columns, "closure dataset")
    require_columns(campaign, ["seed", "final_uptake_mol_kg",
                               "underutilized_particle_fraction"], "campaign dataset")
    data = campaign.merge(closure[columns], on="seed", how="inner", validate="one_to_one")
    if len(data) != 20:
        print(f"warning: accessibility figure uses {len(data)} merged beds", file=sys.stderr)
    rng = np.random.default_rng(args.random_seed + 11)
    fig, axes = single_panels(3)
    scatter_with_fit(axes[0], data.porosity, data.final_uptake_mol_kg,
                     r"Bulk porosity, $\varepsilon_b$", r"Uptake at 5000 s [mol kg$^{-1}$]",
                     COLORS["gray"], rng)
    panel_label(axes[0], "a")
    g_scaled = data.effective_inlet_to_interior_conductance_m3_s * 1e8
    scatter_with_fit(axes[1], g_scaled, data.final_uptake_mol_kg,
                     r"$G_{\mathrm{access}}$ [$10^{-8}$ m$^3$ s$^{-1}$]", r"Uptake at 5000 s [mol kg$^{-1}$]",
                     COLORS["blue"], rng)
    panel_label(axes[1], "b")
    scatter_with_fit(axes[2], g_scaled, data.underutilized_particle_fraction,
                     r"$G_{\mathrm{access}}$ [$10^{-8}$ m$^3$ s$^{-1}$]", "Underutilized-particle fraction",
                     COLORS["orange"], rng)
    panel_label(axes[2], "c")
    manifest["sources"]["accessibility_responses"] = [
        str(args.campaign_dataset), str(closure.attrs.get("source", "closure dataset"))]
    save_figure(fig, args.output, "fig_accessibility_responses", args.formats, args.dpi, manifest)


def figure_homogeneous(args, hom_summary, manifest, seed_records):
    require_columns(hom_summary, ["seed", "effective_diffusivity_m2_s"], "homogeneous summary")
    seeds = representative_seeds(hom_summary, "effective_diffusivity_m2_s")
    seed_records["effective_diffusivity"] = seeds
    colors = {"low": COLORS["red"], "median": COLORS["blue"], "high": COLORS["green"]}
    fig, axes = single_panels(2)
    used = []
    for level, seed in seeds.items():
        path = find_seed_file(args.homogeneous_root, seed, "fit_comparison.csv")
        df = pd.read_csv(path)
        require_columns(df, ["time_s", "reference_loading_mol_kg",
                             "homogeneous_loading_mol_kg"], str(path))
        used.append(str(path))
        axes[0].plot(df.time_s, df.reference_loading_mol_kg, color=colors[level],
                     lw=2.1, label=f"seed {seed}")
        axes[0].plot(df.time_s, df.homogeneous_loading_mol_kg, color=colors[level],
                     lw=1.35, ls="--", label="_nolegend_")
        residual = df.homogeneous_loading_mol_kg - df.reference_loading_mol_kg
        axes[1].plot(df.time_s, residual, color=colors[level], label=f"seed {seed}")
    axes[0].set_xlabel("Time [s]"); axes[0].set_ylabel("Mean loading [mol kg$^{-1}$]")
    axes[0].legend(frameon=False, ncol=1, fontsize=8.0, title="Solid: network; dashed: fitted 1D")
    clean_axes(axes[0]); panel_label(axes[0], "a")
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_xlabel("Time [s]"); axes[1].set_ylabel("1D model $-$ network [mol kg$^{-1}$]")
    axes[1].legend(frameon=False)
    clean_axes(axes[1]); panel_label(axes[1], "b")
    manifest["sources"]["homogeneous_fits"] = used
    save_figure(fig, args.output, "fig_homogeneous_fits", args.formats, args.dpi, manifest)


def power_fit(x, y):
    exponent, log_c = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_c)), float(exponent)


def bootstrap_power_band(x, y, xgrid, nboot, rng):
    n = len(x)
    curves, coeffs = [], []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2:
            continue
        c, exponent = power_fit(x[idx], y[idx])
        curves.append(c * xgrid**exponent)
        coeffs.append((c, exponent))
    curves = np.asarray(curves)
    return (np.quantile(curves, .025, axis=0), np.quantile(curves, .975, axis=0),
            np.asarray(coeffs))


def figure_closure(args, closure, predictions, manifest):
    require_columns(closure, ["accessibility_diffusivity_m2_s",
                              "effective_diffusivity_m2_s"], "closure dataset")
    require_columns(predictions, ["seed", "model", "observed_D_eff_m2_s",
                                  "loo_predicted_D_eff_m2_s"], "closure predictions")
    x = closure.accessibility_diffusivity_m2_s.to_numpy(float)
    y = closure.effective_diffusivity_m2_s.to_numpy(float)
    c, exponent = power_fit(x, y)
    xgrid = np.linspace(.96*x.min(), 1.04*x.max(), 300)
    rng = np.random.default_rng(args.random_seed)
    lo, hi, boot = bootstrap_power_band(x, y, xgrid, args.bootstrap, rng)

    fig, axes = single_panels(2)
    ax = axes[0]
    ax.scatter(x*1e6, y*1e6, color=COLORS["blue"], s=34,
               edgecolor="white", linewidth=.5, zorder=3)
    ax.plot(xgrid*1e6, c*xgrid**exponent*1e6, color=COLORS["red"],
            label=rf"power law, $n={exponent:.2f}$")
    ax.fill_between(xgrid*1e6, lo*1e6, hi*1e6, color=COLORS["red"], alpha=.16,
                    label="bed-bootstrap 95% band")
    rho, p = spearmanr(x, y)
    ax.text(.04, .95, rf"$\rho_s={rho:.3f}$" + "\n" + rf"$p={p:.1e}$",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=.85))
    ax.set_xlabel("Accessibility diffusivity [$10^{-6}$ m$^2$ s$^{-1}$]")
    ax.set_ylabel(r"Fitted $D_{\mathrm{eff}}$ [$10^{-6}$ m$^2$ s$^{-1}$]")
    ax.legend(frameon=False, loc="lower right"); clean_axes(ax); panel_label(ax, "a")

    ax = axes[1]
    model_styles = {
        "porosity_linear": (COLORS["gray"], "porosity"),
        "physical_through_origin": (COLORS["orange"], "proportional"),
        "accessibility_power_law": (COLORS["blue"], "power closure"),
    }
    all_values = []
    for model, (color, label) in model_styles.items():
        d = predictions[predictions.model == model]
        if d.empty:
            raise ValueError(f"closure predictions contain no model '{model}'")
        obs = d.observed_D_eff_m2_s.to_numpy()*1e6
        pred = d.loo_predicted_D_eff_m2_s.to_numpy()*1e6
        ax.scatter(obs, pred, color=color, s=28, alpha=.9, label=label,
                   marker={"porosity_linear": "s", "physical_through_origin": "^", "accessibility_power_law": "o"}[model],
                   edgecolor="white", linewidth=.4)
        all_values.extend(obs); all_values.extend(pred)
    lower, upper = min(all_values), max(all_values)
    pad = .04*(upper-lower)
    ax.plot([lower-pad, upper+pad], [lower-pad, upper+pad], "k--", lw=1.0,
            label="ideal")
    ax.set_xlim(lower-pad, upper+pad); ax.set_ylim(lower-pad, upper+pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Observed $D_{\mathrm{eff}}$ [$10^{-6}$ m$^2$ s$^{-1}$]")
    ax.set_ylabel("Leave-one-bed-out prediction [$10^{-6}$ m$^2$ s$^{-1}$]")
    ax.legend(frameon=False); clean_axes(ax); panel_label(ax, "b")

    pd.DataFrame(boot, columns=["power_coefficient_SI", "power_exponent"]).to_csv(
        args.output / "closure_power_bootstrap_samples.csv", index=False)
    manifest["sources"]["diffusivity_closure"] = [
        str(closure.attrs.get("source", "closure dataset")),
        str(predictions.attrs.get("source", "closure predictions"))]
    manifest["closure_fit"] = {
        "power_coefficient_SI": c, "power_exponent": exponent,
        "dimensionless_coefficient_using_Dm": c*args.molecular_diffusivity**(exponent-1),
        "molecular_diffusivity_m2_s": args.molecular_diffusivity,
        "bootstrap_samples": int(len(boot)),
    }
    save_figure(fig, args.output, "fig_diffusivity_closure", args.formats, args.dpi, manifest)


def figure_depth(args, depth, manifest):
    require_columns(depth, ["depth_dp", "pearson_r", "spearman_rho",
                            "linear_loo_r2"], "depth sensitivity")
    d = depth.sort_values("depth_dp")
    fig, axes = single_panels(2)
    ax = axes[0]
    ax.plot(d.depth_dp, d.pearson_r, "o-", color=COLORS["blue"], label="Pearson $r$")
    ax.plot(d.depth_dp, d.spearman_rho, "s--", color=COLORS["green"], label=r"Spearman $\rho_s$")
    ax.set_xlabel("Interior-plane depth [$d_p$]")
    ax.set_ylabel(r"Correlation with fitted $D_{\mathrm{eff}}$")
    ax.set_xticks(d.depth_dp); ax.set_ylim(.80, 1.0)
    ax.legend(frameon=False); clean_axes(ax); panel_label(ax, "a")
    ax = axes[1]
    ax.plot(d.depth_dp, d.linear_loo_r2, "o-", color=COLORS["orange"],
            label="affine accessibility model")
    if "origin_loo_r2" in d:
        ax.plot(d.depth_dp, d.origin_loo_r2, "s--", color=COLORS["purple"],
                label="proportional model")
    ax.set_xlabel("Interior-plane depth [$d_p$]")
    ax.set_ylabel("Leave-one-bed-out $R^2$")
    ax.set_xticks(d.depth_dp); ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
    ax.legend(frameon=False); clean_axes(ax); panel_label(ax, "b")
    manifest["sources"]["accessibility_depth"] = [str(depth.attrs.get("source", "depth sensitivity"))]
    save_figure(fig, args.output, "fig_accessibility_depth", args.formats, args.dpi, manifest)


def main():
    args = parse_args()
    set_style()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or ["campaign", "accessibility", "homogeneous", "closure", "depth"])

    closure_path = args.closure_dataset or args.closure_dir / "closure_dataset.csv"
    prediction_path = args.closure_predictions or args.closure_dir / "closure_loo_predictions.csv"
    depth_path = args.depth_sensitivity or args.closure_dir / "closure_depth_sensitivity.csv"

    manifest = {"figures": {}, "sources": {}, "settings": vars(args).copy()}
    manifest["settings"] = {k: str(v) if isinstance(v, Path) else v
                            for k, v in manifest["settings"].items()}
    seed_records = {}

    closure = None
    if selected & {"campaign", "accessibility", "closure"}:
        closure = prepare_seed(read_csv(closure_path, "closure dataset"), str(closure_path))
        closure.attrs["source"] = str(closure_path)
    campaign = None
    if selected & {"campaign", "accessibility"}:
        campaign = qa_filter(prepare_seed(read_csv(args.campaign_dataset, "campaign dataset"),
                                          str(args.campaign_dataset)))
    hom = None
    if "homogeneous" in selected:
        hom = qa_filter(prepare_seed(read_csv(args.homogeneous_summary, "homogeneous summary"),
                                     str(args.homogeneous_summary)))

    if "campaign" in selected:
        figure_campaign(args, campaign, closure, manifest, seed_records)
    if "accessibility" in selected:
        figure_accessibility(args, campaign, closure, manifest)
    if "homogeneous" in selected:
        figure_homogeneous(args, hom, manifest, seed_records)
    if "closure" in selected:
        predictions = prepare_seed(read_csv(prediction_path, "closure predictions"), str(prediction_path))
        predictions.attrs["source"] = str(prediction_path)
        figure_closure(args, closure, predictions, manifest)
    if "depth" in selected:
        depth = read_csv(depth_path, "depth sensitivity")
        depth.attrs["source"] = str(depth_path)
        figure_depth(args, depth, manifest)

    rows = []
    for purpose, levels in seed_records.items():
        for level, seed in levels.items():
            rows.append({"purpose": purpose, "level": level, "seed": seed})
    pd.DataFrame(rows).to_csv(args.output / "representative_seeds.csv", index=False)
    with open(args.output / "figure_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"completed {len(manifest['figures'])} publication figures in {args.output}")


if __name__ == "__main__":
    main()
