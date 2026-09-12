#!/usr/bin/env python3
"""Formalize a structure-to-effective-diffusivity closure for the CO2 bed.

The proposed physical closure is

    D_eff = alpha * D_access,
    D_access = G_access * H / A_tube,

where G_access is the network conductance from the inlet to an interior
plane, H is bed height, and A_tube is the full tube cross-sectional area.
The script compares this closure with alternatives, evaluates held-out-bed
predictions, quantifies uncertainty, and tests sensitivity to interior depth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


YCOL = "effective_diffusivity_m2_s"
GCOL = "effective_inlet_to_interior_conductance_m3_s"


def arguments():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fit and validate a pore-network accessibility closure for D_eff.",
    )
    p.add_argument("--homogeneous-summary", required=True, type=Path)
    p.add_argument("--accessibility", required=True, nargs="+", type=Path)
    p.add_argument("--primary-depth", type=float, default=5.0,
                   help="Interior-plane depth in particle diameters used for the main closure.")
    p.add_argument("--tube-radius", type=float, default=0.008, help="Tube radius [m].")
    p.add_argument("--output", type=Path, default=Path("co2_effective_diffusivity_closure"))
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--random-seed", type=int, default=20260824)
    return p.parse_args()


def numeric_seed(s):
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def read_homogeneous(path):
    df = pd.read_csv(path)
    required = {"seed", "bed_height_m", "porosity", YCOL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df.copy()
    df["seed"] = numeric_seed(df["seed"])
    qa_column = "qa_pass" if "qa_pass" in df else ("case_qa_pass" if "case_qa_pass" in df else None)
    if qa_column:
        ok = df[qa_column].astype(str).str.lower().isin(["true", "1", "yes"])
        if not ok.any():
            raise ValueError("No homogeneous cases passed QA")
        df = df[ok].copy()
    return df


def infer_depth(df, path):
    if "interior_depth_dp" in df and df["interior_depth_dp"].notna().any():
        vals = pd.to_numeric(df["interior_depth_dp"], errors="coerce").dropna().unique()
        if len(vals) != 1:
            raise ValueError(f"{path}: expected one interior_depth_dp, found {vals}")
        return float(vals[0])
    import re
    match = re.search(r"(?:_|-)(\d+(?:\.\d+)?)dp", path.stem.lower())
    if not match:
        raise ValueError(f"Cannot infer depth from {path}; add interior_depth_dp column.")
    return float(match.group(1))


def read_accessibility(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        required = {"seed", GCOL}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df = df.copy()
        df["seed"] = numeric_seed(df["seed"])
        df["depth_dp"] = infer_depth(df, path)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    if all_df.duplicated(["seed", "depth_dp"]).any():
        raise ValueError("Duplicate seed/depth rows in accessibility inputs.")
    return all_df


def fit_predict(kind, x_train, y_train, x_test):
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    x_test = np.asarray(x_test, float)
    if kind == "linear":
        slope, intercept = np.polyfit(x_train, y_train, 1)
        return intercept + slope * x_test, {"intercept": intercept, "slope": slope}
    if kind == "origin":
        alpha = float(x_train @ y_train / (x_train @ x_train))
        return alpha * x_test, {"alpha": alpha}
    if kind == "power":
        if np.any(x_train <= 0) or np.any(y_train <= 0) or np.any(x_test <= 0):
            raise ValueError("Power-law fit requires positive x and y.")
        exponent, log_c = np.polyfit(np.log(x_train), np.log(y_train), 1)
        coefficient = float(np.exp(log_c))
        return coefficient * x_test ** exponent, {
            "coefficient": coefficient, "exponent": exponent
        }
    raise ValueError(kind)


def loo_predictions(kind, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pred = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        pred[i] = fit_predict(kind, x[keep], y[keep], x[[i]])[0][0]
    return pred


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    rmse = math.sqrt(mean_squared_error(y, pred))
    scale = np.std(y, ddof=1)
    return {
        "loo_r2": r2_score(y, pred),
        "loo_rmse_m2_s": rmse,
        "loo_nrmse_std": rmse / scale if scale > 0 else np.nan,
        "loo_mae_m2_s": mean_absolute_error(y, pred),
        "loo_mape_percent": 100 * np.mean(np.abs((pred - y) / y)),
    }


def safe_correlations(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return pr, pp, sr, sp


def model_specs(primary):
    return [
        ("porosity_linear", "porosity", "linear"),
        ("raw_conductance_linear", GCOL, "linear"),
        ("accessibility_linear", "accessibility_diffusivity_m2_s", "linear"),
        ("physical_through_origin", "accessibility_diffusivity_m2_s", "origin"),
        ("accessibility_power_law", "accessibility_diffusivity_m2_s", "power"),
    ]


def bootstrap_coefficients(df, specs, nboot, rng):
    rows = []
    n = len(df)
    for model, xcol, kind in specs:
        samples = {}
        for _ in range(nboot):
            take = rng.integers(0, n, n)
            x = df[xcol].to_numpy()[take]
            y = df[YCOL].to_numpy()[take]
            try:
                _, pars = fit_predict(kind, x, y, x[:1])
            except (ValueError, np.linalg.LinAlgError):
                continue
            for key, value in pars.items():
                samples.setdefault(key, []).append(value)
        for parameter, values in samples.items():
            a = np.asarray(values)
            rows.append({
                "model": model, "parameter": parameter, "valid_bootstraps": len(a),
                "median": np.median(a), "ci_2p5": np.quantile(a, .025),
                "ci_97p5": np.quantile(a, .975),
            })
    return pd.DataFrame(rows)


def permutation_tests(df, specs, observed_metrics, nperm, rng):
    rows = []
    y = df[YCOL].to_numpy()
    for model, xcol, kind in specs:
        x = df[xcol].to_numpy()
        obs = observed_metrics[model]["loo_rmse_m2_s"]
        better = 0
        for _ in range(nperm):
            yp = rng.permutation(y)
            null_rmse = math.sqrt(mean_squared_error(yp, loo_predictions(kind, x, yp)))
            better += null_rmse <= obs
        rows.append({"model": model, "test_statistic": "LOOCV_RMSE",
                     "observed_rmse_m2_s": obs, "permutations": nperm,
                     "p_value": (better + 1) / (nperm + 1)})
    return pd.DataFrame(rows)


def depth_sensitivity(merged):
    """Refit every closure on nineteen beds at each diagnostic depth."""
    depth_rows = []
    for depth, group in merged.groupby("depth_dp"):
        group = group.dropna(subset=[YCOL, GCOL, "accessibility_diffusivity_m2_s"])
        x, y = group["accessibility_diffusivity_m2_s"].to_numpy(), group[YCOL].to_numpy()
        pred0 = loo_predictions("origin", x, y)
        pred1 = loo_predictions("linear", x, y)
        pred_power = loo_predictions("power", x, y)
        pred_raw = loo_predictions("linear", group[GCOL].to_numpy(), y)
        pr, pp, sr, sp = safe_correlations(x, y)
        alpha = float(x @ y / (x @ x))
        depth_rows.append({"depth_dp": depth, "n_beds": len(group),
                           "pearson_r": pr, "pearson_p": pp,
                           "spearman_rho": sr, "spearman_p": sp,
                           "alpha_through_origin": alpha,
                           **{f"origin_{k}": v for k, v in metrics(y, pred0).items()},
                           **{f"linear_{k}": v for k, v in metrics(y, pred1).items()},
                           **{f"power_{k}": v for k, v in metrics(y, pred_power).items()},
                           **{f"raw_linear_{k}": v for k, v in metrics(y, pred_raw).items()}})
    depth_df = pd.DataFrame(depth_rows).sort_values("depth_dp")

    return depth_df


def main():
    args = arguments()
    if args.tube_radius <= 0:
        raise ValueError("--tube-radius must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    area = math.pi * args.tube_radius ** 2
    hom = read_homogeneous(args.homogeneous_summary)
    access = read_accessibility(args.accessibility)
    merged = access.merge(hom, on="seed", how="inner", validate="many_to_one")
    merged["tube_area_m2"] = area
    merged["accessibility_diffusivity_m2_s"] = merged[GCOL] * merged["bed_height_m"] / area
    depths = sorted(merged["depth_dp"].unique())
    if not any(np.isclose(depths, args.primary_depth)):
        raise ValueError(f"Primary depth {args.primary_depth:g} not found; available: {depths}")
    primary = merged[np.isclose(merged["depth_dp"], args.primary_depth)].copy()
    primary = primary.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["seed", "porosity", "bed_height_m", YCOL, GCOL,
                "accessibility_diffusivity_m2_s"])
    if len(primary) < 8:
        raise ValueError(f"Only {len(primary)} complete beds at primary depth; at least 8 required.")
    if primary["seed"].nunique() != hom["seed"].nunique():
        missing = sorted(set(hom.seed.dropna().astype(int)) - set(primary.seed.astype(int)))
        raise ValueError(f"Accessibility data missing homogeneous seeds: {missing}")

    specs = model_specs(primary)
    metric_rows, pred_rows, coefficient_rows = [], [], []
    observed = {}
    for model, xcol, kind in specs:
        x, y = primary[xcol].to_numpy(), primary[YCOL].to_numpy()
        pred = loo_predictions(kind, x, y)
        met = metrics(y, pred)
        observed[model] = met
        metric_rows.append({"model": model, "predictor": xcol, "form": kind,
                            "n_beds": len(y), **met})
        full_pred, pars = fit_predict(kind, x, y, x)
        for name, value in pars.items():
            coefficient_rows.append({"model": model, "parameter": name, "estimate": value})
        for seed, obs, loo, fitted in zip(primary.seed.astype(int), y, pred, full_pred):
            pred_rows.append({"seed": seed, "model": model, "observed_D_eff_m2_s": obs,
                              "loo_predicted_D_eff_m2_s": loo,
                              "full_fit_D_eff_m2_s": fitted,
                              "loo_residual_m2_s": obs - loo})
    metrics_df = pd.DataFrame(metric_rows).sort_values("loo_rmse_m2_s")
    predictions_df = pd.DataFrame(pred_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    rng = np.random.default_rng(args.random_seed)
    bootstrap_df = bootstrap_coefficients(primary, specs, args.bootstrap, rng)
    permutation_df = permutation_tests(primary, specs, observed, args.permutations, rng)

    depth_df = depth_sensitivity(merged)

    physical_pred = predictions_df[predictions_df.model == "physical_through_origin"].copy()
    residual_map = physical_pred.set_index("seed")["loo_residual_m2_s"]
    diag = primary.copy()
    diag["closure_loo_residual_m2_s"] = diag.seed.astype(int).map(residual_map)
    diagnostic_candidates = ["porosity", "bed_height_m", GCOL,
                             "accessibility_diffusivity_m2_s",
                             "fit_normalized_rmse", "final_loading_relative_error"]
    residual_rows = []
    for col in diagnostic_candidates:
        if col not in diag:
            continue
        pair = diag[[col, "closure_loo_residual_m2_s"]].dropna()
        pr, pp, sr, sp = safe_correlations(pair[col], pair["closure_loo_residual_m2_s"])
        residual_rows.append({"descriptor": col, "n": len(pair), "pearson_r": pr,
                              "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp})
    residual_df = pd.DataFrame(residual_rows)

    primary.to_csv(args.output / "closure_dataset.csv", index=False)
    metrics_df.to_csv(args.output / "closure_model_metrics.csv", index=False)
    predictions_df.to_csv(args.output / "closure_loo_predictions.csv", index=False)
    coefficients_df.to_csv(args.output / "closure_coefficients.csv", index=False)
    bootstrap_df.to_csv(args.output / "closure_bootstrap_intervals.csv", index=False)
    permutation_df.to_csv(args.output / "closure_permutation_tests.csv", index=False)
    depth_df.to_csv(args.output / "closure_depth_sensitivity.csv", index=False)
    residual_df.to_csv(args.output / "closure_residual_correlations.csv", index=False)

    x = primary["accessibility_diffusivity_m2_s"].to_numpy()
    y = primary[YCOL].to_numpy()
    alpha = coefficients_df.query(
        "model == 'physical_through_origin' and parameter == 'alpha'"
    )["estimate"].iloc[0]
    xx = np.linspace(0, 1.05 * x.max(), 200)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(x, y, s=42, color="#176B87", label="20 packed beds")
    ax.plot(xx, alpha * xx, color="#B03A2E", lw=2,
            label=rf"$D_{{eff}}={alpha:.3f}D_{{access}}$")
    ax.set(xlabel=r"Accessibility diffusivity $D_{access}=G_{access}H/A$ [m$^2$/s]",
           ylabel=r"Fitted homogeneous $D_{eff}$ [m$^2$/s]")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "closure_physical_fit.png", dpi=220)
    fig.savefig(args.output / "closure_physical_fit.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for model in ["porosity_linear", "physical_through_origin", "accessibility_linear"]:
        p = predictions_df[predictions_df.model == model]
        ax.scatter(p.observed_D_eff_m2_s, p.loo_predicted_D_eff_m2_s,
                   s=35, alpha=.8, label=model.replace("_", " "))
    lim = [0.95 * y.min(), 1.05 * y.max()]
    ax.plot(lim, lim, "k--", lw=1, label="ideal")
    ax.set(xlim=lim, ylim=lim, xlabel=r"Observed $D_{eff}$ [m$^2$/s]",
           ylabel=r"Leave-one-bed-out prediction [m$^2$/s]")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(args.output / "closure_loo_predictions.png", dpi=220)
    fig.savefig(args.output / "closure_loo_predictions.pdf")
    plt.close(fig)

    physical_metrics = metrics_df.query("model == 'physical_through_origin'").iloc[0]
    best = metrics_df.iloc[0]
    alpha_ci = bootstrap_df.query(
        "model == 'physical_through_origin' and parameter == 'alpha'"
    ).iloc[0]
    porosity_metrics = metrics_df.query("model == 'porosity_linear'").iloc[0]
    power_pars = coefficients_df.query("model == 'accessibility_power_law'").set_index("parameter")["estimate"]
    power_c = float(power_pars["coefficient"])
    power_n = float(power_pars["exponent"])
    summary = {
        "n_beds": int(len(primary)), "tube_radius_m": args.tube_radius,
        "tube_area_m2": area, "primary_depth_dp": args.primary_depth,
        "closure": "D_eff = alpha * (G_access * H / A_tube)",
        "alpha": float(alpha),
        "alpha_bootstrap_95_percent_ci": [float(alpha_ci.ci_2p5), float(alpha_ci.ci_97p5)],
        "physical_closure_loo_r2": float(physical_metrics.loo_r2),
        "physical_closure_loo_rmse_m2_s": float(physical_metrics.loo_rmse_m2_s),
        "porosity_only_loo_r2": float(porosity_metrics.loo_r2),
        "best_predictive_model": str(best.model),
        "best_model_loo_r2": float(best.loo_r2),
        "positive_power_closure": "D_eff = C * (G_access * H / A_tube)^n",
        "power_coefficient_C": power_c,
        "power_exponent_n": power_n,
        "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
        "random_seed": args.random_seed,
    }
    (args.output / "closure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = f"""# Effective-diffusivity closure analysis

## Closure candidates

The simplest dimensionally consistent, one-parameter closure is

`D_eff = alpha D_access`, with `D_access = G_access H / A_tube`.

Here, `G_access` is the pore-network conductance from the inlet to an interior
plane at {args.primary_depth:g} particle diameters, `H` is bed height, and
`A_tube = pi R^2` is the full tube area. The fitted dimensionless factor is
`alpha = {alpha:.6g}` with a bed-bootstrap 95% interval of
`[{alpha_ci.ci_2p5:.6g}, {alpha_ci.ci_97p5:.6g}]`.

A positive empirical alternative is `D_eff = C D_access^n`, for which this
ensemble gives `C = {power_c:.6g}` and `n = {power_n:.6g}`. Because `C` carries
the units needed by the fitted exponent, its value is tied to SI units. The
model-comparison table should be used to decide whether its added exponent is
justified over the one-parameter closure.

## Held-out-bed validation

The physical closure gives leave-one-bed-out `R2 = {physical_metrics.loo_r2:.4f}`
and `RMSE = {physical_metrics.loo_rmse_m2_s:.4e} m2/s`. A porosity-only linear
model gives `R2 = {porosity_metrics.loo_r2:.4f}`. The best tested predictive
form is `{best.model}` with `R2 = {best.loo_r2:.4f}`.

These statistics describe this 20-bed ensemble. They support a structural
closure inside the studied geometry and parameter range; they do not yet prove
transfer to other tube-to-particle ratios, particle sizes, or sorbents.
"""
    (args.output / "closure_summary.md").write_text(md)

    print(f"Analysed {len(primary)} beds at primary depth {args.primary_depth:g} dp")
    print(f"Closure: D_eff = {alpha:.6g} * (G_access H / A_tube)")
    print(metrics_df[["model", "loo_r2", "loo_rmse_m2_s", "loo_mape_percent"]].to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
