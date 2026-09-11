#!/usr/bin/env python3
"""Extract fixed-time CO2 uptake targets and evaluate simple ML baselines.

This is the step before a GNN.  It reads the finite-inlet campaign outputs,
interpolates every uptake curve at identical physical times, merges the targets
with the existing structure table, and evaluates transparent models with
leave-one-packing-out predictions.

No row used for testing is used to fit its model, scaler, imputer or ridge
regularization.  Results are diagnostic because 20 packings are still a small
dataset.

Outputs
-------
co2_transient_structure_dataset.csv
    One row per seed with structural descriptors and fixed-time targets.
co2_baseline_metrics.csv
    Leave-one-out R2, RMSE, normalized RMSE, MAE and rank correlation.
co2_baseline_predictions.csv
    Every held-out prediction, enabling direct inspection and plotting.
co2_inlet_count_residuals.csv
    Residual targets after the inlet-count-only baseline.
co2_baseline_analysis.json
    Settings, selected features, QA and recommended next decision.
co2_baseline_overview.png
    Baseline comparison and q5000 observed-versus-predicted plot.

Example
-------
python3 analyze_co2_transient_baselines.py \
  --results-root co2_adsorption_campaign_finite_inlet_5000s \
  --campaign-dataset co2_adsorption_finite_inlet_analysis/co2_adsorption_campaign_dataset.csv \
  --output-dir co2_transient_baselines
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TIMES = (100.0, 500.0, 1000.0, 2500.0, 5000.0)

# Predeclared features avoid selecting predictors after seeing target results.
RIDGE_FEATURES = (
    "network_inlet_pores",
    "network_analytic_porosity",
    "network_geometric_tortuosity",
    "network_pore_volume_cv",
    "network_mean_throat_radius_m",
    "network_throat_radius_p10_m",
    "network_mean_degree",
    "network_pore_count",
    "network_throat_count",
)

OPTIONAL_TARGETS = (
    "t50_final_uptake_s",
    "normalized_uptake_auc",
    "particle_loading_cv",
    "axial_loading_bottom_minus_top_over_mean",
    "underutilized_particle_fraction",
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--results-root", type=Path, required=True,
                   help="root containing seed_*/timeseries.csv")
    p.add_argument("--campaign-dataset", type=Path, required=True,
                   help="finite-inlet co2_adsorption_campaign_dataset.csv")
    p.add_argument("--output-dir", type=Path,
                   default=Path("co2_transient_baselines"))
    p.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES),
                   help="physical times for interpolated uptake targets [s]")
    p.add_argument("--expected-cases", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=18427)
    p.add_argument("--rf-trees", type=int, default=500)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    if any(t < 0 for t in a.times) or len(set(a.times)) != len(a.times):
        p.error("--times must be unique and nonnegative")
    a.times = sorted(a.times)
    return a


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot determine seed from {path}")
    return int(match.group(1))


def target_name(time_s: float) -> str:
    if float(time_s).is_integer():
        return f"uptake_{int(time_s)}s_mol_kg"
    return f"uptake_{time_s:g}s_mol_kg".replace(".", "p")


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    required = {"time_s", "mean_loading_mol_kg", "adsorbed_inventory_mol"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    time = data["time_s"].to_numpy(float)
    loading = data["mean_loading_mol_kg"].to_numpy(float)
    adsorbed = data["adsorbed_inventory_mol"].to_numpy(float)
    if len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise ValueError(f"invalid time axis in {path}")
    if not np.all(np.isfinite(loading)) or not np.all(np.isfinite(adsorbed)):
        raise ValueError(f"nonfinite uptake values in {path}")
    if np.min(np.diff(loading)) < -max(abs(loading[-1]), 1e-30) * 1e-8:
        raise ValueError(f"nonmonotonic loading in {path}")
    return time, loading, adsorbed


def extract_targets(results_root: Path, times: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metadata_files = sorted(results_root.glob("**/run_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"no run_metadata.json found below {results_root}")
    for metadata_path in metadata_files:
        seed = seed_from_path(metadata_path)
        run_dir = metadata_path.parent
        timeseries = run_dir / "timeseries.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parameters = metadata.get("parameters", {})
        if not timeseries.is_file():
            raise FileNotFoundError(f"missing {timeseries}")
        if not metadata.get("solver_success", False):
            raise ValueError(f"seed {seed} did not report solver success")
        if metadata.get("physical_bounds_ok") is not True:
            raise ValueError(f"seed {seed} did not pass physical bounds")
        if parameters.get("inlet_boundary") != "finite":
            raise ValueError(
                f"seed {seed} is not a finite-inlet production run: "
                f"{parameters.get('inlet_boundary')!r}"
            )
        time, loading, adsorbed = read_curve(timeseries)
        if max(times) > time[-1] + 1e-10:
            raise ValueError(
                f"requested time {max(times):g} s exceeds seed {seed} end time {time[-1]:g} s"
            )
        row: dict[str, Any] = {
            "seed": seed,
            "result_directory": str(run_dir),
            "curve_final_time_s": float(time[-1]),
        }
        for requested in times:
            row[target_name(requested)] = float(np.interp(requested, time, loading))
            row[target_name(requested).replace("uptake_", "adsorbed_").replace("_mol_kg", "_mol")] = float(
                np.interp(requested, time, adsorbed)
            )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if result["seed"].duplicated().any():
        duplicates = result.loc[result["seed"].duplicated(False), "seed"].tolist()
        raise ValueError(f"duplicate result seeds: {duplicates}")
    return result


def validate_and_merge(campaign_path: Path, targets: pd.DataFrame,
                       expected_cases: int) -> pd.DataFrame:
    campaign = pd.read_csv(campaign_path)
    if "seed" not in campaign:
        raise ValueError("campaign dataset has no seed column")
    campaign["seed"] = campaign["seed"].astype(int)
    if campaign["seed"].duplicated().any():
        raise ValueError("campaign dataset contains duplicate seeds")
    if "case_qa_pass" in campaign and not campaign["case_qa_pass"].astype(bool).all():
        bad = campaign.loc[~campaign["case_qa_pass"].astype(bool), "seed"].tolist()
        raise ValueError(f"campaign contains QA failures: {bad}")
    merged = campaign.merge(targets, on="seed", how="outer", validate="one_to_one",
                            indicator=True, suffixes=("", "_curve"))
    if not (merged["_merge"] == "both").all():
        problem = merged.loc[merged["_merge"] != "both", ["seed", "_merge"]]
        raise ValueError(f"campaign/result seed mismatch:\n{problem.to_string(index=False)}")
    merged = merged.drop(columns="_merge").sort_values("seed").reset_index(drop=True)
    if len(merged) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(merged)}")
    return merged


def model_definitions(available: set[str], random_seed: int,
                      rf_trees: int) -> dict[str, tuple[list[str], Any]]:
    def require(names: list[str]) -> list[str]:
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"required baseline features missing: {missing}")
        return names

    ridge_features = [name for name in RIDGE_FEATURES if name in available]
    if len(ridge_features) < 4:
        raise ValueError("fewer than four predefined structural features are available")
    linear = lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])
    ridge = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=KFold(5, shuffle=True,
                                                                  random_state=random_seed))),
    ])
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=rf_trees, max_depth=3, min_samples_leaf=2,
            max_features=0.7, random_state=random_seed, n_jobs=-1,
        )),
    ])
    return {
        "inlet_count_linear": (require(["network_inlet_pores"]), linear()),
        "porosity_linear": (require(["network_analytic_porosity"]), linear()),
        "inlet_plus_tortuosity_linear": (
            require(["network_inlet_pores", "network_geometric_tortuosity"]), linear()
        ),
        "structural_ridge": (ridge_features, ridge),
        "structural_random_forest": (ridge_features, forest),
    }


def loo_mean_predictions(y: np.ndarray) -> np.ndarray:
    total = float(np.sum(y))
    return (total - y) / (len(y) - 1)


def loo_model_predictions(X: np.ndarray, y: np.ndarray, estimator: Any) -> np.ndarray:
    predictions = np.empty_like(y, dtype=float)
    for train, test in LeaveOneOut().split(X):
        fitted = clone(estimator)
        fitted.fit(X[train], y[train])
        predictions[test[0]] = float(fitted.predict(X[test])[0])
    return predictions


def metrics(target: str, model: str, y: np.ndarray, prediction: np.ndarray,
            feature_names: list[str]) -> dict[str, Any]:
    rmse = math.sqrt(mean_squared_error(y, prediction))
    std = float(np.std(y, ddof=1))
    rank = spearmanr(y, prediction)
    return {
        "target": target,
        "model": model,
        "n_cases": len(y),
        "n_features": len(feature_names),
        "features": ";".join(feature_names),
        "loo_r2": float(r2_score(y, prediction)),
        "loo_rmse": rmse,
        "loo_nrmse_by_target_std": rmse / std if std > 0 else math.nan,
        "loo_mae": float(mean_absolute_error(y, prediction)),
        "spearman_r": float(rank.statistic),
        "spearman_p": float(rank.pvalue),
        "target_mean": float(np.mean(y)),
        "target_std": std,
        "target_cv": std / abs(float(np.mean(y))) if np.mean(y) != 0 else math.nan,
    }


def evaluate(dataset: pd.DataFrame, targets: list[str], random_seed: int,
             rf_trees: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = model_definitions(set(dataset.columns), random_seed, rf_trees)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    residual_table = dataset[["seed"]].copy()
    for target in targets:
        if target not in dataset:
            continue
        y = dataset[target].to_numpy(float)
        if not np.all(np.isfinite(y)) or np.std(y, ddof=1) <= 0:
            print(f"WARNING skipping invalid or constant target: {target}")
            continue
        mean_prediction = loo_mean_predictions(y)
        metric_rows.append(metrics(target, "training_mean", y, mean_prediction, []))
        for seed, observed, predicted in zip(dataset["seed"], y, mean_prediction):
            prediction_rows.append({
                "seed": int(seed), "target": target, "model": "training_mean",
                "observed": observed, "predicted": predicted,
                "residual_observed_minus_predicted": observed - predicted,
            })
        for model_name, (features, estimator) in models.items():
            X = dataset[features].to_numpy(float)
            prediction = loo_model_predictions(X, y, estimator)
            metric_rows.append(metrics(target, model_name, y, prediction, features))
            for seed, observed, predicted in zip(dataset["seed"], y, prediction):
                prediction_rows.append({
                    "seed": int(seed), "target": target, "model": model_name,
                    "observed": observed, "predicted": predicted,
                    "residual_observed_minus_predicted": observed - predicted,
                })
            if model_name == "inlet_count_linear":
                residual_table[f"{target}_residual_after_inlet_count"] = y - prediction
    metric_frame = pd.DataFrame(metric_rows).sort_values(["target", "loo_r2"],
                                                         ascending=[True, False])
    predictions = pd.DataFrame(prediction_rows).sort_values(["target", "model", "seed"])
    return metric_frame, predictions, residual_table


def residual_correlations(dataset: pd.DataFrame, residuals: pd.DataFrame) -> list[dict[str, Any]]:
    joined = dataset.merge(residuals, on="seed", validate="one_to_one")
    descriptors = [name for name in RIDGE_FEATURES if name in joined]
    rows = []
    for residual_name in [name for name in joined if name.endswith("_residual_after_inlet_count")]:
        for feature in descriptors:
            x = joined[feature].to_numpy(float)
            y = joined[residual_name].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 4 or np.ptp(x[mask]) == 0:
                continue
            rank = spearmanr(x[mask], y[mask])
            rows.append({
                "residual_target": residual_name,
                "descriptor": feature,
                "spearman_r": float(rank.statistic),
                "spearman_p": float(rank.pvalue),
            })
    return rows


def make_plot(path: Path, metrics_frame: pd.DataFrame,
              predictions: pd.DataFrame, main_target: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable"
    fixed = metrics_frame[metrics_frame["target"].str.startswith("uptake_")].copy()
    models = [
        "inlet_count_linear", "porosity_linear", "inlet_plus_tortuosity_linear",
        "structural_ridge", "structural_random_forest",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    pivot = fixed.pivot(index="target", columns="model", values="loo_r2")
    pivot = pivot[[m for m in models if m in pivot]].sort_index()
    x = np.arange(len(pivot)); width = 0.8 / max(len(pivot.columns), 1)
    for j, model in enumerate(pivot.columns):
        axes[0].bar(x + (j - (len(pivot.columns)-1)/2)*width,
                    pivot[model], width=width, label=model.replace("_", " "))
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_xticks(x, [name.replace("uptake_", "q(").replace("s_mol_kg", " s)")
                           for name in pivot.index], rotation=35, ha="right")
    axes[0].set_ylabel("Leave-one-out $R^2$")
    axes[0].set_title("Fixed-time uptake baselines")
    axes[0].legend(fontsize=7)
    selected = predictions[
        (predictions["target"] == main_target)
        & (predictions["model"].isin(["inlet_count_linear", "structural_ridge",
                                      "structural_random_forest"]))
    ]
    for model, group in selected.groupby("model"):
        axes[1].scatter(group["observed"], group["predicted"], label=model.replace("_", " "))
    values = selected[["observed", "predicted"]].to_numpy(float)
    if values.size:
        lo, hi = float(np.min(values)), float(np.max(values))
        axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1].set_xlabel("Observed uptake [mol/kg]")
    axes[1].set_ylabel("Held-out prediction [mol/kg]")
    axes[1].set_title(main_target.replace("_", " "))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return None


def main() -> int:
    args = arguments()
    curve_targets = extract_targets(args.results_root, args.times)
    dataset = validate_and_merge(args.campaign_dataset, curve_targets,
                                 args.expected_cases)
    fixed_time_targets = [target_name(t) for t in args.times]
    targets = fixed_time_targets + [name for name in OPTIONAL_TARGETS if name in dataset]
    metric_frame, predictions, residuals = evaluate(
        dataset, targets, args.random_seed, args.rf_trees
    )
    residual_corr = residual_correlations(dataset, residuals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "co2_transient_structure_dataset.csv"
    metrics_path = args.output_dir / "co2_baseline_metrics.csv"
    predictions_path = args.output_dir / "co2_baseline_predictions.csv"
    residuals_path = args.output_dir / "co2_inlet_count_residuals.csv"
    residual_corr_path = args.output_dir / "co2_residual_descriptor_correlations.csv"
    json_path = args.output_dir / "co2_baseline_analysis.json"
    plot_path = args.output_dir / "co2_baseline_overview.png"
    dataset.to_csv(dataset_path, index=False)
    metric_frame.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    pd.DataFrame(residual_corr).to_csv(residual_corr_path, index=False)

    best = {}
    for target, group in metric_frame.groupby("target"):
        candidates = group[group["model"] != "training_mean"]
        row = candidates.sort_values("loo_r2", ascending=False).iloc[0]
        best[target] = {
            "model": row["model"], "loo_r2": float(row["loo_r2"]),
            "loo_nrmse_by_target_std": float(row["loo_nrmse_by_target_std"]),
        }
    summary = {
        "results_root": str(args.results_root),
        "campaign_dataset": str(args.campaign_dataset),
        "case_count": len(dataset),
        "fixed_times_s": args.times,
        "cross_validation": "leave one complete packing out",
        "ridge_selection": "RidgeCV inside each outer training fold",
        "predeclared_ridge_features": [name for name in RIDGE_FEATURES if name in dataset],
        "targets": targets,
        "best_nontrivial_baseline_by_target": best,
        "interpretation": (
            "These 20-case results define what a future graph model must beat. "
            "They are not a final estimate of GNN generalization."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    warning = None if args.no_plots else make_plot(
        plot_path, metric_frame, predictions, target_name(max(args.times))
    )

    print(f"Cases: {len(dataset)}; targets: {len(targets)}")
    print("\nBest leave-one-out baseline by target:")
    for target, info in best.items():
        print(f"  {target}: {info['model']}, R2={info['loo_r2']:.4f}, "
              f"NRMSE={info['loo_nrmse_by_target_std']:.4f}")
    if warning:
        print(f"WARNING: {warning}; plot skipped")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
