"""Robustness analyses for Project 1.

1. Validation schemes
   * LOSO - leave-one-segment-out (15 folds): the original protocol.
   * LORO - leave-one-run-out (5 folds): holds out an entire machining run (3 segments, one per
     geometry) at a time. Stricter, because it also removes everything specific to a run (setup,
     tool state, day) from training.
2. Feature sets
   * full               - the original feature set (must reproduce the stored LOSO results exactly).
   * no_design_proxies  - removes ``machining_order``, ``run_id``, ``geometry`` and
     ``cutting_direction_axis``. The last is removed too because in this dataset it is a deterministic
     function of geometry (x-axis <-> straight_2_x), so keeping it would silently re-admit the geometry.
     Trade-off: geometry is partly a genuine physical factor (cutting direction), so removing it removes
     real information as well as a design shortcut. The comparison shows how much the models relied on it.
3. Physics baselines (see src/physics_baseline.py) under both validation schemes.
4. Cluster bootstrap CIs (resampling unit: segment) incl. paired differences vs the mean-predictor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

from src.bootstrap_ci import bootstrap_regression_ci  # noqa: E402
from src.data_loader import build_modeling_frame  # noqa: E402
from src.feature_engineering import add_process_features  # noqa: E402
from src.physics_baseline import PHYSICS_MODELS, PowerLawFz  # noqa: E402
from src.preprocessing import build_feature_matrix  # noqa: E402
from src.train import build_estimator  # noqa: E402
from src.utils import ensure_dirs, load_config, save_json, set_seed  # noqa: E402

DESIGN_PROXY_COLUMNS = ("machining_order", "run_id", "geometry", "cutting_direction_axis")
ML_MODELS = ("dummy_mean", "ridge", "random_forest", "xgboost", "gaussian_process")
REFERENCE_MODEL = "dummy_mean"
VALIDATIONS = {"LOSO": "segment_id", "LORO": "run_id"}
REPRODUCTION_TOLERANCE = 1e-9
ROOT = Path(__file__).resolve().parents[1]


def feature_sets(numeric: list[str], categorical: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    drop = set(DESIGN_PROXY_COLUMNS)
    return {
        "full": (list(numeric), list(categorical)),
        "no_design_proxies": ([c for c in numeric if c not in drop], [c for c in categorical if c not in drop]),
    }


def oof_predict(make_estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Leave-one-group-out out-of-fold predictions; a fresh estimator is fitted per fold."""
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        est = make_estimator()
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = est.predict(X.iloc[test_idx])
    if np.isnan(pred).any():
        raise RuntimeError("Some rows received no out-of-fold prediction.")
    return pred


def _load_frame(cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = build_modeling_frame(cfg)
    df, numeric, categorical = add_process_features(raw)
    target = cfg["data"]["target_column"]
    df = df.loc[pd.to_numeric(df[target], errors="coerce").notna()].reset_index(drop=True)
    return df, numeric, categorical


def run_robustness(n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])
    paths = ensure_dirs(cfg)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df, numeric, categorical = _load_frame(cfg)
    target = cfg["data"]["target_column"]
    y = pd.to_numeric(df[target])
    sets = feature_sets(numeric, categorical)

    oof_records: list[pd.DataFrame] = []
    keyed_preds: dict[str, dict[tuple[str, str, str], np.ndarray]] = {v: {} for v in VALIDATIONS}

    for validation, group_col in VALIDATIONS.items():
        groups = df[group_col]
        for set_name, (num, cat) in sets.items():
            X = build_feature_matrix(df, num + cat, cat)
            for model_name in ML_MODELS:
                print(f"  {validation} | {set_name} | {model_name}", flush=True)
                pred = oof_predict(lambda m=model_name, n=num, c=cat: build_estimator(m, n, c), X, y, groups)
                keyed_preds[validation][(set_name, model_name)] = pred
        X_phys = df[["feed_per_tooth_mm", "tool_diameter_mm"]]
        for phys_name, cls in PHYSICS_MODELS.items():
            print(f"  {validation} | physics | {phys_name}", flush=True)
            keyed_preds[validation][("physics", phys_name)] = oof_predict(lambda c=cls: c(), X_phys, y, groups)

    # ---- bootstrap (same cluster = segment for both validation schemes) -------------------------------
    clusters = df["segment_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    for validation, store in keyed_preds.items():
        # Reference: mean predictor of the SAME validation scheme and the full feature set.
        names = {f"{s}|{m}": p for (s, m), p in store.items()}
        ref_key = f"full|{REFERENCE_MODEL}"
        ci = bootstrap_regression_ci(
            y.to_numpy(), names, clusters, reference=ref_key, n_boot=n_boot, seed=seed, alpha=alpha
        )
        for key, entry in ci.items():
            set_name, model_name = key.split("|")
            row = {"validation": validation, "feature_set": set_name, "model": model_name, **entry}
            rows.append(row)
            oof_records.append(
                pd.DataFrame(
                    {
                        "validation": validation,
                        "feature_set": set_name,
                        "model": model_name,
                        "row": np.arange(len(df)),
                        "segment_id": df["segment_id"],
                        "run_id": df["run_id"],
                        "y_true": y.to_numpy(),
                        "y_pred": names[key],
                    }
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(paths["tables"] / "robustness_summary.csv", index=False)
    pd.concat(oof_records, ignore_index=True).to_csv(paths["metrics"] / "robustness_oof.csv", index=False)

    # ---- reproduction check against the stored original LOSO results -----------------------------------
    stored_path = paths["metrics"] / "cv_model_metrics.csv"
    reproduction = {"checked": False}
    if stored_path.exists():
        stored = pd.read_csv(stored_path).set_index("model")
        diffs = []
        for model_name in ML_MODELS:
            r = summary[(summary.validation == "LOSO") & (summary.feature_set == "full") & (summary.model == model_name)].iloc[0]
            for metric in ("MAE", "RMSE", "R2"):
                diffs.append(abs(float(r[metric]) - float(stored.loc[model_name, metric])))
        reproduction = {
            "checked": True,
            "max_abs_diff": float(max(diffs)),
            "reproduces_original": bool(max(diffs) <= REPRODUCTION_TOLERANCE),
        }

    # ---- descriptive power-law exponent on all 45 profiles (descriptive, not a CV result) -------------
    pl = PowerLawFz().fit(df[["feed_per_tooth_mm", "tool_diameter_mm"]], y)
    rng = np.random.default_rng(seed)
    uniq = df["segment_id"].unique()
    exps = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(df["segment_id"].to_numpy() == c)[0] for c in chosen])
        sub = df.iloc[idx]
        if sub["feed_per_tooth_mm"].nunique() < 2:
            continue
        exps.append(PowerLawFz().fit(sub[["feed_per_tooth_mm", "tool_diameter_mm"]], y.iloc[idx]).exponent_)
    exp_lo, exp_hi = np.percentile(exps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    powerlaw = {
        "fitted_exponent_all_data": float(pl.exponent_),
        "exponent_ci_95": [float(exp_lo), float(exp_hi)],
        "kinematic_expected_exponent": 2.0,
        "note": "Descriptive fit on all 45 profiles; not a validated physical parameter.",
    }

    # ---- uncalibrated physics: dimensional check -------------------------------------------------------
    unc = keyed_preds["LOSO"][("physics", "physics_kinematic_uncalibrated")]
    uncal = {
        "predicted_ra_range_um": [float(unc.min()), float(unc.max())],
        "observed_ra_range_um": [float(y.min()), float(y.max())],
        "median_observed_over_predicted": float(np.median(y.to_numpy() / unc)),
    }

    # ---- same-feed spread: how much Ra varies at (almost) the same fz -------------------------------
    seg = df.groupby("segment_id").agg(run=("run_id", "first"), fz=("feed_per_tooth_mm", "first"), geometry=("geometry", "first"), ra=(target, "mean"))
    run01 = seg[seg.run == "run_01"]
    same_feed = {
        "run_01_identical_programmed_feed_ra_um": {r.geometry: round(float(r.ra), 3) for r in run01.itertuples()},
        "note": "Run 01 used the same programmed feed/spindle for all three segments; Ra differs by geometry.",
    }

    results = {
        "n_boot": n_boot,
        "seed": seed,
        "cluster_unit": "segment_id (15 clusters)",
        "n_profiles": int(len(df)),
        "n_segments": int(df["segment_id"].nunique()),
        "n_runs": int(df["run_id"].nunique()),
        "design_proxy_columns_removed": list(DESIGN_PROXY_COLUMNS),
        "feature_sets": {k: {"numeric": v[0], "categorical": v[1]} for k, v in sets.items()},
        "reproduction_check": reproduction,
        "powerlaw_exponent": powerlaw,
        "uncalibrated_physics": uncal,
        "same_feed_spread": same_feed,
        "summary_rows": json.loads(summary.to_json(orient="records")),
    }
    save_json(results, paths["metrics"] / "robustness_results.json")
    _forest_plot(summary, paths["figures"] / "13_robustness_mae_ci.png")
    _write_report(summary, results, reports_dir / "robustness_report.md")
    print("Reproduction check:", reproduction, flush=True)
    return results


def _forest_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    for ax, (validation, g) in zip(axes, summary.groupby("validation", sort=False)):
        g = g.copy()
        g["label"] = np.where(g.feature_set == "physics", g.model, g.model + " [" + g.feature_set + "]")
        g = g.sort_values("MAE").reset_index(drop=True)
        ref = g[(g.model == REFERENCE_MODEL) & (g.feature_set == "full")].iloc[0]
        colors = ["#7f7f7f" if r.feature_set == "physics" else ("#1f77b4" if r.feature_set == "full" else "#ff7f0e") for r in g.itertuples()]
        y_pos = np.arange(len(g))
        ax.hlines(y_pos, g.MAE_lo, g.MAE_hi, color=colors, lw=2)
        ax.scatter(g.MAE, y_pos, color=colors, zorder=3, s=25)
        ax.axvline(ref.MAE, color="black", ls="--", lw=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(g.label, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("MAE (um), 95% cluster-bootstrap CI over segments")
        ax.set_title(f"{validation}  (dashed = mean predictor)", fontsize=10)
    fig.suptitle("Ra prediction error with uncertainty: blue = full features, orange = no design proxies, grey = physics", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _cell(r: pd.Series, m: str) -> str:
    return f"{r[m]:.3f} [{r[m + '_lo']:.2f}, {r[m + '_hi']:.2f}]"


def _write_report(summary: pd.DataFrame, results: dict[str, Any], path: Path) -> None:
    lines = [
        "# Robustness and uncertainty report (Project 1)",
        "",
        f"Cluster bootstrap over segments (B={results['n_boot']}); 95% percentile intervals. "
        "LOSO = leave-one-segment-out (15 folds), LORO = leave-one-run-out (5 folds).",
        "",
        f"Reproduction of the stored LOSO/full results: {results['reproduction_check']}",
        "",
        "**Caveat.** 15 segments from 5 runs. Intervals are wide and the bootstrap is approximate; "
        "it reflects evaluation sampling only. `no_design_proxies` also removes geometry, which is "
        "partly a genuine physical factor.",
        "",
    ]
    for validation, g in summary.groupby("validation", sort=False):
        lines += [
            f"## {validation}",
            "",
            "| Feature set | Model | MAE [95% CI] | RMSE [95% CI] | R2 [95% CI] | dMAE vs mean-predictor [95% CI] | Share of resamples better |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in g.sort_values("MAE").itertuples(index=False):
            s = pd.Series(r._asdict())
            if pd.isna(s.get("dMAE_vs_ref")):
                dm, share = "reference", "-"
            else:
                dm = f"{s['dMAE_vs_ref']:+.3f} [{s['dMAE_vs_ref_lo']:+.2f}, {s['dMAE_vs_ref_hi']:+.2f}]"
                share = f"{s['share_better_than_ref']:.2f}"
            lines.append(f"| {s['feature_set']} | {s['model']} | {_cell(s, 'MAE')} | {_cell(s, 'RMSE')} | {_cell(s, 'R2')} | {dm} | {share} |")
        lines.append("")
    pl = results["powerlaw_exponent"]
    un = results["uncalibrated_physics"]
    lines += [
        "## Physics baseline notes",
        "",
        f"* Uncalibrated kinematic estimate spans {un['predicted_ra_range_um'][0]:.3f}-{un['predicted_ra_range_um'][1]:.3f} um "
        f"versus observed {un['observed_ra_range_um'][0]:.2f}-{un['observed_ra_range_um'][1]:.2f} um "
        f"(median observed/predicted = {un['median_observed_over_predicted']:.1f}).",
        f"* Power-law exponent of Ra on f_z, fitted on all 45 profiles (descriptive): {pl['fitted_exponent_all_data']:.2f} "
        f"(95% segment-bootstrap CI {pl['exponent_ci_95'][0]:.2f} to {pl['exponent_ci_95'][1]:.2f}); the idealised kinematic model expects 2.",
        f"* Same programmed feed, different geometry (run 01): {results['same_feed_spread']['run_01_identical_programmed_feed_ra_um']} um.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
