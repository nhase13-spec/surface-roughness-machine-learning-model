"""End-to-end research pipeline: audit → features → models → report artefacts."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import audit_summary, build_modeling_frame
from src.evaluate import (
    prediction_interval_coverage,
    regression_metrics,
    save_actual_vs_predicted,
    save_model_comparison,
    save_residual_plot,
    save_segment_errors,
)
from src.explainability import shap_feature_importance
from src.feature_engineering import add_process_features, sensor_feature_placeholder
from src.preprocessing import (
    build_feature_matrix,
    holdout_test_indices,
    identify_duplicates,
    outlier_report,
)
from src.train import build_estimator, fit_full_model, grouped_cv_predictions, model_catalog
from src.utils import ensure_dirs, load_config, save_json, set_seed


def eda_figures(df: pd.DataFrame, paths: dict[str, Path], target: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[target], kde=True, ax=ax)
    ax.set_xlabel("Gaussian-filtered Ra (µm)")
    ax.set_title("Distribution of measured Ra")
    fig.savefig(paths["figures"] / "01_ra_distribution.png", dpi=150)
    plt.close(fig)

    proc_cols = ["feed_mean_mm_min", "spindle_mean_rpm", "feed_per_tooth_mm", "cutting_speed_m_min"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, col in zip(axes.ravel(), proc_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
    fig.suptitle("Machining parameter distributions")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "02_parameter_distributions.png", dpi=150)
    plt.close(fig)

    num = df[proc_cols + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap (process + Ra label)")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="feed_mean_mm_min", y=target, hue="geometry", ax=ax)
    ax.set_title("Ra vs programmed feed")
    fig.savefig(paths["figures"] / "04_ra_vs_feed.png", dpi=150)
    plt.close(fig)

    rep = df.groupby("segment_id")[target].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=rep, x="segment_id", y="std", ax=ax, color="gray")
    ax.set_title("Within-segment Ra variability (std across 3 profilograms)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "05_within_segment_variability.png", dpi=150)
    plt.close(fig)

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 4))
        miss.plot(kind="bar", ax=ax)
        ax.set_ylabel("Missing fraction")
        ax.set_title("Missing data by column")
        fig.tight_layout()
        fig.savefig(paths["figures"] / "06_missing_data.png", dpi=150)
        plt.close(fig)

def write_audit_report(audit: dict, df: pd.DataFrame, paths: dict[str, Path]) -> None:
    lines = [
        "# Dataset audit report",
        "",
        "## Source",
        "- Dataset: Experimental Surface Roughness Dataset for CNC End Milling (straight segments)",
        "- DOI: 10.5281/zenodo.21518387 (CC BY 4.0)",
        "- Material: St3ps carbon structural steel; 12 mm 4-flute coated end mill",
        "- Operation: finish end milling; programmed feed/spindle from CNC (not independently measured)",
        "",
        "## Counts",
        f"- Independent machined segments: **{audit['n_segments']}**",
        f"- Surface profilograms (profiles): **{audit['n_profiles']}**",
        f"- Machining runs: **{audit['n_runs']}**",
        "",
        "## Labels and inputs",
        f"- Primary target: `{audit['target_column']}` (Gaussian L-filter, λc=0.8 mm, central 4 mm evaluation)",
        "- Predictors: programmed feed/spindle bands, tool/workpiece constants, geometry/run categoricals",
        "- **No in-process sensor signals** are included in this public release.",
        "",
        "## Quality checks",
        f"- Missing target values: {audit['missing_target']}",
        f"- Duplicate profile keys: {audit['duplicate_profile_keys']}",
        f"- Ra range (µm): {audit['target_range_um'][0]:.3f} – {audit['target_range_um'][1]:.3f}",
        "",
        "## Validity considerations",
        "- Three repeated profilograms per segment share the same programmed regime → use grouped CV by `segment_id`.",
        "- Variable feed/spindle within a segment is summarized by min/max/mean; true instantaneous values are not logged.",
        "- Geometry (scan axis vs feed) affects interpretability of texture; do not over-interpret spectral diagnostics.",
        "- Small segment count (15) limits hyperparameter tuning and external generalisation claims.",
        "",
    ]
    (paths["reports"] / "dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(audit, paths["tables"] / "dataset_audit.json")


def gpr_uncertainty_on_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    pipe = build_estimator("gaussian_process", numeric_features, categorical_features)
    pipe.fit(X_train, y_train)
    model: object = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]
    Xt = pre.transform(X_test)
    y_pred, y_std = model.predict(Xt, return_std=True)
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    cov = prediction_interval_coverage(y_test.to_numpy(), lower, upper)
    cov.update(regression_metrics(y_test.to_numpy(), y_pred))
    return cov


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])
    paths = ensure_dirs(cfg)
    paths["reports"] = ROOT / "reports"
    paths["reports"].mkdir(parents=True, exist_ok=True)

    raw = build_modeling_frame(cfg)
    raw.to_csv(paths["interim"] / "profile_modeling_frame.csv", index=False)

    df, numeric_features, categorical_features = add_process_features(raw)
    target = cfg["data"]["target_column"]
    group_col = cfg["data"]["group_column"]

    audit = audit_summary(df, cfg)
    write_audit_report(audit, df, paths)
    eda_figures(df, paths, target)

    dup = identify_duplicates(df, ["segment_id", "profile_id"])
    dup.to_csv(paths["tables"] / "duplicate_check.csv", index=False)
    out_rep = outlier_report(df, [target] + numeric_features)
    out_rep.to_csv(paths["tables"] / "outlier_report.csv", index=False)

    feature_cols = numeric_features + categorical_features
    X = build_feature_matrix(df, feature_cols, categorical_features)
    y = pd.to_numeric(df[target], errors="coerce")
    groups = df[group_col]
    valid = y.notna()
    X, y, groups = X.loc[valid], y.loc[valid], groups.loc[valid]

    df.loc[valid].to_csv(paths["processed"] / "modeling_dataset.csv", index=False)

    cv_strategy = cfg["modeling"]["cv_strategy"]
    metrics_rows = []
    preds_store = {}

    for name in model_catalog():
        est = build_estimator(name, numeric_features, categorical_features)
        pred = grouped_cv_predictions(est, X, y, groups, cv_strategy)
        m = regression_metrics(y.to_numpy(), pred)
        m["model"] = name
        m["evaluation"] = "grouped_cv"
        metrics_rows.append(m)
        preds_store[name] = pred

    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
    metrics_df.to_csv(paths["metrics"] / "cv_model_metrics.csv", index=False)
    save_json(metrics_rows, paths["metrics"] / "cv_model_metrics.json")

    best = metrics_df.iloc[0]["model"]
    save_model_comparison(metrics_df, paths["figures"] / "08_model_comparison.png")
    save_actual_vs_predicted(
        y.to_numpy(),
        preds_store[best],
        groups,
        f"Grouped CV — best model ({best})",
        paths["figures"] / f"09_actual_vs_pred_{best}.png",
    )
    save_residual_plot(y.to_numpy(), preds_store[best], paths["figures"] / "10_residuals_best.png")
    save_segment_errors(y.to_numpy(), preds_store[best], groups, paths["figures"] / "11_segment_errors.png")

    n_hold = cfg["modeling"]["test_holdout_groups"]
    train_idx, test_idx = holdout_test_indices(groups, n_hold, cfg["project"]["random_seed"])
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_test = groups.iloc[test_idx]

    holdout_rows = []
    for name in ["ridge", "random_forest", "xgboost"]:
        pipe = fit_full_model(name, X_train, y_train, numeric_features, categorical_features)
        pred_test = pipe.predict(X_test)
        m = regression_metrics(y_test.to_numpy(), pred_test)
        m["model"] = name
        m["evaluation"] = "holdout_segments"
        holdout_rows.append(m)
        pipe_path = paths["models"] / f"{name}_holdout.joblib"
        import joblib

        joblib.dump(pipe, pipe_path)

    gpr_cov = gpr_uncertainty_on_holdout(
        X_train, y_train, X_test, y_test, numeric_features, categorical_features
    )
    gpr_cov["model"] = "gaussian_process"
    gpr_cov["evaluation"] = "holdout_segments"
    holdout_rows.append(gpr_cov)

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(paths["metrics"] / "holdout_test_metrics.csv", index=False)

    xgb_pipe = fit_full_model("xgboost", X, y, numeric_features, categorical_features)
    shap_imp = shap_feature_importance(
        xgb_pipe,
        X,
        numeric_features,
        categorical_features,
        paths["figures"] / "12_shap_xgb.png",
    )
    shap_imp.to_csv(paths["tables"] / "shap_importance_xgb.csv", index=False)

    sensor_feature_placeholder()
    save_json(sensor_feature_placeholder(), paths["tables"] / "sensor_extension.json")

    summary = {
        "best_cv_model": best,
        "cv_metrics": metrics_rows,
        "holdout_metrics": holdout_rows,
        "sensor_signals": False,
    }
    save_json(summary, paths["metrics"] / "pipeline_summary.json")
    print("Pipeline complete. Best grouped-CV model:", best)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
