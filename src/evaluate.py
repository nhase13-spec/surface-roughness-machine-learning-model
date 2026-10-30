"""Metrics, plots, and export of model evaluation artefacts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def prediction_interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> dict[str, float]:
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    width = y_upper - y_lower
    return {
        "PICP": float(np.mean(inside)),
        "mean_interval_width": float(np.mean(width)),
    }


def save_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, hue=groups, ax=ax, legend=False)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Actual Ra (µm)")
    ax.set_ylabel("Predicted Ra (µm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(resid, kde=True, ax=axes[0])
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Predicted − actual (µm)")
    sns.scatterplot(x=y_pred, y=resid, ax=axes[1])
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("Predicted Ra (µm)")
    axes[1].set_ylabel("Residual (µm)")
    axes[1].set_title("Residuals vs prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = metrics_df.melt(id_vars="model", value_vars=["MAE", "RMSE"], var_name="metric", value_name="value")
    sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Grouped CV model comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_segment_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    path: Path,
) -> None:
    df = pd.DataFrame({"segment_id": groups, "abs_error": np.abs(y_pred - y_true)})
    seg = df.groupby("segment_id")["abs_error"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=seg, x="segment_id", y="abs_error", ax=ax, color="steelblue")
    ax.set_title("Mean absolute error by held-out segment (CV)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
