"""Cluster bootstrap confidence intervals for regression metrics.

The three profilograms of a segment share one programmed regime, so rows are NOT independent. The
resampling unit is therefore the machined segment (15 clusters); all rows of a drawn segment are kept
together. Every model is evaluated on the SAME resamples, which makes paired differences (model minus
reference) meaningful: a CI of the difference that excludes 0 on the favourable side is the evidence
that a model beats the reference.

Limits: with 15 segments (and only 5 runs) the intervals are wide and the bootstrap itself is
approximate. It reflects evaluation-set sampling variability only, not hyperparameter or training-set
variability, and it cannot manufacture information that 15 segments do not contain.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst == 0 else 1.0 - float(np.sum((y - p) ** 2)) / sst


def cluster_resamples(clusters: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row-index arrays for each bootstrap draw of whole clusters."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        draws.append(np.concatenate([members[c] for c in chosen]))
    return draws


def _ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_regression_ci(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray,
    *,
    reference: str,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """CIs for MAE/RMSE/R2 per prediction set plus paired MAE/RMSE difference vs ``reference``."""
    y = np.asarray(y, dtype=float)
    preds = {k: np.asarray(v, dtype=float) for k, v in predictions.items()}
    if reference not in preds:
        raise KeyError(f"Reference '{reference}' not among predictions.")
    draws = cluster_resamples(clusters, n_boot, seed)

    stats = {k: {"MAE": [], "RMSE": [], "R2": []} for k in preds}
    for idx in draws:
        yb = y[idx]
        for k, p in preds.items():
            pb = p[idx]
            stats[k]["MAE"].append(_mae(yb, pb))
            stats[k]["RMSE"].append(_rmse(yb, pb))
            stats[k]["R2"].append(_r2(yb, pb))

    out: dict[str, dict[str, Any]] = {}
    ref_mae = np.asarray(stats[reference]["MAE"])
    ref_rmse = np.asarray(stats[reference]["RMSE"])
    for k, p in preds.items():
        point = {"MAE": _mae(y, p), "RMSE": _rmse(y, p), "R2": _r2(y, p)}
        entry: dict[str, Any] = {}
        for m in ("MAE", "RMSE", "R2"):
            arr = np.asarray(stats[k][m])
            lo, hi = _ci(arr, alpha)
            entry[m] = point[m]
            entry[f"{m}_lo"], entry[f"{m}_hi"] = lo, hi
        if k != reference:
            for m, ref_arr in (("MAE", ref_mae), ("RMSE", ref_rmse)):
                delta = np.asarray(stats[k][m]) - ref_arr  # negative = better than reference
                lo, hi = _ci(delta, alpha)
                entry[f"d{m}_vs_ref"] = point[m] - (_mae(y, preds[reference]) if m == "MAE" else _rmse(y, preds[reference]))
                entry[f"d{m}_vs_ref_lo"], entry[f"d{m}_vs_ref_hi"] = lo, hi
            entry["share_better_than_ref"] = float(np.mean((np.asarray(stats[k]["MAE"]) - ref_mae) < 0))
        out[k] = entry
    return out
