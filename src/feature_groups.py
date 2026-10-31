"""Engineering feature groups and in-fold feature transformers for Project 1.

Everything that learns from data (correlation filter, PCA, scaler, imputer) lives INSIDE the sklearn
pipeline, so it is re-fitted on the training segments of every fold and never sees the held-out
segment or run. No transformer here uses the target.

Groups are fixed a priori from engineering meaning, not from any model result:

* feed              - programmed feed summaries (+ whether feed is constant or varies in the segment)
* spindle           - programmed spindle-speed summaries (+ constant/variable pattern)
* derived_physical  - quantities computed from feed, spindle and tool geometry
* process_constants - tool/workpiece constants (zero variance in this dataset; kept to show they add nothing)
* design_metadata   - experimental-design descriptors: machining order, geometry, scan axis, run id.
                      These identify WHICH experiment a row came from; they are not machining physics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train import model_catalog

FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "feed": {
        "numeric": ["feed_min_mm_min", "feed_max_mm_min", "feed_mean_mm_min", "feed_range_mm_min"],
        "categorical": ["programmed_feed_pattern"],
    },
    "spindle": {
        "numeric": ["spindle_min_rpm", "spindle_max_rpm", "spindle_mean_rpm", "spindle_range_rpm"],
        "categorical": ["programmed_spindle_pattern"],
    },
    "derived_physical": {
        "numeric": ["feed_per_tooth_mm", "cutting_speed_m_min", "material_removal_rate_proxy"],
        "categorical": [],
    },
    "process_constants": {
        "numeric": ["cutting_depth_mm", "cutting_width_mm", "tool_diameter_mm", "tool_teeth"],
        "categorical": [],
    },
    "design_metadata": {
        "numeric": ["machining_order"],
        "categorical": ["geometry", "cutting_direction_axis", "run_id"],
    },
}
GROUP_ORDER = list(FEATURE_GROUPS)
PROCESS_GROUPS = ["feed", "spindle", "derived_physical", "process_constants"]


def validate_groups(numeric: list[str], categorical: list[str]) -> None:
    """Every original feature must belong to exactly one group, and no group may invent features."""
    seen: list[str] = []
    for spec in FEATURE_GROUPS.values():
        seen += spec["numeric"] + spec["categorical"]
    if len(seen) != len(set(seen)):
        raise ValueError("A feature appears in more than one group.")
    if set(seen) != set(numeric) | set(categorical):
        raise ValueError(
            f"Groups do not partition the feature set. Missing: {sorted((set(numeric) | set(categorical)) - set(seen))}; "
            f"unknown: {sorted(set(seen) - (set(numeric) | set(categorical)))}"
        )


def columns_for(groups: list[str], numeric: list[str], categorical: list[str]) -> tuple[list[str], list[str]]:
    """Columns of the chosen groups, keeping the ORIGINAL column order (identical order => identical models)."""
    num = {c for g in groups for c in FEATURE_GROUPS[g]["numeric"]}
    cat = {c for g in groups for c in FEATURE_GROUPS[g]["categorical"]}
    return [c for c in numeric if c in num], [c for c in categorical if c in cat]


def cumulative_stages() -> list[tuple[str, list[str]]]:
    """Predefined order feed -> spindle -> derived -> constants -> design (not chosen from results)."""
    return [(f"stage{i + 1}_{g}", GROUP_ORDER[: i + 1]) for i, g in enumerate(GROUP_ORDER)]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop constant columns and, greedily in column order, any column with |Pearson r| >= threshold
    to an already-kept column. Label-free; learns its mask from the training data only."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):  # y is deliberately ignored
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        const = X.std(axis=0) < 1e-12
        with np.errstate(all="ignore"):
            corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
        keep = ~const
        for j in range(p):
            if not keep[j]:
                continue
            later = np.abs(corr[j, j + 1 :]) >= self.threshold
            keep[j + 1 :] &= ~later
        if not keep.any():  # degenerate training fold: keep the first column rather than return nothing
            keep[0] = True
        self.keep_ = keep
        self.n_features_in_ = p
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def build_pipeline(
    model_name: str,
    numeric: list[str],
    categorical: list[str],
    *,
    corr_threshold: float | None = None,
    pca_k: int | None = None,
    log_target: bool = False,
):
    """preprocess (+ optional in-fold correlation filter / PCA on the numeric block) -> model."""
    if pca_k is not None and categorical:
        raise ValueError("PCA variants are numeric-only; categorical design fields are not projected.")
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if corr_threshold is not None:
        steps.append(("corr_filter", CorrelationFilter(corr_threshold)))
    if pca_k is not None:
        steps.append(("pca", PCA(n_components=pca_k, svd_solver="full")))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    pipe = Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model_catalog()[model_name])])
    if log_target:
        from sklearn.compose import TransformedTargetRegressor

        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def max_pca_components(n_independent_train_units: int) -> int:
    """Parsimony rule: at most half as many components as independent training units (segments/runs),
    never more than the rank limit n - 1. Rows of one segment are NOT independent, so units, not rows."""
    return max(1, min(n_independent_train_units - 1, n_independent_train_units // 2))
