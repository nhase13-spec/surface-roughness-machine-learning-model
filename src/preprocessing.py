"""Leakage-aware preprocessing and grouped dataset splits."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLUMNS = {
    "ra_gaussian_filtered_um",
    "rq_gaussian_filtered_um",
    "rp_evaluation_um",
    "rv_evaluation_um",
    "rt_evaluation_um",
    "rz_5_section_gaussian_filtered_um",
    "rz_section_1_um",
    "rz_section_2_um",
    "rz_section_3_um",
    "rz_section_4_um",
    "rz_section_5_um",
    "source_ra_unfiltered_um",
    "source_rq_unfiltered_um",
    "source_rz_unfiltered_5_section_um",
    "linear_form_slope_um_per_mm",
    "linear_form_intercept_um",
    "evaluation_mean_offset_um",
    "evaluation_residual_mean_um",
    "roughness_identity_max_abs_error_um",
    "filtered_profile_file",
    "source_profile_file",
    "interpretation_status",
    "profile_key",
    "dataset_source",
}


def identify_duplicates(df: pd.DataFrame, subset: Iterable[str]) -> pd.DataFrame:
    dup = df.duplicated(subset=list(subset), keep=False)
    return df.loc[dup].copy()


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    categorical = set(categorical_features or [])
    for col in X.columns:
        if col in categorical or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))
    return ColumnTransformer(transformers=transformers)


def grouped_cv_splitter(n_groups: int, strategy: str = "leave_one_group_out"):
    if strategy == "leave_one_group_out":
        return LeaveOneGroupOut()
    if strategy == "group_kfold":
        n_splits = min(5, n_groups)
        if n_splits < 2:
            raise ValueError("Not enough groups for GroupKFold.")
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unknown CV strategy: {strategy}")


def holdout_test_indices(
    groups: pd.Series,
    n_holdout: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out entire segment groups for a final test evaluation."""
    unique_groups = groups.unique()
    if n_holdout >= len(unique_groups):
        raise ValueError("Holdout count must be smaller than number of groups.")
    rng = np.random.default_rng(random_state)
    held = rng.choice(unique_groups, size=n_holdout, replace=False)
    test_mask = groups.isin(held).to_numpy()
    train_mask = ~test_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


def outlier_report(df: pd.DataFrame, cols: list[str], z_threshold: float = 3.0) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        z = (s - mu) / sigma
        flagged = s.index[np.abs(z) > z_threshold].tolist()
        rows.append({"column": col, "n_outliers_z_gt_3": len(flagged), "indices": flagged})
    return pd.DataFrame(rows)
