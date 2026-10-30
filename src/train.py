"""Train baseline and advanced regressors with grouped cross-validation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import grouped_cv_splitter, make_preprocessor


def model_catalog(predefined: bool = True) -> dict[str, Any]:
    """Conservative hyperparameters for n=15 segment groups."""
    if predefined:
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
        )
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=42,
            objective="reg:squarederror",
        )
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=2)
    else:
        raise NotImplementedError("Nested tuning disabled for small group count.")

    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": rf,
        "xgboost": xgb,
        "gaussian_process": gpr,
    }


def build_estimator(model_name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    pre = make_preprocessor(numeric_features, categorical_features)
    reg = model_catalog()[model_name]
    return Pipeline(steps=[("preprocess", pre), ("model", reg)])


def grouped_cv_predictions(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    strategy: str,
) -> np.ndarray:
    cv = grouped_cv_splitter(groups.nunique(), strategy=strategy)
    return cross_val_predict(estimator, X, y, cv=cv, groups=groups, n_jobs=None)


def fit_full_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    pipe = build_estimator(model_name, numeric_features, categorical_features)
    pipe.fit(X, y)
    return pipe
