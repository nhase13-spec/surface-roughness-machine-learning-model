"""Tests for the robustness analyses: physics baselines, run-level grouping, proxy removal, bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from src.bootstrap_ci import bootstrap_regression_ci, cluster_resamples
from src.feature_engineering import add_process_features
from src.physics_baseline import KinematicCalibrated, KinematicUncalibrated, PowerLawFz, kinematic_ra_um
from src.robustness import DESIGN_PROXY_COLUMNS, feature_sets


def test_kinematic_formula_known_value():
    # fz = 0.1 mm, D = 12 mm -> 1000 * 0.01 / (16 * 12) = 0.052083 um
    assert kinematic_ra_um(0.1, 12.0) == pytest.approx(1000 * 0.01 / 192.0)


def test_kinematic_uncalibrated_ignores_targets():
    X = pd.DataFrame({"feed_per_tooth_mm": [0.05, 0.1], "tool_diameter_mm": [12.0, 12.0]})
    a = KinematicUncalibrated().fit(X, [100.0, 200.0]).predict(X)
    b = KinematicUncalibrated().fit(X, [1.0, 2.0]).predict(X)
    np.testing.assert_allclose(a, b)


def test_calibrated_recovers_linear_law_and_powerlaw_recovers_exponent():
    fz = np.linspace(0.04, 0.15, 12)
    X = pd.DataFrame({"feed_per_tooth_mm": fz, "tool_diameter_mm": 12.0})
    y = 0.3 + 40.0 * fz**2 / 12.0
    pred = KinematicCalibrated().fit(X, y).predict(X)
    np.testing.assert_allclose(pred, y, atol=1e-9)
    pl = PowerLawFz().fit(X, 2.0 * fz**2)
    assert pl.exponent_ == pytest.approx(2.0, abs=1e-9)


def test_physics_models_reject_missing_inputs():
    X = pd.DataFrame({"feed_per_tooth_mm": [0.05, np.nan], "tool_diameter_mm": [12.0, 12.0]})
    with pytest.raises(ValueError):
        KinematicUncalibrated().predict(X)


def test_design_proxies_are_removed_including_axis():
    num = ["feed_mean_mm_min", "machining_order"]
    cat = ["geometry", "cutting_direction_axis", "run_id", "programmed_feed_pattern"]
    sets = feature_sets(num, cat)
    for col in DESIGN_PROXY_COLUMNS:
        assert col not in sets["no_design_proxies"][0] + sets["no_design_proxies"][1]
    assert "feed_mean_mm_min" in sets["no_design_proxies"][0]
    assert "programmed_feed_pattern" in sets["no_design_proxies"][1]
    assert sets["full"] == (num, cat)  # untouched original


def test_leave_one_run_out_holds_out_whole_runs():
    groups = pd.Series(np.repeat([f"run_0{i}" for i in range(1, 6)], 9))  # 5 runs x 9 profiles
    folds = list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))
    assert len(folds) == 5
    for train, test in folds:
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
        assert len(test) == 9


def test_cluster_resamples_keep_clusters_whole():
    clusters = np.repeat(np.arange(5), 3)
    for idx in cluster_resamples(clusters, n_boot=20, seed=1):
        counts = pd.Series(clusters[idx]).value_counts()
        assert (counts % 3 == 0).all()  # each drawn cluster contributes all 3 rows


def test_bootstrap_paired_difference_favours_better_model():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(15), 3)
    y = rng.normal(2.0, 1.0, 45) + np.repeat(rng.normal(0, 1, 15), 3)
    good = y + rng.normal(0, 0.1, 45)
    bad = np.full(45, y.mean())
    out = bootstrap_regression_ci(y, {"bad": bad, "good": good}, clusters, reference="bad", n_boot=300, seed=2)
    assert out["good"]["dMAE_vs_ref_hi"] < 0  # CI of (good - reference) lies entirely below zero
    assert out["good"]["MAE_lo"] <= out["good"]["MAE"] <= out["good"]["MAE_hi"]


def test_add_process_features_still_lists_proxy_columns():
    raw = pd.DataFrame(
        {
            "programmed_feed_min_mm_min": [138.0],
            "programmed_feed_max_mm_min": [138.0],
            "programmed_spindle_min_rpm": [822.0],
            "programmed_spindle_max_rpm": [822.0],
            "tool_teeth": [4],
            "tool_diameter_mm": [12.0],
            "cutting_depth_mm": [1.5],
            "cutting_width_mm": [5.0],
        }
    )
    _, num, cat = add_process_features(raw)
    assert "machining_order" in num and "run_id" in cat and "geometry" in cat
