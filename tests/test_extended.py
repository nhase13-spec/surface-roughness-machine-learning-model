"""Leakage-prevention and reproducibility tests for the reviewer-driven extensions (Project 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

from src.extended_analysis import (
    CORR_THRESHOLD,
    MODELS,
    breusch_pagan,
    cv_collect,
    fold_signature,
    make_variants,
    oof_group_permutation,
    oof_grouped_shap,
)
from src.feature_groups import (
    FEATURE_GROUPS,
    GROUP_ORDER,
    CorrelationFilter,
    build_pipeline,
    columns_for,
    cumulative_stages,
    max_pca_components,
    validate_groups,
)
from src.preprocessing import build_feature_matrix
from src.robustness import _load_frame
from src.utils import load_config


@pytest.fixture(scope="module")
def real():
    cfg = load_config()
    df, numeric, categorical = _load_frame(cfg)
    target = cfg["data"]["target_column"]
    return df, numeric, categorical, pd.to_numeric(df[target]), target


def _synthetic(n_groups=15, per=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    numeric = [c for g in GROUP_ORDER for c in FEATURE_GROUPS[g]["numeric"]]
    categorical = [c for g in GROUP_ORDER for c in FEATURE_GROUPS[g]["categorical"]]
    df = pd.DataFrame({c: rng.normal(size=n) for c in numeric})
    for c in categorical:
        df[c] = rng.choice(["a", "b", "c"], size=n)
    groups = np.repeat(np.arange(n_groups), per)
    return df, numeric, categorical, groups


# ---------------------------------------------------------------- structure ---------------------------------
def test_groups_partition_the_real_feature_set(real):
    df, numeric, categorical, y, target = real
    validate_groups(numeric, categorical)
    all_cols = set(numeric + categorical)
    assert target not in all_cols
    for forbidden in ("segment_id", "profile_id", "Ra", "Rz", "Rq"):
        assert forbidden not in all_cols


def test_full_group_selection_keeps_original_column_order(real):
    _, numeric, categorical, *_ = real
    num, cat = columns_for(GROUP_ORDER, numeric, categorical)
    assert (num, cat) == (numeric, categorical)


def test_cumulative_stages_are_nested_and_predefined():
    stages = cumulative_stages()
    assert [s[1] for s in stages] == [GROUP_ORDER[: i + 1] for i in range(len(GROUP_ORDER))]


def test_pca_cap_rule():
    assert max_pca_components(14) == 7
    assert max_pca_components(4) == 2
    assert max_pca_components(2) == 1
    assert max_pca_components(1) == 1
    for n in range(2, 40):
        assert max_pca_components(n) <= n // 2 or n < 4


def test_variants_are_predefined_and_pca_numeric_only(real):
    _, numeric, categorical, *_ = real
    variants = make_variants(numeric, categorical)
    names = [v["name"] for v in variants]
    assert len(names) == len(set(names))
    for v in variants:
        if v.get("pca_k"):
            assert v["categorical"] == []
    assert any(v.get("corr_threshold") == CORR_THRESHOLD for v in variants)


# ---------------------------------------------------------------- leakage prevention -------------------------
def test_correlation_filter_is_label_free_and_test_row_independent():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(40, 3))
    X = np.column_stack([base, base[:, 0] * 2 + 1e-3 * rng.normal(size=40), np.ones(40)])
    f1 = CorrelationFilter(0.95).fit(X, rng.normal(size=40))
    f2 = CorrelationFilter(0.95).fit(X, rng.normal(size=40) * 100)  # different labels -> identical mask
    assert np.array_equal(f1.keep_, f2.keep_)
    assert list(f1.keep_) == [True, True, True, False, False]  # near-duplicate and constant dropped
    X_test_a = rng.normal(size=(5, 5))
    X_test_b = X_test_a * 1e6
    assert f1.transform(X_test_a).shape == f1.transform(X_test_b).shape  # test rows cannot change the mask


def test_pca_is_fitted_on_training_rows_only():
    df, numeric, categorical, groups = _synthetic()
    y = pd.Series(np.random.default_rng(2).normal(size=len(df)))
    tr = np.where(groups != 0)[0]
    pipe = build_pipeline("ridge", numeric, [], pca_k=3).fit(df.iloc[tr][numeric], y.iloc[tr])
    num_pipe = pipe.named_steps["preprocess"].named_transformers_["num"]
    pca, scaler = num_pipe.named_steps["pca"], num_pipe.named_steps["scaler"]
    assert pca.n_components_ == 3
    assert scaler.n_samples_seen_ == len(tr)
    # PCA mean of standardised TRAIN rows is ~0 (it would not be ~0 if held-out rows had been included)
    assert np.allclose(pca.mean_, 0, atol=1e-8)
    held = df.iloc[np.where(groups == 0)[0]][numeric].to_numpy()
    assert not np.allclose(scaler.transform(held).mean(axis=0), 0, atol=1e-3)


def test_pipeline_pca_rejects_categorical_inputs():
    with pytest.raises(ValueError):
        build_pipeline("ridge", ["feed_mean_mm_min"], ["geometry"], pca_k=2)


def test_noise_features_with_infold_filter_and_pca_show_no_skill_but_leaky_selection_does():
    rng = np.random.default_rng(3)
    n_groups, per, p = 15, 3, 80
    groups = np.repeat(np.arange(n_groups), per)
    X = pd.DataFrame(rng.normal(size=(n_groups * per, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.normal(size=n_groups * per))
    num = list(X.columns)
    mae = lambda p_: float(np.abs(y.to_numpy() - p_).mean())  # noqa: E731
    mean_oof, _ = cv_collect(lambda: build_pipeline("dummy_mean", num, []), X, y, pd.Series(groups))
    honest, _ = cv_collect(lambda: build_pipeline("ridge", num, [], corr_threshold=0.9, pca_k=3), X, y, pd.Series(groups))
    assert mae(honest) > 0.9 * mae(mean_oof)  # no real skill on pure noise
    # deliberately leaky: choose the 5 features most correlated with y on ALL rows (incl. held-out) first
    leaky_cols = list(X.columns[SelectKBest(f_regression, k=5).fit(X, y).get_support()])
    leaky = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        leaky[te] = Ridge(alpha=1.0).fit(X.iloc[tr][leaky_cols], y.iloc[tr]).predict(X.iloc[te][leaky_cols])
    assert mae(leaky) < mae(honest)  # the harness is sensitive enough to expose a leaky selector


def test_each_fold_holds_out_exactly_one_whole_group(real):
    df, *_ = real
    for col in ("segment_id", "run_id"):
        groups = df[col].to_numpy()
        seen = set()
        for tr, te in LeaveOneGroupOut().split(df, groups=groups):
            assert len(set(groups[te])) == 1
            assert set(groups[te]).isdisjoint(set(groups[tr]))
            seen |= set(te.tolist())
        assert seen == set(range(len(df)))
    # a run contains whole segments: holding out a run can never split a segment
    for run, g in df.groupby("run_id"):
        assert set(g.segment_id).isdisjoint(set(df[df.run_id != run].segment_id))


def test_outer_folds_identical_across_variants(real):
    df, *_ = real
    a, b = fold_signature(df["segment_id"]), fold_signature(df["segment_id"].copy())
    assert a == b
    assert len(fold_signature(df["segment_id"])) == df["segment_id"].nunique()
    assert len(fold_signature(df["run_id"])) == df["run_id"].nunique()


# ---------------------------------------------------------------- reproducibility ----------------------------
def test_cv_collect_is_deterministic_and_reproduces_stored_results(real):
    df, numeric, categorical, y, _ = real
    X = build_feature_matrix(df, numeric + categorical, categorical)
    stored = pd.read_csv(load_dir() / "cv_model_metrics.csv").set_index("model")
    for m in ("ridge", "random_forest"):
        p1, sd1 = cv_collect(lambda m=m: build_pipeline(m, numeric, categorical), X, y, df["segment_id"])
        p2, _ = cv_collect(lambda m=m: build_pipeline(m, numeric, categorical), X, y, df["segment_id"])
        assert np.array_equal(p1, p2)
        assert abs(np.abs(y.to_numpy() - p1).mean() - stored.loc[m, "MAE"]) < 1e-9
        assert np.all(sd1 >= 0)


def load_dir():
    from src.utils import ensure_dirs

    return ensure_dirs(load_config())["metrics"]


# ---------------------------------------------------------------- group importance / SHAP --------------------
def test_group_permutation_finds_the_group_that_generates_the_target():
    df, numeric, categorical, groups = _synthetic(seed=5)
    rng = np.random.default_rng(6)
    y = pd.Series(3.0 * df["feed_mean_mm_min"].to_numpy() + 0.1 * rng.normal(size=len(df)))
    X = build_feature_matrix(df, numeric + categorical, categorical)
    table, _ = oof_group_permutation("ridge", numeric, categorical, X, y, pd.Series(groups), n_repeats=5, seed=0)
    t = table.set_index("group")
    assert t.dMAE_mean.idxmax() == "feed"
    assert t.loc["feed", "dMAE_mean"] > 5 * max(t.drop("feed").dMAE_mean.max(), 1e-6)


def test_grouped_shap_is_additive_and_finds_the_true_group():
    df, numeric, categorical, groups = _synthetic(seed=7)
    rng = np.random.default_rng(8)
    y = pd.Series(2.0 * df["spindle_mean_rpm"].to_numpy() + 0.1 * rng.normal(size=len(df)))
    X = build_feature_matrix(df, numeric + categorical, categorical)
    summary, mat, gap = oof_grouped_shap("xgboost", numeric, categorical, X, y, pd.Series(groups))
    assert gap < 1e-3  # group sums + base value reproduce the model output
    assert summary.set_index("group").mean_abs_grouped_shap.idxmax() == "spindle"
    assert np.isfinite(mat).all()


# ---------------------------------------------------------------- heteroskedasticity test ------------------
def test_breusch_pagan_detects_and_does_not_invent_heteroskedasticity():
    rng = np.random.default_rng(9)
    x = rng.uniform(1, 5, size=600)
    homo = rng.normal(scale=1.0, size=600)
    hetero = rng.normal(scale=1.0, size=600) * x**2
    assert breusch_pagan(homo, x)["p"] > 0.01
    assert breusch_pagan(hetero, x)["p"] < 1e-6
    out = breusch_pagan(hetero, x)
    assert out["LM"] == pytest.approx(out["n"] * out["R2_aux"])
