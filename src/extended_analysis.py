"""Reviewer-driven controlled extensions for Project 1 (surface roughness). Hypothesis-generating only.

Design principles
-----------------
* All variants are PREDEFINED below (feature groups, cumulative order, PCA k, filter threshold). Nothing is
  chosen from outer-fold results and no "best" variant is selected or promoted.
* Identical outer folds for every variant: LOSO (leave-one-segment-out, 15 folds) and LORO (leave-one-run-out,
  5 folds), the same `LeaveOneGroupOut` splitter over the same rows.
* Everything data-driven (imputer, scaler, correlation filter, PCA) is fitted inside each training fold only.
* The original LOSO results are reproduced exactly (reference variant) before the new numbers are trusted.
* 15 segments / 5 runs: all intervals describe evaluation sampling only and are wide; segments within a run
  share setup and tool state, so they are not fully independent.
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
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.bootstrap_ci import bootstrap_regression_ci  # noqa: E402
from src.feature_groups import (  # noqa: E402
    FEATURE_GROUPS,
    GROUP_ORDER,
    PROCESS_GROUPS,
    build_pipeline,
    columns_for,
    cumulative_stages,
    max_pca_components,
    validate_groups,
)
from src.physics_baseline import PHYSICS_MODELS  # noqa: E402
from src.preprocessing import build_feature_matrix  # noqa: E402
from src.robustness import _load_frame, oof_predict  # noqa: E402
from src.utils import ensure_dirs, load_config, save_json, set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VALIDATIONS = {"LOSO": "segment_id", "LORO": "run_id"}
MODELS = ("ridge", "random_forest", "xgboost", "gaussian_process")
REFERENCE_KEY = "reference_mean|dummy_mean"
CORR_THRESHOLD = 0.95
PCA_KS = (1, 2, 3, 5)  # predefined; each is checked against max_pca_components(n_train_units)
REPRODUCTION_TOLERANCE = 1e-9
N_PERM_REPEATS = 20


# --------------------------------------------------------------------------------------------------------
# Variant definitions
# --------------------------------------------------------------------------------------------------------
def make_variants(numeric: list[str], categorical: list[str]) -> list[dict[str, Any]]:
    validate_groups(numeric, categorical)
    variants: list[dict[str, Any]] = []

    def add(name, family, groups=None, num=None, cat=None, **kw):
        if groups is not None:
            num, cat = columns_for(groups, numeric, categorical)
        variants.append({"name": name, "family": family, "numeric": num, "categorical": cat, **kw})

    add("reference_full", "reference", groups=GROUP_ORDER)  # == original feature set, original column order
    for stage_name, groups in cumulative_stages():
        add(stage_name, "cumulative", groups=groups)
    for g in GROUP_ORDER:
        add(f"only_{g}", "single_group", groups=[g])
    for g in GROUP_ORDER:
        add(f"drop_{g}", "drop_group", groups=[x for x in GROUP_ORDER if x != g])
    add(f"full_corrfilter{CORR_THRESHOLD}", "correlation_filter", groups=GROUP_ORDER, corr_threshold=CORR_THRESHOLD)
    num_proc, _ = columns_for(PROCESS_GROUPS, numeric, categorical)
    for k in PCA_KS:
        add(f"pca{k}_process_numeric", "pca", num=list(num_proc), cat=[], pca_k=k)
    add("process_numeric_no_pca", "pca", num=list(num_proc), cat=[])  # same inputs as the PCA variants
    add("full_log_target", "target_transform", groups=GROUP_ORDER, log_target=True)
    return variants


def factory(variant: dict[str, Any], model_name: str):
    return lambda: build_pipeline(
        model_name,
        variant["numeric"],
        variant["categorical"],
        corr_threshold=variant.get("corr_threshold"),
        pca_k=variant.get("pca_k"),
        log_target=variant.get("log_target", False),
    )


# --------------------------------------------------------------------------------------------------------
# Cross-validation with jackknife stability
# --------------------------------------------------------------------------------------------------------
def cv_collect(make_estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Leave-one-group-out OOF predictions + row-wise jackknife SD.

    For each row, the SD is taken over the fold models that were TRAINED ON that row (i.e. every fold except
    its own held-out one), so it measures how much the fitted function at a training profile moves when one
    OTHER segment/run is deleted from the training set."""
    n = len(y)
    oof = np.full(n, np.nan)
    fold_preds, fold_of_row = [], np.full(n, -1)
    for k, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups)):
        est = make_estimator()
        est.fit(X.iloc[tr], y.iloc[tr])
        allp = np.asarray(est.predict(X), dtype=float)
        oof[te] = allp[te]
        fold_of_row[te] = k
        fold_preds.append(allp)
    J = np.vstack(fold_preds)  # (folds, n)
    sd = np.empty(n)
    for i in range(n):
        keep = np.arange(J.shape[0]) != fold_of_row[i]
        sd[i] = J[keep, i].std(ddof=1)
    if np.isnan(oof).any():
        raise RuntimeError("Missing out-of-fold predictions.")
    return oof, sd


def fold_signature(groups: pd.Series) -> list[tuple[int, ...]]:
    """Held-out row indices of every outer fold (used to prove folds are identical across variants)."""
    return [tuple(te.tolist()) for _, te in LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)]


# --------------------------------------------------------------------------------------------------------
# A. Correlation / redundancy analysis
# --------------------------------------------------------------------------------------------------------
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b).to_numpy()
    if min(tab.shape) < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def correlation_analysis(df: pd.DataFrame, numeric: list[str], categorical: list[str], target: str) -> dict[str, Any]:
    seg = df.groupby("segment_id").agg({**{c: "first" for c in numeric + categorical}, target: "mean"})
    varying = [c for c in numeric if seg[c].nunique() > 1]
    constants = [c for c in numeric if seg[c].nunique() <= 1]
    cols = varying + [target]
    spear = seg[cols].corr(method="spearman")
    pear = seg[cols].corr(method="pearson")
    rows = []
    for i, a in enumerate(varying):
        for b in varying[i + 1 :]:
            rows.append({"feature_a": a, "feature_b": b, "spearman": spear.loc[a, b], "pearson": pear.loc[a, b]})
    pairs = pd.DataFrame(rows)
    pairs["abs_spearman"] = pairs.spearman.abs()
    pairs = pairs.sort_values("abs_spearman", ascending=False)
    X = StandardScaler().fit_transform(seg[varying].to_numpy(dtype=float))
    sv = np.linalg.svd(X, compute_uv=False)
    cat_cols = categorical + ["machining_order"]
    v = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for a in cat_cols:
        for b in cat_cols:
            v.loc[a, b] = cramers_v(seg[a].astype(str), seg[b].astype(str))
    return {
        "segment_table": seg,
        "varying": varying,
        "constants": constants,
        "spearman": spear,
        "pairs": pairs,
        "numeric_rank": int(np.linalg.matrix_rank(X, tol=1e-8)),
        "n_varying_numeric": len(varying),
        "singular_values": sv.tolist(),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 1e-10 else None,  # None = numerically singular
        "cramers_v": v,
        "n_segments": int(len(seg)),
    }


def plot_correlation(corr: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    s = corr["spearman"]
    ax = axes[0]
    im = ax.imshow(s.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(s)))
    ax.set_yticks(range(len(s)))
    ax.set_xticklabels(s.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(s.index, fontsize=8)
    for i in range(len(s)):
        for j in range(len(s)):
            ax.text(j, i, f"{s.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(s.iloc[i, j]) > 0.6 else "black")
    ax.set_title(f"Spearman correlation, segment level (n={corr['n_segments']} segments)\n"
                 f"{len(corr['constants'])} zero-variance constants omitted: {', '.join(corr['constants'])}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    v = corr["cramers_v"]
    ax = axes[1]
    im = ax.imshow(v.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(v)))
    ax.set_yticks(range(len(v)))
    ax.set_xticklabels(v.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(v.index, fontsize=8)
    for i in range(len(v)):
        for j in range(len(v)):
            val = v.iloc[i, j]
            ax.text(j, i, "n/a" if np.isnan(val) else f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if (np.isnan(val) or val < 0.6) else "black")
    ax.set_title("Cramer's V among design descriptors\n(1.00 = one determines the other)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Project 1 - predictor correlation and redundancy (descriptive; 15 segments, so |rho| estimates are very noisy)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------
# PCA descriptive: explained variance
# --------------------------------------------------------------------------------------------------------
def pca_explained_variance(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """Per-LOSO-fold PCA spectrum on training rows only (+ all-data spectrum for description)."""
    X = df[numeric_cols].to_numpy(dtype=float)
    seg = df["segment_id"]
    ratios = []
    for tr, _ in LeaveOneGroupOut().split(X, groups=seg):
        Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[tr]))
        p = PCA(svd_solver="full").fit(Z)
        r = np.zeros(len(numeric_cols))
        r[: len(p.explained_variance_ratio_)] = p.explained_variance_ratio_
        ratios.append(r)
    Z_all = StandardScaler().fit_transform(X)
    allr = PCA(svd_solver="full").fit(Z_all).explained_variance_ratio_
    fr = np.vstack(ratios)
    eff = (fr > 1e-8).sum(axis=1)  # components carrying real variance in each training fold
    return {"fold_ratios": fr, "all_data_ratios": allr, "columns": numeric_cols,
            "effective_rank_min": int(eff.min()), "effective_rank_max": int(eff.max())}


def plot_pca_variance(pv: dict[str, Any], n_independent_segments: int, path: Path) -> None:
    fr = pv["fold_ratios"]
    k = np.arange(1, fr.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(k, fr.mean(axis=0), yerr=[fr.mean(0) - fr.min(0), fr.max(0) - fr.mean(0)], capsize=3, color="#4c72b0")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("Per-fold PCA (training rows only): mean, min-max over 15 folds", fontsize=9)
    cum = np.cumsum(fr, axis=1)
    axes[1].plot(k, cum.mean(0), "o-", color="#4c72b0")
    axes[1].fill_between(k, cum.min(0), cum.max(0), alpha=0.25, color="#4c72b0")
    for kk in PCA_KS:
        axes[1].axvline(kk, color="grey", ls=":", lw=0.8)
    axes[1].axhline(0.95, color="black", ls="--", lw=0.8)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title(f"Predefined k = {list(PCA_KS)} (dotted); cap = n_train_segments // 2 = {max_pca_components(n_independent_segments - 1)}", fontsize=9)
    fig.suptitle(f"PCA of standardised numeric process inputs ({len(pv['columns'])} varying columns, {n_independent_segments} segments): "
                 f"only {pv['effective_rank_min']}-{pv['effective_rank_max']} components carry variance in any training fold", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------
# D. Group permutation importance and grouped SHAP (out-of-fold)
# --------------------------------------------------------------------------------------------------------
def group_columns_map(numeric: list[str], categorical: list[str]) -> dict[str, list[str]]:
    return {g: [c for c in numeric + categorical if c in FEATURE_GROUPS[g]["numeric"] + FEATURE_GROUPS[g]["categorical"]] for g in GROUP_ORDER}


def oof_group_permutation(model_name, numeric, categorical, X, y, split_groups, n_repeats, seed):
    """Out-of-fold group permutation importance.

    The model of each fold is fitted on training rows; a whole group of columns of the held-out rows is
    replaced jointly by the same donor row (donors come from OTHER held-out groups, so the group's internal
    structure is kept but its link to the target and other groups is cut). Importance = MAE increase.
    CAVEAT: permuting a group creates feature combinations that never occur (e.g. feed_mean inconsistent with
    feed_per_tooth) - extrapolation can inflate the importance of strongly interlinked groups."""
    rng = np.random.default_rng(seed)
    gmap = group_columns_map(numeric, categorical)
    G = list(gmap)
    n = len(y)
    sg = np.asarray(split_groups)
    base = np.full(n, np.nan)
    perm = np.full((len(G), n_repeats, n), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, split_groups):
        est = build_pipeline(model_name, numeric, categorical).fit(X.iloc[tr], y.iloc[tr])
        base[te] = est.predict(X.iloc[te])
        donors_pool = np.where(sg != sg[te][0])[0]
        batches = []
        for g in G:
            for _ in range(n_repeats):
                Xb = X.iloc[te].copy()
                donors = rng.choice(donors_pool, size=len(te), replace=True)
                for c in gmap[g]:
                    Xb[c] = X.iloc[donors][c].to_numpy()
                batches.append(Xb)
        p = np.asarray(est.predict(pd.concat(batches, ignore_index=True))).reshape(len(G), n_repeats, len(te))
        perm[:, :, te] = p
    yv = y.to_numpy()
    base_err = np.abs(yv - base)
    rows = []
    segs = np.asarray(sg)
    for gi, g in enumerate(G):
        d_mae = np.array([np.abs(yv - perm[gi, r]).mean() - base_err.mean() for r in range(n_repeats)])
        d_rmse = np.array([np.sqrt(((yv - perm[gi, r]) ** 2).mean()) - np.sqrt(((yv - base) ** 2).mean()) for r in range(n_repeats)])
        mean_perm_err = np.mean([np.abs(yv - perm[gi, r]) for r in range(n_repeats)], axis=0)
        per_group_increase = pd.Series(mean_perm_err - base_err).groupby(segs).mean()
        rows.append(
            {
                "group": g,
                "n_columns": len(gmap[g]),
                "dMAE_mean": d_mae.mean(),
                "dMAE_sd_over_repeats": d_mae.std(ddof=1),
                "dRMSE_mean": d_rmse.mean(),
                "share_heldout_units_worse": float((per_group_increase > 0).mean()),
                "n_heldout_units": int(per_group_increase.size),
            }
        )
    return pd.DataFrame(rows), float(base_err.mean())


def _feature_to_group(transformed_name: str, categorical: list[str], numeric: list[str]) -> str:
    kind, body = transformed_name.split("__", 1)
    if kind == "num":
        col = body
    else:
        col = next(c for c in sorted(categorical, key=len, reverse=True) if body.startswith(c + "_"))
    return next(g for g in GROUP_ORDER if col in FEATURE_GROUPS[g]["numeric"] + FEATURE_GROUPS[g]["categorical"])


def oof_grouped_shap(model_name, numeric, categorical, X, y, split_groups):
    """Out-of-fold SHAP. Grouped value of a row = SIGNED sum of SHAP values over the group's columns
    (SHAP is additive, so group values + base value reproduce the model output exactly; summing absolute
    values would double count cancelling effects). Group importance = mean |grouped value|.

    CAVEATS: SHAP explains the fitted model, not the physical process. Correlated / deterministic columns
    (feed_mean vs feed_per_tooth, geometry vs machining_order) split credit between them arbitrarily, which
    is exactly why groups, not single columns, are reported."""
    import shap

    n = len(y)
    G = GROUP_ORDER
    out = np.full((n, len(G)), np.nan)
    additivity_gap = 0.0
    for tr, te in LeaveOneGroupOut().split(X, y, split_groups):
        est = build_pipeline(model_name, numeric, categorical).fit(X.iloc[tr], y.iloc[tr])
        pre, mdl = est.named_steps["preprocess"], est.named_steps["model"]
        Xt = pre.transform(X.iloc[te])
        names = pre.get_feature_names_out()
        expl = shap.TreeExplainer(mdl)
        sv = np.asarray(expl.shap_values(Xt))
        base_val = float(np.ravel(expl.expected_value)[0])
        gidx = np.array([G.index(_feature_to_group(nm, categorical, numeric)) for nm in names])
        for gi in range(len(G)):
            out[te, gi] = sv[:, gidx == gi].sum(axis=1)
        recon = base_val + sv.sum(axis=1)
        additivity_gap = max(additivity_gap, float(np.max(np.abs(recon - mdl.predict(Xt)))))
    summary = pd.DataFrame({"group": G, "mean_abs_grouped_shap": np.abs(out).mean(axis=0), "mean_signed_grouped_shap": out.mean(axis=0)})
    return summary, out, additivity_gap


def plot_grouped_importance(perm: pd.DataFrame, shap_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    colors = dict(zip(GROUP_ORDER, ["#4c72b0", "#dd8452", "#55a868", "#8c8c8c", "#c44e52"]))
    for col, validation in enumerate(["LOSO", "LORO"]):
        ax = axes[0, col]
        sub = perm[perm.validation == validation]
        models = list(sub.model.unique())
        w = 0.8 / len(models)
        for mi, m in enumerate(models):
            s = sub[sub.model == m].set_index("group").loc[GROUP_ORDER]
            ax.bar(np.arange(len(GROUP_ORDER)) + mi * w, s.dMAE_mean, w, yerr=s.dMAE_sd_over_repeats, capsize=2, label=m)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(GROUP_ORDER)) + w * (len(models) - 1) / 2)
        ax.set_xticklabels([g.replace("_", "\n") for g in GROUP_ORDER], fontsize=8)
        ax.set_ylabel("MAE increase when group is permuted (um)")
        ax.set_title(f"Out-of-fold group permutation importance - {validation}\n(error bars = permutation repeats only, NOT sampling uncertainty)", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1, col]
        sub = shap_df[shap_df.validation == validation]
        models = list(sub.model.unique())
        w = 0.8 / len(models)
        for mi, m in enumerate(models):
            s = sub[sub.model == m].set_index("group").loc[GROUP_ORDER]
            ax.bar(np.arange(len(GROUP_ORDER)) + mi * w, s.mean_abs_grouped_shap, w, label=m)
        ax.set_xticks(np.arange(len(GROUP_ORDER)) + w * (len(models) - 1) / 2)
        ax.set_xticklabels([g.replace("_", "\n") for g in GROUP_ORDER], fontsize=8)
        ax.set_ylabel("mean |grouped SHAP| (um)")
        ax.set_title(f"Grouped SHAP (signed group sums, out-of-fold) - {validation}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Project 1 - what the models rely on, by engineering group (model reliance, not causal effect; "
                 "feed/derived/design columns are deterministically linked)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------
# E. Heteroskedasticity diagnostics
# --------------------------------------------------------------------------------------------------------
def breusch_pagan(resid: np.ndarray, regressors: np.ndarray) -> dict[str, float]:
    """Koenker (studentised) Breusch-Pagan: LM = n * R^2 of the regression of resid^2 on the regressors
    (+ intercept); LM ~ chi2(k) under homoskedasticity. Implemented directly (statsmodels not installed)."""
    e2 = np.asarray(resid, dtype=float) ** 2
    Z = np.column_stack([np.ones(len(e2)), np.asarray(regressors, dtype=float).reshape(len(e2), -1)])
    beta, *_ = np.linalg.lstsq(Z, e2, rcond=None)
    ss_res = float(((e2 - Z @ beta) ** 2).sum())
    ss_tot = float(((e2 - e2.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    k = Z.shape[1] - 1
    lm = len(e2) * r2
    return {"LM": lm, "p": float(stats.chi2.sf(lm, k)), "R2_aux": r2, "n": int(len(e2))}


def _spearman(a, b) -> tuple[float, float]:
    if np.ptp(a) == 0 or np.ptp(b) == 0 or len(a) < 4:
        return float("nan"), float("nan")
    r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def residual_diagnostics(label: str, validation: str, y: np.ndarray, pred: np.ndarray, seg: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for scale in ("absolute", "log_ratio"):
        if scale == "log_ratio":
            if (pred <= 0).any() or (y <= 0).any():
                rows.append({"model": label, "validation": validation, "scale": scale, "note": "non-positive predictions; log residual undefined"})
                continue
            res = np.log(y) - np.log(pred)
            x = np.log(pred)
        else:
            res, x = y - pred, pred
        d = pd.DataFrame({"res": res, "x": x, "seg": seg})
        sg = d.groupby("seg").agg(mean_x=("x", "mean"), msq=("res", lambda r: float((r**2).mean())), mabs=("res", lambda r: float(np.abs(r).mean())))
        bp_p = breusch_pagan(res, x)
        # segment level: regress mean squared residual of each segment on the segment's mean fitted value
        bp_s = breusch_pagan(np.sqrt(sg.msq.to_numpy()), sg.mean_x.to_numpy())
        rho_p, p_p = _spearman(np.abs(res), x)
        rho_s, p_s = _spearman(sg.mabs.to_numpy(), sg.mean_x.to_numpy())
        rows.append(
            {
                "model": label, "validation": validation, "scale": scale, "note": "",
                "bp_profile_LM": bp_p["LM"], "bp_profile_p": bp_p["p"], "bp_segment_LM": bp_s["LM"], "bp_segment_p": bp_s["p"],
                "spearman_abs_resid_vs_fitted_profile": rho_p, "spearman_profile_p": p_p,
                "spearman_abs_resid_vs_fitted_segment": rho_s, "spearman_segment_p": p_s,
                "n_profiles": len(res), "n_segments": int(sg.shape[0]),
            }
        )
    return rows


def pure_error_table(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Model-free check: does between-profile spread of Ra WITHIN a segment grow with the segment mean Ra?"""
    g = df.groupby("segment_id")[target].agg(["mean", "std", "count"]).reset_index()
    g = g[g["std"] > 0]
    rho, p = _spearman(g["mean"].to_numpy(), g["std"].to_numpy())
    slope, intercept = np.polyfit(np.log(g["mean"]), np.log(g["std"]), 1)
    return g, {"spearman_sd_vs_mean": rho, "spearman_p": p, "loglog_slope": float(slope), "n_segments": int(len(g)),
               "note": "slope ~ 1 means spread proportional to level (multiplicative noise); 3 profiles per segment give very noisy SDs"}


def plot_residuals(oof_store, pure, pure_stats, diag: pd.DataFrame, y, df, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    runs = df["run_id"].to_numpy()
    palette = dict(zip(sorted(set(runs)), plt.cm.tab10.colors))
    for ax, (label, scale) in zip(axes.ravel()[:2], [("random_forest", "absolute"), ("random_forest", "log_ratio")]):
        p = oof_store[("LOSO", label)]
        res, x = (y - p, p) if scale == "absolute" else (np.log(y) - np.log(p), np.log(p))
        ax.scatter(x, res, c=[palette[r] for r in runs], s=28)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("out-of-fold fitted" + (" Ra (um)" if scale == "absolute" else " log Ra"))
        ax.set_ylabel("residual (um)" if scale == "absolute" else "log(y / fitted)")
        ax.set_title(f"{label}, LOSO, {scale} residual (colour = run)", fontsize=9)
    ax = axes.ravel()[2]
    ax.scatter(pure["mean"], pure["std"], color="#4c72b0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("segment mean Ra (um)")
    ax.set_ylabel("within-segment SD of Ra (um)")
    ax.set_title(f"Model-free: spread vs level; rho={pure_stats['spearman_sd_vs_mean']:.2f}, log-log slope={pure_stats['loglog_slope']:.2f}", fontsize=9)
    for ax, (label, scale) in zip(axes.ravel()[3:5], [("xgboost", "absolute"), ("physics_powerlaw_fz", "absolute")]):
        p = oof_store[("LOSO", label)]
        ax.scatter(p, y - p, c=[palette[r] for r in runs], s=28)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("out-of-fold fitted Ra (um)")
        ax.set_ylabel("residual (um)")
        ax.set_title(f"{label}, LOSO, absolute residual", fontsize=9)
    ax = axes.ravel()[5]
    d = diag[(diag.validation == "LOSO") & diag.spearman_abs_resid_vs_fitted_segment.notna()]
    models = list(d.model.unique())
    w = 0.38
    for si, scale in enumerate(("absolute", "log_ratio")):
        s = d[d.scale == scale].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + si * w, s.spearman_abs_resid_vs_fitted_segment, w, label=scale)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(models)) + w / 2)
    ax.set_xticklabels([m.replace("physics_", "phys_") for m in models], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho (|resid| vs fitted), segment level")
    ax.set_title("n=15 segments: |rho| < ~0.5 is not distinguishable from 0", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Project 1 - residual diagnostics for heteroskedasticity (small n, clustered, uncorrected p-values: indicative only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------
# Plots: stages and PCA
# --------------------------------------------------------------------------------------------------------
def plot_stage(summary: pd.DataFrame, path: Path) -> None:
    stages = [s for s, _ in cumulative_stages()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, validation in zip(axes, VALIDATIONS):
        sub = summary[summary.validation == validation]
        mean_mae = sub[sub.variant == "reference_mean"].MAE.iloc[0]
        for mi, m in enumerate(MODELS):
            s = sub[(sub.model == m) & sub.variant.isin(stages)].set_index("variant").loc[stages]
            x = np.arange(len(stages)) + (mi - 1.5) * 0.09
            ax.errorbar(x, s.MAE, yerr=[s.MAE - s.MAE_lo, s.MAE_hi - s.MAE], marker="o", ms=4, capsize=2, lw=1, label=m)
        ax.axhline(mean_mae, color="black", ls="--", lw=1, label="mean predictor")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"stage {i + 1}\n+ " + s.split("_", 1)[1].replace("_", "\n") for i, s in enumerate(stages)], fontsize=7)
        ax.set_ylabel("MAE (um), 95% cluster-bootstrap CI")
        ax.set_title(f"{validation}: cumulative inclusion (feed -> spindle -> derived -> constants -> design)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Project 1 - performance vs feature-inclusion stage (predefined order; stage 5 = original full feature set). "
                 "CIs overlap heavily: differences between stages are hypothesis-generating", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pca_performance(summary: pd.DataFrame, path: Path, eff_rank_max: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ks = list(PCA_KS)
    for ax, validation in zip(axes, VALIDATIONS):
        sub = summary[summary.validation == validation]
        mean_mae = sub[sub.variant == "reference_mean"].MAE.iloc[0]
        for mi, m in enumerate(MODELS):
            s = sub[(sub.model == m) & sub.variant.isin([f"pca{k}_process_numeric" for k in ks])].set_index("variant")
            s = s.reindex([f"pca{k}_process_numeric" for k in ks])
            if s.MAE.isna().all():
                continue
            x = np.array(ks) + (mi - 1.5) * 0.06
            eb = ax.errorbar(x, s.MAE, yerr=[s.MAE - s.MAE_lo, s.MAE_hi - s.MAE], marker="o", ms=4, capsize=2, lw=1, label=m)
            nopca = sub[(sub.model == m) & (sub.variant == "process_numeric_no_pca")]
            if len(nopca):
                ax.scatter([max(ks) + 0.8], nopca.MAE, marker="s", s=35, color=eb.lines[0].get_color())
        ax.axhline(mean_mae, color="black", ls="--", lw=1, label="mean predictor")
        ax.axvspan(eff_rank_max + 0.5, max(ks) + 0.4, color="red", alpha=0.08)
        ax.text(eff_rank_max + 0.55, ax.get_ylim()[1], "k > effective rank:\nnull (noise) components", fontsize=7, va="top", color="darkred")
        ax.set_xticks(ks + [max(ks) + 0.8])
        ax.set_xticklabels([str(k) for k in ks] + ["no PCA"])
        ax.set_xlabel("number of principal components (numeric process inputs only)")
        ax.set_ylabel("MAE (um)")
        ax.set_title(f"{validation}", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Project 1 - MAE vs number of principal components (PCA fitted inside each training fold; squares = same inputs, no PCA)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stability(summary: pd.DataFrame, path: Path) -> None:
    keep = ["reference_full", "full_corrfilter0.95", "pca3_process_numeric", "process_numeric_no_pca", "only_feed", "only_design_metadata"]
    keep = [k for k in keep if k in set(summary.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, validation in zip(axes, VALIDATIONS):
        sub = summary[summary.validation == validation]
        w = 0.8 / len(MODELS)
        for mi, m in enumerate(MODELS):
            s = sub[sub.model == m].set_index("variant").reindex(keep)
            ax.bar(np.arange(len(keep)) + mi * w, s.stability_sd_rel, w, label=m)
        ax.set_xticks(np.arange(len(keep)) + 0.4 - w / 2)
        ax.set_xticklabels([k.replace("_", "\n") for k in keep], fontsize=7)
        ax.set_title(validation, fontsize=10)
        ax.set_ylabel("jackknife SD of prediction / SD(Ra)")
        ax.legend(fontsize=7)
    fig.suptitle("Project 1 - prediction stability when one training segment/run is removed (lower = more stable)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------------
def run_extended(n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])
    paths = ensure_dirs(cfg)
    df, numeric, categorical = _load_frame(cfg)
    target = cfg["data"]["target_column"]
    y = pd.to_numeric(df[target])
    yv = y.to_numpy()
    clusters = df["segment_id"].to_numpy()
    variants = make_variants(numeric, categorical)
    X_all = build_feature_matrix(df, numeric + categorical, categorical)

    # --- identical-folds evidence and PCA cap check -------------------------------------------------------
    sigs = {v: fold_signature(df[col]) for v, col in VALIDATIONS.items()}
    n_train_units = {v: int(df[col].nunique()) - 1 for v, col in VALIDATIONS.items()}
    pca_cap = {v: max_pca_components(n) for v, n in n_train_units.items()}

    oof_store: dict[tuple[str, str], np.ndarray] = {}
    preds: dict[str, dict[str, np.ndarray]] = {v: {} for v in VALIDATIONS}
    stab: dict[str, dict[str, float]] = {v: {} for v in VALIDATIONS}
    skipped: list[dict[str, Any]] = []
    meta: dict[str, dict[str, Any]] = {}

    def record(validation, variant_name, model_name, pred, sd):
        key = f"{variant_name}|{model_name}"
        preds[validation][key] = pred
        stab[validation][key] = float(sd.mean() / yv.std(ddof=0))
        if variant_name == "reference_full":
            oof_store[(validation, model_name)] = pred

    for validation, col in VALIDATIONS.items():
        groups = df[col]
        # fold identity check on the actual splitter output used below
        assert fold_signature(groups) == sigs[validation]
        # mean-predictor reference
        ref_var = {"numeric": numeric, "categorical": categorical}
        pred, sd = cv_collect(factory(ref_var, "dummy_mean"), X_all, y, groups)
        preds[validation][REFERENCE_KEY] = pred
        stab[validation][REFERENCE_KEY] = float(sd.mean() / yv.std(ddof=0))
        for variant in variants:
            if variant.get("pca_k") is not None and variant["pca_k"] > pca_cap[validation]:
                skipped.append({"validation": validation, "variant": variant["name"], "reason": f"k={variant['pca_k']} > cap {pca_cap[validation]} (= n_train_units//2 with n_train_units={n_train_units[validation]})"})
                continue
            X = X_all[variant["numeric"] + variant["categorical"]]
            meta[variant["name"]] = {"family": variant["family"], "numeric": variant["numeric"], "categorical": variant["categorical"], "pca_k": variant.get("pca_k")}
            for model_name in MODELS:
                print(f"  {validation} | {variant['name']} | {model_name}", flush=True)
                pred, sd = cv_collect(factory(variant, model_name), X, y, groups)
                record(validation, variant["name"], model_name, pred, sd)
        # physics reference models (descriptive comparators)
        X_phys = df[["feed_per_tooth_mm", "tool_diameter_mm"]]
        for pname, cls in PHYSICS_MODELS.items():
            p = oof_predict(lambda c=cls: c(), X_phys, y, groups)
            preds[validation][f"physics|{pname}"] = p
            oof_store[(validation, pname)] = p

    # --- reproduction of stored original results ---------------------------------------------------------
    stored = pd.read_csv(paths["metrics"] / "cv_model_metrics.csv").set_index("model")
    diffs = []
    from src.evaluate import regression_metrics

    for m in MODELS:
        met = regression_metrics(yv, preds["LOSO"][f"reference_full|{m}"])
        diffs += [abs(met[k] - float(stored.loc[m, k])) for k in ("MAE", "RMSE", "R2")]
    reproduction = {"max_abs_diff_vs_cv_model_metrics": float(max(diffs)), "reproduces_original": bool(max(diffs) <= REPRODUCTION_TOLERANCE)}
    # stage 5 must equal the original full model (same columns, same order)
    stage5_equal = all(np.array_equal(preds[v][f"stage5_design_metadata|{m}"], preds[v][f"reference_full|{m}"]) for v in VALIDATIONS for m in MODELS)

    num_proc, _ = columns_for(PROCESS_GROUPS, numeric, categorical)
    pv = pca_explained_variance(df, [c for c in num_proc if df[c].nunique() > 1])

    # --- bootstrap ---------------------------------------------------------------------------------------
    rows = []
    for validation, store in preds.items():
        ci = bootstrap_regression_ci(yv, store, clusters, reference=REFERENCE_KEY, n_boot=n_boot, seed=seed, alpha=alpha)
        for key, entry in ci.items():
            variant_name, model_name = key.split("|")
            fam = "mean_predictor" if variant_name == "reference_mean" else ("physics" if variant_name == "physics" else meta[variant_name]["family"])
            rows.append({"validation": validation, "family": fam, "variant": variant_name, "model": model_name,
                         "n_numeric_cols": len(meta[variant_name]["numeric"]) if variant_name in meta else np.nan,
                         "n_categorical_cols": len(meta[variant_name]["categorical"]) if variant_name in meta else np.nan,
                         "pca_k": meta[variant_name]["pca_k"] if variant_name in meta else None,
                         "pca_k_exceeds_effective_rank": bool(variant_name in meta and meta[variant_name]["pca_k"] is not None
                                                              and meta[variant_name]["pca_k"] > pv["effective_rank_max"]),
                         "stability_sd_rel": stab[validation].get(key, np.nan), **entry})
    summary = pd.DataFrame(rows)
    summary.to_csv(paths["tables"] / "extended_comparison_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(paths["tables"] / "extended_skipped_variants.csv", index=False)

    # increments between consecutive cumulative stages (paired, same resamples)
    inc_rows = []
    stage_names = [s for s, _ in cumulative_stages()]
    for validation in VALIDATIONS:
        for m in MODELS:
            prev_key = REFERENCE_KEY
            prev_name = "reference_mean"
            for s in stage_names:
                cur = f"{s}|{m}"
                ci = bootstrap_regression_ci(yv, {prev_key: preds[validation][prev_key], cur: preds[validation][cur]}, clusters,
                                             reference=prev_key, n_boot=n_boot, seed=seed, alpha=alpha)[cur]
                inc_rows.append({"validation": validation, "model": m, "stage": s, "compared_with": prev_name, "MAE": ci["MAE"],
                                 "dMAE_vs_previous": ci["dMAE_vs_ref"], "dMAE_lo": ci["dMAE_vs_ref_lo"], "dMAE_hi": ci["dMAE_vs_ref_hi"],
                                 "share_better_than_previous": ci["share_better_than_ref"]})
                prev_key, prev_name = cur, s
    increments = pd.DataFrame(inc_rows)
    increments.to_csv(paths["tables"] / "extended_stage_increments.csv", index=False)

    # LORO has only 5 independent runs and segments inside a run share setup/tool state, so the segment-clustered
    # intervals above could be too narrow. Companion tables resample whole RUNS instead. Observed outcome: these are
    # NOT systematically wider (errors of segments inside a run partly cancel), and a percentile bootstrap over 5
    # clusters is itself unreliable, so neither scheme resolves LORO differences; both are reported.
    run_clusters = df["run_id"].to_numpy()
    ci_run = bootstrap_regression_ci(yv, preds["LORO"], run_clusters, reference=REFERENCE_KEY, n_boot=n_boot, seed=seed, alpha=alpha)
    run_rows = [{"validation": "LORO", "variant": k.split("|")[0], "model": k.split("|")[1], **e} for k, e in ci_run.items()]
    pd.DataFrame(run_rows).to_csv(paths["tables"] / "extended_comparison_summary_loro_runcluster.csv", index=False)
    inc_run = []
    for m in MODELS:
        prev_key, prev_name = REFERENCE_KEY, "reference_mean"
        for s in stage_names:
            cur = f"{s}|{m}"
            ci = bootstrap_regression_ci(yv, {prev_key: preds["LORO"][prev_key], cur: preds["LORO"][cur]}, run_clusters,
                                         reference=prev_key, n_boot=n_boot, seed=seed, alpha=alpha)[cur]
            inc_run.append({"validation": "LORO", "model": m, "stage": s, "compared_with": prev_name, "MAE": ci["MAE"],
                            "dMAE_vs_previous": ci["dMAE_vs_ref"], "dMAE_lo": ci["dMAE_vs_ref_lo"], "dMAE_hi": ci["dMAE_vs_ref_hi"],
                            "share_better_than_previous": ci["share_better_than_ref"]})
            prev_key, prev_name = cur, s
    pd.DataFrame(inc_run).to_csv(paths["tables"] / "extended_stage_increments_loro_runcluster.csv", index=False)

    # --- A. correlation -----------------------------------------------------------------------------------
    corr = correlation_analysis(df, numeric, categorical, target)
    corr["pairs"].to_csv(paths["tables"] / "predictor_correlation_pairs.csv", index=False)
    corr["spearman"].to_csv(paths["tables"] / "predictor_spearman_matrix.csv")
    corr["cramers_v"].to_csv(paths["tables"] / "design_cramers_v.csv")
    plot_correlation(corr, paths["figures"] / "14_correlation_heatmap.png")

    # --- PCA descriptives ---------------------------------------------------------------------------------
    plot_pca_variance(pv, int(df["segment_id"].nunique()), paths["figures"] / "15_pca_explained_variance.png")
    plot_pca_performance(summary, paths["figures"] / "16_performance_vs_pca_components.png", pv["effective_rank_max"])
    plot_stage(summary, paths["figures"] / "17_performance_vs_feature_stage.png")
    plot_stability(summary, paths["figures"] / "20_prediction_stability.png")

    # --- D. group permutation importance and grouped SHAP -------------------------------------------------
    perm_tables, shap_tables, shap_gap = [], [], 0.0
    for validation, col in VALIDATIONS.items():
        for m in ("ridge", "random_forest", "xgboost"):
            print(f"  group permutation | {validation} | {m}", flush=True)
            t, base_mae = oof_group_permutation(m, numeric, categorical, X_all, y, df[col], N_PERM_REPEATS, seed)
            t.insert(0, "model", m)
            t.insert(0, "validation", validation)
            t["baseline_oof_MAE"] = base_mae
            perm_tables.append(t)
        for m in ("random_forest", "xgboost"):
            print(f"  grouped SHAP | {validation} | {m}", flush=True)
            s, _, gap = oof_grouped_shap(m, numeric, categorical, X_all, y, df[col])
            s.insert(0, "model", m)
            s.insert(0, "validation", validation)
            shap_tables.append(s)
            shap_gap = max(shap_gap, gap)
    perm = pd.concat(perm_tables, ignore_index=True)
    shap_df = pd.concat(shap_tables, ignore_index=True)
    perm.to_csv(paths["tables"] / "group_permutation_importance.csv", index=False)
    shap_df.to_csv(paths["tables"] / "grouped_shap.csv", index=False)
    plot_grouped_importance(perm, shap_df, paths["figures"] / "18_grouped_feature_importance.png")

    # --- E. residual diagnostics --------------------------------------------------------------------------
    diag_rows = []
    labels = list(MODELS) + list(PHYSICS_MODELS)
    for validation in VALIDATIONS:
        for label in labels:
            p = oof_store[(validation, label)]
            diag_rows += residual_diagnostics(label, validation, yv, p, clusters)
        for m in MODELS:  # log-target variant
            p = preds[validation][f"full_log_target|{m}"]
            diag_rows += residual_diagnostics(f"{m} [log-target]", validation, yv, p, clusters)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(paths["tables"] / "residual_diagnostics.csv", index=False)
    pure, pure_stats = pure_error_table(df, target)
    pure.to_csv(paths["tables"] / "within_segment_spread.csv", index=False)
    plot_residuals(oof_store, pure, pure_stats, diag.dropna(subset=["spearman_abs_resid_vs_fitted_segment"]), yv, df, paths["figures"] / "19_residual_diagnostics.png")

    results = {
        "n_boot": n_boot, "seed": seed, "corr_threshold": CORR_THRESHOLD, "pca_ks": list(PCA_KS), "pca_cap_by_validation": pca_cap,
        "n_train_units_by_validation": n_train_units, "skipped_variants": skipped,
        "reproduction_check": reproduction, "stage5_identical_to_reference_full": bool(stage5_equal),
        "grouped_shap_max_additivity_gap": shap_gap,
        "correlation": {"n_varying_numeric": corr["n_varying_numeric"], "numeric_rank": corr["numeric_rank"],
                        "condition_number": corr["condition_number"], "zero_variance": corr["constants"]},
        "pure_error": pure_stats,
        "pca_effective_rank_by_fold": {"min": pv["effective_rank_min"], "max": pv["effective_rank_max"]},
        "variants": meta,
    }
    save_json(results, paths["metrics"] / "extended_results.json")
    print("Reproduction:", reproduction, "| stage5==full:", stage5_equal, "| SHAP gap:", shap_gap, flush=True)
    return results
