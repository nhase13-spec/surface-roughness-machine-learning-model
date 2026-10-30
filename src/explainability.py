"""Model explainability (SHAP) for tree-based regressors."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def shap_feature_importance(
    pipe: Pipeline,
    X: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    out_path: Path,
    max_samples: int = 45,
) -> pd.DataFrame:
    """Compute SHAP values for the tree model step after preprocessing."""
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]
    X_t = pre.transform(X)
    feature_names = pre.get_feature_names_out()

    X_sample = X_t[: min(len(X_t), max_samples)]
    explainer = shap.TreeExplainer(model)
    shvals = explainer.shap_values(X_sample)
    mean_abs = np.abs(shvals).mean(axis=0)

    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp.head(15)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_title("Mean |SHAP| — XGBoost (training fit)")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return imp
