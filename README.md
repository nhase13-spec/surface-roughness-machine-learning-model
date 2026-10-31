# Surface Roughness Machine Learning Model

GitHub: [nhase13-spec/surface-roughness-machine-learning-model](https://github.com/nhase13-spec/surface-roughness-machine-learning-model)

Predict **CNC surface roughness (Ra)** from **programmed feed and spindle settings** using a small public milling dataset.

**Data:** [Zenodo 21518387](https://zenodo.org/records/21518387) (CC BY 4.0) — this repo includes only the two summary CSVs needed for training (see `data/raw/dataset/DATASET.txt`).

## Quick start

```powershell
cd surface_roughness_machine_learning_model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

## Project layout

| Folder / file | Purpose |
|---------------|---------|
| `config.yaml` | Target column, paths, CV settings |
| `run_pipeline.py` | Run training and save results |
| `src/` | Load data, features, train models, plots |
| `tests/` | Checks for data leakage and grouped CV |
| `data/raw/dataset/` | Input CSV files |
| `results/` | Created after you run the pipeline (figures, metrics) |
| `models/` | Saved models after you run the pipeline |

## Models compared

Dummy mean, Ridge, Random Forest, XGBoost, Gaussian Process — with **leave-one-segment-out** cross-validation (15 segments, 45 profiles).

## Robustness, uncertainty and physics baseline

```powershell
python run_robustness.py   # writes new files only; original results are untouched
```

* **Leave-one-run-out** (5 runs; each fold holds out 3 segments, one per geometry) in addition to leave-one-segment-out.
* **No design proxies**: re-run without `machining_order`, `run_id`, `geometry` and `cutting_direction_axis` (the last is a deterministic function of geometry here). Note this also removes geometry, which is partly a real physical factor.
* **Physics baselines** (`src/physics_baseline.py`): idealised kinematic cusp estimate Ra ~ f_z^2 / (16 D) (uncalibrated; the constant is an idealisation, not verified against a specific reference), the same form calibrated per training fold, and a fitted power law in f_z.
* **Cluster bootstrap CIs** (resampling unit: segment) for MAE/RMSE/R2 and paired differences versus the mean predictor. Intervals are wide with 15 segments and 5 runs.
* Outputs: `results/tables/robustness_summary.csv`, `results/metrics/robustness_results.json`, `results/metrics/robustness_oof.csv`, `results/figures/13_robustness_mae_ci.png`, `reports/robustness_report.md`. The re-run reproduces the stored leave-one-segment-out metrics exactly.

Headline: under leave-one-segment-out no model is distinguishable from the mean predictor; under leave-one-run-out only XGBoost without design proxies clearly beats it (one of many post-hoc comparisons, not replicated under the other scheme). The uncalibrated kinematic estimate under-predicts Ra by a median factor of about 22, and the fitted exponent of Ra on f_z (-0.20, 95% CI -1.40 to 1.03) is incompatible with the idealised value of 2.

## Controlled extensions (added after reviewer feedback)

```powershell
python run_extended.py   # writes new files only (extended_*, figures 14-20); ~3 min
```

* Correlation/redundancy analysis, engineering feature groups (feed, spindle, derived physical, process constants, design metadata), cumulative/single-group/drop-group inclusion, an in-fold correlation filter, in-fold PCA (k = 1, 2, 3, 5, capped at n_train_units // 2 and flagged when k exceeds the effective rank), out-of-fold group permutation importance, grouped SHAP (signed group sums), and heteroskedasticity diagnostics (Breusch-Pagan, spread-vs-level, log-target variants), all under leave-one-segment-out **and** leave-one-run-out on identical folds.
* The reference variant reproduces the stored LOSO results exactly. Everything here is exploratory: no variant is selected or promoted. Discussion and caveats: `reports/research_addendum.md`.
* Headline: nothing overturns the original conclusion. The 12 varying numeric columns have rank 5 over 15 segments (4 without `machining_order`); adding design metadata hurts under LORO even though models rely on it most; a feed-only signal under LORO is not replicated under LOSO; PCA and correlation filtering give no consistent gain.
* Tests: `tests/test_extended.py` (leakage prevention and reproducibility).
