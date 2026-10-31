# Research addendum (Project 1, surface roughness): what changed and why

Scope: reviewer-requested controlled experiments (correlation, engineering feature groups, gradual inclusion, PCA,
grouped importance, residual diagnostics). All baseline experiments and result files are unchanged; everything new
is written to new files (`extended_*` tables, `results/figures/14_…20_*.png`). Reproduce with `python run_extended.py`.

**Status of every number below: exploratory and hypothesis-generating.** 45 profiles = 15 segments = 5 runs; the
independent units are segments (LOSO, 15 folds) and runs (LORO, 5 folds). Variants, thresholds and PCA sizes were fixed
before running; no variant was selected from the results, and ~170 model/variant/validation cells were examined, so a
few "good-looking" cells are expected by chance.

## Validity checks run first
* Reference variant reproduces the stored LOSO results exactly (max |diff| = 2e-16); stage 5 of the cumulative
  design is bit-identical to the original full model.
* All folds are leave-one-segment-out / leave-one-run-out on identical rows; imputer, scaler, correlation filter and
  PCA are fitted on training folds only (tests in `tests/test_extended.py`, 27 tests in total pass).
* PCA sizes k = 1, 2, 3, 5 are capped at `n_train_units // 2` (7 under LOSO, 2 under LORO), so k = 3, 5 are **not run**
  under LORO (`extended_skipped_variants.csv`). **Additionally k = 5 exceeds the effective rank:** the 11 varying
  numeric process columns contain only 4 independent dimensions (3 in two folds), so PC5 is numerical noise; its
  results (and an XGBoost stability value 10x larger than any other cell) are an artefact and are flagged
  (`pca_k_exceeds_effective_rank`).

## A. Correlation and redundancy (heatmap `14_…`)
* 12 varying numeric columns span only **rank 5** over the 15 segments (`machining_order` included).
* Exact/near-exact redundancy: `spindle_max/mean/range_rpm` and `cutting_speed` (rho = 1.00); `feed_mean` and
  `material_removal_rate_proxy` (1.00); `feed_per_tooth` vs `feed_mean` (0.99). 9 of 66 pairs have |rho| >= 0.95.
* Design descriptors are deterministic: `geometry`, `cutting_direction_axis` and `machining_order` identify each other
  (Cramer's V = 1.00); `run_id` shares V = 0.88 with the feed pattern. Four tool/workpiece constants have zero variance.
* Consequence: group-level analysis is the defensible unit; single-feature importances are not identifiable here.

## B-C. Feature groups and gradual inclusion (`17_…`)
MAE in um (LOSO | LORO); mean predictor = 0.733 | 0.753; order of values: ridge / random forest / XGBoost / GP.

| Variant | LOSO | LORO |
|---|---|---|
| Full (original) | 0.83 / 0.81 / 0.85 / 0.76 | 0.82 / 0.68 / 0.78 / 0.73 |
| Stage 1 feed only | 0.75 / 0.83 / 0.75 / 0.80 | 0.70 / 0.60 / 0.55 / 0.72 |
| Stages 1-4 (all process, no design metadata) | 1.04 / 0.81 / 0.73 / 0.86 | 0.68 / 0.59 / 0.56 / 0.74 |
| Design metadata only | 0.88 / 0.77 / 0.96 / 0.77 | 0.72 / 0.74 / 0.82 / 0.79 |
| Physics, 2-parameter fit on fz^2/D | 0.65 | 0.61 |

* LOSO: no feature group made any model clearly better than the mean predictor (every paired interval includes 0).
* LORO: with feed-based inputs only XGBoost has an interval below the mean predictor (+feed: dMAE = -0.20 um,
  95% CI [-0.37, -0.03] by segment bootstrap, [-0.31, -0.09] by whole-run bootstrap); random forest has a similar
  point estimate (-0.16) but its segment-bootstrap interval includes 0, and the GP effects are tiny (-0.02 to -0.07). **This is not replicated under
  LOSO** (feed only: 0.75 vs 0.73), so it is a hypothesis, not a finding.
* Adding **design metadata made LORO worse** (RF +0.09, XGBoost +0.23 um; ridge +0.14; GP unchanged) and had
  sign-inconsistent effects under LOSO. Spindle, derived physical and constant groups added nothing measurable
  (the constants have zero variance, so they cannot).
* Under LORO the only-derived-physical set performs like the feed set (0.61 / 0.58 / 0.58), as expected for
  near-deterministic functions of feed.
* LORO intervals: segment-clustered and run-clustered bootstraps disagree in width (run-clustered not wider),
  and a percentile bootstrap over 5 runs is unreliable; neither resolves LORO differences.

## Correlation filter and PCA (`15_…`, `16_…`)
* In-fold correlation filter (|r| >= 0.95) on the full set: LOSO 0.93 / 0.81 / 1.00 / 0.75, LORO 0.81 / 0.73 / 0.90 / 0.71 -
  **no consistent improvement** over the unfiltered set (and mostly slightly worse for XGBoost).
* PCA on the numeric process inputs (design metadata excluded, since PCA is numeric-only). Versus the same inputs without
  PCA (LOSO 0.92 / 0.79 / 0.73 / 0.86; LORO 0.73 / 0.59 / 0.55 / 0.74), k = 2 changed ridge in the favourable direction
  under both schemes (0.79 and 0.63). Tree models were mixed: RF 0.70 vs 0.79 under LOSO but 0.64 vs 0.59 under LORO;
  XGBoost worse in both (0.85 vs 0.73; 0.77 vs 0.55). MAE is not monotone in k. No paired interval for this
  contrast was computed; treat the ridge change as unconfirmed.
* PCA did not make models more stable (jackknife SD of predictions is similar or larger than without PCA; `20_…`).

## D. Group importance (`18_…`)
Out-of-fold group permutation (MAE increase) and grouped SHAP (signed group sums; additivity verified to 1.4e-6).
* Grouped SHAP: design metadata is the largest group in 3 of the 4 model/scheme cells (XGBoost-LOSO: feed first);
  feed and derived physical (deterministically linked) come next; spindle and constants are ~0.
* Permutation importance is model- and scheme-dependent (e.g. RF feed under LOSO is *negative*, -0.06).
* **Reliance is not usefulness:** ridge under LORO relies heavily on design metadata (permuting it raises MAE by 0.29 um
  in all 5 held-out runs), yet *removing* it lowers MAE (0.82 -> 0.68). Permuting a group also creates impossible
  combinations (feed vs feed-per-tooth), so importances of linked groups are inflated.

## E. Heteroskedasticity (`19_…`)
* Model-free: within-segment SD of Ra vs segment mean Ra: Spearman 0.42 (p = 0.12), log-log slope 0.68 (15 segments, 3 profiles each).
* Out-of-fold residuals vs fitted: Breusch-Pagan (implemented directly; `statsmodels` unavailable). For the four
  untransformed ML models on the absolute scale the segment-level p is >= 0.08 in both schemes; profile-level tests
  are pseudo-replicated (3 profiles per segment) and shown only for completeness. Largest ML signal: XGBoost under
  LORO (profile p = 0.004, segment p = 0.08, rho = 0.51).
* Segment-level p < 0.05 occurs in 10 of 42 tests, concentrated in the physics baselines (e.g. power law), the
  log-target ridge (absolute scale, p = 0.001-0.007, rho = 0.58) and the GP on the log-ratio scale (p = 0.004). Tests
  are uncorrected for multiplicity and not independent; these point to model-specific misfit, not a general variance law.
* Log-target models reduce the rank correlation for XGBoost/RF but do not improve MAE reliably (GP 0.71 vs 0.76; ridge worse, 1.04 vs 0.83).
* Interpretation: weak, inconsistent evidence that error grows with Ra. The residual plot is dominated by **systematic
  per-segment/per-run bias** (high-Ra segments under-predicted), which a heteroskedasticity test cannot separate from
  non-constant variance. With n = 15, |rho| below ~0.5 is not distinguishable from 0.

## What this changes in the original conclusions
Nothing is overturned; the original message is reinforced: with 15 segments and 5 runs, the sensor-free process
inputs do not demonstrably beat a mean predictor under LOSO, and under LOSO a 2-parameter physics-shaped fit (0.65 um)
is lower than every ML variant tried (best 0.70), although its own interval vs the mean predictor still includes 0
(dMAE -0.08, [-0.22, +0.08]); under LORO feed-based tree models (0.55-0.60) are lower than it (0.61). New information: (i) the design-metadata group is both the most relied-upon and, under LORO, harmful;
(ii) redundancy is structural (rank 5), so dimensionality reduction targets a problem that is real but is not what
limits performance; (iii) the LORO feed signal is a hypothesis awaiting new runs.
