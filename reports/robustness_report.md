# Robustness and uncertainty report (Project 1)

Cluster bootstrap over segments (B=2000); 95% percentile intervals. LOSO = leave-one-segment-out (15 folds), LORO = leave-one-run-out (5 folds).

Reproduction of the stored LOSO/full results: {'checked': True, 'max_abs_diff': 0.0, 'reproduces_original': True}

**Caveat.** 15 segments from 5 runs. Intervals are wide and the bootstrap is approximate; it reflects evaluation sampling only. `no_design_proxies` also removes geometry, which is partly a genuine physical factor.

## LOSO

| Feature set | Model | MAE [95% CI] | RMSE [95% CI] | R2 [95% CI] | dMAE vs mean-predictor [95% CI] | Share of resamples better |
|---|---|---|---|---|---|---|
| physics | physics_kinematic_calibrated | 0.653 [0.37, 0.97] | 0.900 [0.50, 1.21] | -0.107 [-1.30, 0.30] | -0.080 [-0.22, +0.07] | 0.86 |
| full | dummy_mean | 0.733 [0.49, 1.02] | 0.915 [0.60, 1.19] | -0.144 [-0.86, -0.14] | reference | - |
| no_design_proxies | dummy_mean | 0.733 [0.49, 1.02] | 0.915 [0.60, 1.19] | -0.144 [-0.86, -0.14] | +0.000 [+0.00, +0.00] | 0.00 |
| no_design_proxies | xgboost | 0.734 [0.39, 1.13] | 1.051 [0.65, 1.38] | -0.510 [-2.47, 0.18] | +0.001 [-0.25, +0.32] | 0.52 |
| physics | physics_powerlaw_fz | 0.742 [0.37, 1.16] | 1.090 [0.58, 1.47] | -0.625 [-2.18, -0.07] | +0.009 [-0.17, +0.22] | 0.48 |
| full | gaussian_process | 0.764 [0.48, 1.09] | 0.983 [0.63, 1.29] | -0.321 [-1.36, -0.12] | +0.031 [-0.03, +0.12] | 0.24 |
| full | random_forest | 0.805 [0.50, 1.13] | 1.029 [0.67, 1.32] | -0.445 [-2.04, -0.04] | +0.072 [-0.11, +0.27] | 0.24 |
| no_design_proxies | random_forest | 0.809 [0.48, 1.17] | 1.067 [0.70, 1.37] | -0.556 [-2.49, -0.02] | +0.076 [-0.16, +0.36] | 0.29 |
| full | ridge | 0.825 [0.51, 1.20] | 1.069 [0.65, 1.45] | -0.560 [-4.00, 0.33] | +0.093 [-0.26, +0.44] | 0.30 |
| full | xgboost | 0.845 [0.45, 1.30] | 1.189 [0.72, 1.58] | -0.932 [-3.59, 0.00] | +0.113 [-0.18, +0.44] | 0.25 |
| no_design_proxies | gaussian_process | 0.859 [0.54, 1.20] | 1.090 [0.71, 1.40] | -0.622 [-2.46, -0.18] | +0.126 [-0.04, +0.36] | 0.12 |
| no_design_proxies | ridge | 1.040 [0.44, 1.91] | 1.816 [0.65, 2.90] | -3.503 [-10.86, -0.01] | +0.307 [-0.16, +1.03] | 0.18 |
| physics | physics_kinematic_uncalibrated | 1.276 [0.88, 1.73] | 1.543 [1.04, 2.01] | -2.255 [-5.01, -1.50] | +0.543 [+0.19, +0.86] | 0.00 |

## LORO

| Feature set | Model | MAE [95% CI] | RMSE [95% CI] | R2 [95% CI] | dMAE vs mean-predictor [95% CI] | Share of resamples better |
|---|---|---|---|---|---|---|
| no_design_proxies | xgboost | 0.555 [0.29, 0.87] | 0.814 [0.45, 1.12] | 0.094 [-0.34, 0.32] | -0.197 [-0.32, -0.07] | 1.00 |
| no_design_proxies | random_forest | 0.587 [0.31, 0.92] | 0.851 [0.50, 1.16] | 0.010 [-0.86, 0.37] | -0.165 [-0.32, +0.01] | 0.97 |
| physics | physics_kinematic_calibrated | 0.608 [0.36, 0.91] | 0.843 [0.44, 1.15] | 0.030 [-1.10, 0.53] | -0.145 [-0.32, +0.04] | 0.95 |
| no_design_proxies | ridge | 0.679 [0.41, 0.99] | 0.902 [0.49, 1.22] | -0.112 [-1.43, 0.42] | -0.074 [-0.26, +0.11] | 0.78 |
| full | random_forest | 0.680 [0.39, 0.98] | 0.900 [0.57, 1.18] | -0.107 [-1.19, 0.14] | -0.073 [-0.23, +0.11] | 0.80 |
| full | gaussian_process | 0.732 [0.50, 1.01] | 0.900 [0.59, 1.17] | -0.107 [-0.83, -0.05] | -0.020 [-0.04, -0.00] | 0.99 |
| no_design_proxies | gaussian_process | 0.737 [0.50, 1.02] | 0.915 [0.60, 1.20] | -0.145 [-0.81, -0.05] | -0.015 [-0.06, +0.03] | 0.72 |
| full | dummy_mean | 0.753 [0.52, 1.02] | 0.909 [0.62, 1.18] | -0.128 [-0.89, -0.07] | reference | - |
| no_design_proxies | dummy_mean | 0.753 [0.52, 1.02] | 0.909 [0.62, 1.18] | -0.128 [-0.89, -0.07] | +0.000 [+0.00, +0.00] | 0.00 |
| full | xgboost | 0.784 [0.50, 1.08] | 0.977 [0.68, 1.23] | -0.305 [-2.14, 0.13] | +0.031 [-0.20, +0.29] | 0.43 |
| full | ridge | 0.821 [0.50, 1.22] | 1.105 [0.62, 1.52] | -0.668 [-3.44, 0.30] | +0.069 [-0.21, +0.39] | 0.35 |
| physics | physics_powerlaw_fz | 0.975 [0.45, 1.62] | 1.505 [0.72, 2.18] | -2.096 [-9.02, -0.05] | +0.222 [-0.19, +0.77] | 0.20 |
| physics | physics_kinematic_uncalibrated | 1.276 [0.88, 1.73] | 1.543 [1.04, 2.01] | -2.255 [-5.01, -1.50] | +0.523 [+0.18, +0.83] | 0.00 |

## Physics baseline notes

* Uncalibrated kinematic estimate spans 0.009-0.114 um versus observed 0.18-3.40 um (median observed/predicted = 21.7).
* Power-law exponent of Ra on f_z, fitted on all 45 profiles (descriptive): -0.20 (95% segment-bootstrap CI -1.40 to 1.03); the idealised kinematic model expects 2.
* Same programmed feed, different geometry (run 01): {'straight_1_y': 1.743, 'straight_2_x': 0.201, 'straight_3_y': 2.775} um.
