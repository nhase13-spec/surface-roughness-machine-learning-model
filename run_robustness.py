"""Run leave-one-run-out, design-proxy removal, physics baselines and bootstrap CIs.

Usage:  python run_robustness.py
Writes new files only (results/tables/robustness_summary.csv, results/metrics/robustness_*.{json,csv},
results/figures/13_robustness_mae_ci.png, reports/robustness_report.md); the original results are untouched.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robustness import run_robustness  # noqa: E402

if __name__ == "__main__":
    # GPR convergence warnings are expected with 45 rows; they do not change the stored protocol.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.gaussian_process")
    run_robustness()
