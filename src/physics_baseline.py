"""Physics-motivated baselines for finish-milling Ra.

IMPORTANT - read before interpreting:
These are *idealised kinematic* scaling models. They represent only the geometric cusp left between
successive tool passes and ignore tool run-out, tool wear, vibration/chatter, built-up edge, material
side-flow and the scan direction of the stylus. The kinematic formula below is the standard textbook
cusp-height idealisation (cusp height ~ f_z^2 / (4 D) for a circular cutter path, and Ra roughly a
quarter of that peak-to-valley height); the constant factors were NOT verified against a specific
reference for this dataset and no citation is claimed. Treat the *uncalibrated* variant as a
dimensional sanity check, and the *calibrated* variants as one- or two-parameter statistical fits that
carry a physical functional form (Ra increasing with f_z^2 / D).

Inputs required (per row): ``feed_per_tooth_mm`` [mm/tooth] and ``tool_diameter_mm`` [mm].
Output: Ra in micrometres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

FZ_COL = "feed_per_tooth_mm"
D_COL = "tool_diameter_mm"

# Ra ~ Rt / 4 with Rt ~ fz^2 / (4 D)  ->  Ra ~ fz^2 / (16 D)   (idealised; see module docstring)
KINEMATIC_DIVISOR = 16.0
MM_TO_UM = 1000.0


def kinematic_ra_um(fz_mm: np.ndarray, d_mm: np.ndarray) -> np.ndarray:
    """Uncalibrated idealised cusp-height Ra estimate in micrometres."""
    fz = np.asarray(fz_mm, dtype=float)
    d = np.asarray(d_mm, dtype=float)
    return MM_TO_UM * fz**2 / (KINEMATIC_DIVISOR * d)


def _inputs(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fz = pd.to_numeric(X[FZ_COL], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(X[D_COL], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(fz).all() and np.isfinite(d).all()):
        raise ValueError("Physics baseline needs finite feed_per_tooth_mm and tool_diameter_mm.")
    return fz, d


class KinematicUncalibrated(BaseEstimator, RegressorMixin):
    """Ra = 1000 * fz^2 / (16 D). No fitting at all - nothing can leak."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        fz, d = _inputs(X)
        return kinematic_ra_um(fz, d)


class KinematicCalibrated(BaseEstimator, RegressorMixin):
    """Ra = a + b * (fz^2 / D); a and b fitted by least squares on the TRAINING fold only."""

    def fit(self, X, y):
        fz, d = _inputs(X)
        x = fz**2 / d
        if np.ptp(x) == 0:
            self.slope_, self.intercept_ = 0.0, float(np.mean(y))
        else:
            self.slope_, self.intercept_ = np.polyfit(x, np.asarray(y, dtype=float), 1)
        return self

    def predict(self, X):
        fz, d = _inputs(X)
        return self.intercept_ + self.slope_ * fz**2 / d


class PowerLawFz(BaseEstimator, RegressorMixin):
    """log Ra = a + b log fz, fitted on the training fold. The kinematic model predicts b = 2."""

    def fit(self, X, y):
        fz, _ = _inputs(X)
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0) or np.any(fz <= 0):
            raise ValueError("Power-law fit needs strictly positive Ra and fz.")
        if np.ptp(fz) == 0:
            self.exponent_, self.log_intercept_ = 0.0, float(np.mean(np.log(y)))
        else:
            self.exponent_, self.log_intercept_ = np.polyfit(np.log(fz), np.log(y), 1)
        return self

    def predict(self, X):
        fz, _ = _inputs(X)
        return np.exp(self.log_intercept_ + self.exponent_ * np.log(fz))


PHYSICS_MODELS = {
    "physics_kinematic_uncalibrated": KinematicUncalibrated,
    "physics_kinematic_calibrated": KinematicCalibrated,
    "physics_powerlaw_fz": PowerLawFz,
}
