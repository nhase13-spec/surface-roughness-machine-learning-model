"""Process-parameter features (no post-process inspection leakage)."""
from __future__ import annotations

import math

import pandas as pd


def add_process_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Derive machining features from documented programmed settings."""
    out = df.copy()

    out["feed_min_mm_min"] = pd.to_numeric(out["programmed_feed_min_mm_min"], errors="coerce")
    out["feed_max_mm_min"] = pd.to_numeric(out["programmed_feed_max_mm_min"], errors="coerce")
    out["spindle_min_rpm"] = pd.to_numeric(out["programmed_spindle_min_rpm"], errors="coerce")
    out["spindle_max_rpm"] = pd.to_numeric(out["programmed_spindle_max_rpm"], errors="coerce")

    out["feed_mean_mm_min"] = (out["feed_min_mm_min"] + out["feed_max_mm_min"]) / 2.0
    out["spindle_mean_rpm"] = (out["spindle_min_rpm"] + out["spindle_max_rpm"]) / 2.0
    out["feed_range_mm_min"] = out["feed_max_mm_min"] - out["feed_min_mm_min"]
    out["spindle_range_rpm"] = out["spindle_max_rpm"] - out["spindle_min_rpm"]

    teeth = pd.to_numeric(out["tool_teeth"], errors="coerce")
    diameter = pd.to_numeric(out["tool_diameter_mm"], errors="coerce")

    # Feed per tooth fz [mm/tooth] = F / (n * z)
    denom = out["spindle_mean_rpm"] * teeth
    out["feed_per_tooth_mm"] = out["feed_mean_mm_min"] / denom.replace(0, pd.NA)

    # Cutting speed vc [m/min] = pi * D_mm * n_rpm / 1000
    out["cutting_speed_m_min"] = math.pi * diameter * out["spindle_mean_rpm"] / 1000.0

    out["material_removal_rate_proxy"] = (
        out["feed_mean_mm_min"]
        * pd.to_numeric(out["cutting_depth_mm"], errors="coerce")
        * pd.to_numeric(out["cutting_width_mm"], errors="coerce")
    )

    numeric_features = [
        "feed_min_mm_min",
        "feed_max_mm_min",
        "feed_mean_mm_min",
        "feed_range_mm_min",
        "spindle_min_rpm",
        "spindle_max_rpm",
        "spindle_mean_rpm",
        "spindle_range_rpm",
        "feed_per_tooth_mm",
        "cutting_speed_m_min",
        "cutting_depth_mm",
        "cutting_width_mm",
        "tool_diameter_mm",
        "tool_teeth",
        "material_removal_rate_proxy",
        "machining_order",
    ]

    categorical_features = [
        "geometry",
        "cutting_direction_axis",
        "run_id",
        "programmed_feed_pattern",
        "programmed_spindle_pattern",
    ]

    return out, numeric_features, categorical_features


def sensor_feature_placeholder() -> dict:
    """Document extension point for future in-process sensor channels."""
    return {
        "available": False,
        "planned_channels": ["spindle_load", "vibration_rms", "acoustic_emission"],
    }
