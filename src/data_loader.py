"""Load and validate the Zenodo CNC roughness dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import load_config, project_root


def dataset_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return project_root() / cfg["data"]["raw_extract_dir"]


def load_profile_table(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = dataset_root(cfg)
    path = root / cfg["data"]["profile_summary"]
    if not path.exists():
        raise FileNotFoundError(
            f"Expected profile summary at {path}. Download Zenodo record "
            f"{cfg['data']['zenodo_record']} into data/raw/."
        )
    df = pd.read_csv(path)
    df["dataset_source"] = cfg["data"]["doi"]
    return df


def load_segment_index(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = dataset_root(cfg)
    path = root / cfg["data"]["segment_index"]
    return pd.read_csv(path)


def build_modeling_frame(cfg: dict | None = None) -> pd.DataFrame:
    """Profile-level table with process inputs and Gaussian-filtered Ra labels."""
    cfg = cfg or load_config()
    profiles = load_profile_table(cfg)
    segments = load_segment_index(cfg)

    seg_cols = [
        "segment_id",
        "run_id",
        "geometry",
        "cutting_direction_axis",
        "machining_order",
        "programmed_feed_pattern",
        "programmed_spindle_pattern",
        "cutting_depth_mm",
        "cutting_width_mm",
        "tool_diameter_mm",
        "tool_teeth",
        "workpiece_material",
    ]
    seg = segments[seg_cols].drop_duplicates("segment_id")
    df = profiles.merge(seg, on="segment_id", how="left", suffixes=("", "_seg"))

    target = cfg["data"]["target_column"]
    required = {"segment_id", "profile_id", target, "programmed_feed_min_mm_min", "programmed_spindle_min_rpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Modeling frame missing columns: {sorted(missing)}")

    df["profile_key"] = df["segment_id"].astype(str) + "_" + df["profile_id"].astype(str)
    return df


def audit_summary(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    target = cfg["data"]["target_column"]
    group = cfg["data"]["group_column"]
    return {
        "n_profiles": int(len(df)),
        "n_segments": int(df[group].nunique()),
        "n_runs": int(df["run_id"].nunique()) if "run_id" in df.columns else None,
        "profiles_per_segment": df.groupby(group).size().describe().to_dict(),
        "target_column": target,
        "target_range_um": [float(df[target].min()), float(df[target].max())],
        "missing_target": int(df[target].isna().sum()),
        "duplicate_profile_keys": int(df["profile_key"].duplicated().sum()),
        "in_process_sensor_signals": False,
    }
