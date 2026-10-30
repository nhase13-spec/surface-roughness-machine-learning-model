"""Ensure grouped splits keep segment profiles together."""
from pathlib import Path
import sys

import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import build_modeling_frame
from src.feature_engineering import add_process_features
from src.preprocessing import build_feature_matrix


def test_leave_one_segment_out_keeps_profiles_together():
    df, numeric, categorical = add_process_features(build_modeling_frame())
    feature_cols = numeric + categorical
    X = build_feature_matrix(df, feature_cols, categorical)
    y = df["ra_gaussian_filtered_um"]
    groups = df["segment_id"]
    cv = LeaveOneGroupOut()
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        train_segments = set(groups.iloc[train_idx])
        test_segments = set(groups.iloc[test_idx])
        assert train_segments.isdisjoint(test_segments)
        assert len(test_segments) == 1


def test_no_duplicate_profile_keys():
    df = build_modeling_frame()
    assert df["profile_key"].duplicated().sum() == 0
