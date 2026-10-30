from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import add_process_features
from src.preprocessing import LEAKAGE_COLUMNS


def test_feature_list_excludes_post_process_roughness():
    df, numeric, categorical = add_process_features(
        __import__("src.data_loader", fromlist=["build_modeling_frame"]).build_modeling_frame()
    )
    used = set(numeric + categorical)
    overlap = used & LEAKAGE_COLUMNS
    assert not overlap, f"Leakage columns in features: {overlap}"
