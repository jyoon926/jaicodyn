"""Per-frame features built from a Match. See team.py for the columns."""

import pandas as pd

from ..data import paths
from .ball import clean_ball
from .team import team_features

FEATURES = paths.DATA / "features"


def feature_path(match):
    return FEATURES / match.source / "{}.parquet".format(match.id)


def build(match, force=False):
    """Team features for a match, cached under data/features/."""
    path = feature_path(match)
    if path.exists() and not force:
        return pd.read_parquet(path)
    table = team_features(match)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False, compression="zstd")
    return table


__all__ = ["build", "clean_ball", "feature_path", "team_features"]
