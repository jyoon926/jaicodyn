"""Per-frame features built from a Match. See team.py for the columns."""

import pandas as pd

from ..data import paths
from .ball import clean_ball
from .team import FEATURES_VERSION, team_features

FEATURES = paths.DATA / "features"
REQUIRED = {"ball_x", "ball_y", "near_H", "near_A", "speed_med", "ghost_H", "touch_H"}  # add new columns here


def feature_path(match):
    return FEATURES / match.source / "{}.parquet".format(match.id)


def build(match, force=False):
    """Team features for a match, cached under data/features/."""
    path = feature_path(match)
    source = match.folder / "frames.parquet"
    fresh = path.exists() and (not source.exists()
                               or path.stat().st_mtime >= source.stat().st_mtime)
    if fresh and not force:  # stale if the match was re-processed after the features were built
        table = pd.read_parquet(path)
        current = ("features_version" in table and len(table)
                   and int(table["features_version"].iloc[0]) == FEATURES_VERSION)
        if REQUIRED <= set(table.columns) and current:
            return table  # otherwise built by an older version: rebuild
    table = team_features(match)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False, compression="zstd")
    return table


__all__ = ["build", "clean_ball", "feature_path", "team_features"]
