import numpy as np
import pandas as pd

from jaicodyn.features.ball import BRIDGED, clean_ball
from jaicodyn.features.team import deduplicate, team_features

PITCH = (100.0, 50.0)
FPS = 25


def frames(xs, ys=None, periods=None):
    n = len(xs)
    ys = [0.5] * n if ys is None else ys
    return pd.DataFrame({"frame": np.arange(n), "period": periods or [1] * n,
                         "ball_x": np.asarray(xs, float), "ball_y": np.asarray(ys, float),
                         "label": ["X"] * n})


def test_teleport_is_removed():
    xs = np.linspace(0.4, 0.41, 50)
    xs[20:23] = 0.9  # 3 frames, 50 m away
    xy, src, _ = clean_ball(frames(xs), FPS, PITCH)
    assert np.all(src[20:23] == BRIDGED)  # removed, then the 3-frame hole is bridged
    assert np.allclose(xy[20:23, 0], np.linspace(40, 41, 50)[20:23], atol=0.05)


def test_short_hole_bridged_long_hole_kept():
    xs = np.linspace(0.4, 0.42, 200)
    xs[10:30] = np.nan      # 0.8 s
    xs[100:140] = np.nan    # 1.6 s
    xy, src, _ = clean_ball(frames(xs), FPS, PITCH)
    assert np.all(src[10:30] == BRIDGED)
    assert np.all(np.isnan(xy[100:140, 0]))


def test_no_bridge_across_periods():
    xs = np.full(20, 0.5)
    xs[9:11] = np.nan
    xy, _, _ = clean_ball(frames(xs, periods=[1] * 10 + [2] * 10), FPS, PITCH)
    assert np.all(np.isnan(xy[9:11, 0]))


def test_dedup_merges_close_tracks_only():
    pos = np.array([[[0, 0], [0.6, 0], [5, 5], [np.nan, np.nan]]], float)
    out = deduplicate(pos)
    kept = out[0][~np.isnan(out[0, :, 0])]
    assert len(kept) == 2
    assert np.allclose(sorted(kept[:, 0]), [0.3, 5])


class FakeMatch:
    fps, pitch = FPS, PITCH

    def __init__(self, frames, players):
        self.frames, self.players = frames, players


def test_team_features():
    f = frames([0.5, 0.5])
    p = pd.DataFrame({"frame": [0, 0, 0, 0, 1], "team": ["H", "H", "A", "A", "H"],
                      "track_id": ["h1", "h2", "a1", "a2", "h1"],
                      "x": [0.51, 0.512, 0.6, 0.9, 0.5], "y": [0.5] * 5})
    t = team_features(FakeMatch(f, p))
    row = t.iloc[0]
    assert row["n_H"] == 1 and row["n_A"] == 2         # h1/h2 are 0.2 m apart
    assert abs(row["near_H"] - 1.1) < 1e-3
    assert abs(row["near_A"] - 10) < 1e-3
    assert row["cnt2_H"] == 1 and row["cnt10_A"] == 0 and row["cnt5_A"] == 0
    assert abs(row["cx_A"] - 75) < 1e-3
    assert t.iloc[1]["n_A"] == 0 and np.isnan(t.iloc[1]["near_A"]) and t.iloc[1]["cnt2_A"] == 0
