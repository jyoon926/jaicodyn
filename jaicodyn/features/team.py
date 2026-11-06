"""Team-level features per frame.

Player tracking has duplicate tracks (two boxes on one player) and ID switches, so nothing here
follows an individual track. Each frame, each team's visible positions are deduplicated -- tracks
within DEDUP_M of a kept track are merged into it, position averaged -- and then summarised:

    n_{t}        players visible after dedup
    near_{t}     metres from the ball to the team's nearest player
    cnt{r}_{t}   players within r metres of the ball, for r in RADII
    cx_{t}, cy_{t}  team centroid in metres

t is H or A. Ball-relative columns are NaN where the ball is unknown.
"""

import warnings

import numpy as np
import pandas as pd

from .ball import clean_ball

DEDUP_M = 1.0
RADII = (2, 5, 10)
TEAMS = ("H", "A")


def dense_positions(players, frame_index, team, pitch):
    """[n_frames, k, 2] metres for one team, NaN padded. Rows follow `frame_index`."""
    length, width = pitch
    rows = players[(players["team"] == team) & players["x"].notna() & players["y"].notna()]
    row = frame_index.get_indexer(rows["frame"])
    rows, row = rows[row >= 0], row[row >= 0]
    slot = rows.groupby(row).cumcount().to_numpy()
    k = int(slot.max()) + 1 if len(slot) else 1
    out = np.full((len(frame_index), k, 2), np.nan, dtype=np.float64)
    out[row, slot, 0] = rows["x"].to_numpy(float) * length
    out[row, slot, 1] = rows["y"].to_numpy(float) * width
    return out


def deduplicate(pos, radius=DEDUP_M):
    """Merge positions within `radius` of an earlier kept one; merged position is the mean.

    pos is [n, k, 2] with NaN padding. Returns a same-shaped array, NaN where merged away.
    Greedy in slot order: each slot joins the first kept slot within radius, if any.
    """
    n, k, _ = pos.shape
    valid = ~np.isnan(pos[:, :, 0])
    owner = np.tile(np.arange(k), (n, 1))
    for i in range(1, k):
        free = valid[:, i].copy()
        for j in range(i):
            kept = valid[:, j] & (owner[:, j] == j)
            close = free & kept & (np.linalg.norm(pos[:, i] - pos[:, j], axis=1) < radius)
            owner[close, i] = j
            free &= ~close
    total = np.zeros_like(pos)
    count = np.zeros((n, k))
    rows = np.arange(n)
    for i in range(k):
        v = valid[:, i]
        total[rows[v], owner[v, i]] += pos[v, i]
        count[rows[v], owner[v, i]] += 1
    with np.errstate(invalid="ignore"):
        return np.where(count[:, :, None] > 0, total / count[:, :, None], np.nan)


def ball_relative(pos, ball, radii=RADII):
    """Nearest distance and per-radius counts. pos [n,k,2], ball [n,2]."""
    d = np.linalg.norm(pos - ball[:, None, :], axis=2)
    has_ball = ~np.isnan(ball[:, 0])
    any_player = ~np.all(np.isnan(d), axis=1)
    near = np.full(len(ball), np.nan)
    ok = has_ball & any_player
    near[ok] = np.nanmin(d[ok], axis=1)
    counts = {}
    for r in radii:
        c = np.sum(d < r, axis=1).astype(float)
        c[~has_ball] = np.nan
        counts[r] = c
    return near, counts


def team_features(match, radii=RADII, dedup_m=DEDUP_M):
    """One row per frame of `match.frames`: ball, its source, and the team features."""
    frames = match.frames.sort_values("frame", ignore_index=True)
    ball, src, _ = clean_ball(frames, match.fps, match.pitch)
    index = pd.Index(frames["frame"])
    out = {"frame": frames["frame"].to_numpy(), "period": frames["period"].to_numpy(),
           "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_src": src}
    for t in TEAMS:
        pos = deduplicate(dense_positions(match.players, index, t, match.pitch), dedup_m)
        near, counts = ball_relative(pos, ball, radii)
        out["n_" + t] = np.sum(~np.isnan(pos[:, :, 0]), axis=1).astype(np.int16)
        out["near_" + t] = near
        for r in radii:
            out["cnt{}_{}".format(r, t)] = counts[r]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames give NaN
            out["cx_" + t] = np.nanmean(pos[:, :, 0], axis=1)
            out["cy_" + t] = np.nanmean(pos[:, :, 1], axis=1)
    table = pd.DataFrame(out)
    float_cols = [c for c in table.columns if table[c].dtype == np.float64]
    table[float_cols] = table[float_cols].astype(np.float32)
    return table
