"""Touches: the ball changing direction or speed at a player's position.

A ball only changes velocity sharply when something hits it. Rolling friction slows it by well
under 1 m/s over a fraction of a second, and the competition's tracking error drifts over seconds
(data/degrade.py), so neither looks like a kick. A touch is a frame where

    dv     |v_out - v_in| >= DV_MIN_MPS, with v_in the ball's mean velocity over the WINDOW_S
           before the frame and v_out over the WINDOW_S after (every frame of both windows has a
           known ball), and dv the largest within WINDOW_S either side, so one touch is one frame
    at     some player within AT_M of the ball on that frame
    clear  the other team's nearest player at least CLEAR_M further away; otherwise it is a duel
           and the touch names no team

A touch says that team had the ball for those frames: the team is marked from WINDOW_S before the
frame to WINDOW_S after. Every constant is a physical size, not fitted to labels. A ball in the
air also swerves in the image when it lands, but nobody is at it.
"""

import numpy as np

WINDOW_S = 0.2      # velocity is measured over this long either side of the touch
DV_MIN_MPS = 3.0    # a firm pass from rest, or a ball stopped or turned at a player's feet
AT_M = 1.5          # a player this close can be the one who touched it
CLEAR_M = 1.0       # ... and the other team's nearest player at least this much further away


def velocity_change(ball, fps, period, window_s=WINDOW_S):
    """[n] |v_out - v_in| in m/s, NaN unless the ball is known over both windows in one period."""
    n = len(ball)
    k = max(1, int(round(window_s * fps)))
    out = np.full(n, np.nan)
    if n <= 2 * k:
        return out
    t = np.arange(k, n - k)
    v_in = (ball[t] - ball[t - k]) * fps / k
    v_out = (ball[t + k] - ball[t]) * fps / k
    dv = np.linalg.norm(v_out - v_in, axis=1)
    known = ~np.isnan(ball[:, 0])
    run = np.convolve(known, np.ones(2 * k + 1, dtype=int), mode="valid") == 2 * k + 1
    same = period[t - k] == period[t + k]
    out[t] = np.where(run & same, dv, np.nan)
    return out


def touch_frames(ball, near_h, near_a, fps, period, window_s=WINDOW_S, dv_min=DV_MIN_MPS,
                 at_m=AT_M, clear_m=CLEAR_M):
    """(frame rows, team) of each touch: team +1 home, -1 away, 0 a duel."""
    dv = velocity_change(ball, fps, period, window_s)
    k = max(1, int(round(window_s * fps)))
    filled = np.nan_to_num(dv, nan=-1.0)
    pad = np.pad(filled, k, constant_values=-1.0)
    local_max = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1).max(axis=1)
    big = 1e3
    nh, na = np.nan_to_num(near_h, nan=big), np.nan_to_num(near_a, nan=big)
    rows = np.flatnonzero((filled >= dv_min) & (filled >= local_max) & (np.minimum(nh, na) <= at_m))
    # keep the first of equal maxima within one window
    keep = np.r_[True, np.diff(rows) > k] if len(rows) else np.array([], bool)
    rows = rows[keep]
    team = np.where(na[rows] - nh[rows] >= clear_m, 1,
                    np.where(nh[rows] - na[rows] >= clear_m, -1, 0))
    return rows, team


def touch_columns(ball, near_h, near_a, fps, period, window_s=WINDOW_S, **kw):
    """(touch_H, touch_A): int8 per frame, 1 within window_s of a touch by that team."""
    n = len(ball)
    rows, team = touch_frames(ball, near_h, near_a, fps, period, window_s, **kw)
    k = max(1, int(round(window_s * fps)))
    out = {1: np.zeros(n, np.int8), -1: np.zeros(n, np.int8)}
    for r, t in zip(rows, team):
        if t:
            lo, hi = max(0, r - k), min(n, r + k + 1)
            same = period[lo:hi] == period[r]
            out[t][lo:hi][same] = 1
    return out[1], out[-1]
