"""De-spiked ball track and the per-frame nearest-team signal.

    python src/ball_signal.py          # builds data/derived/ball_signal.npz and prints a summary

Steps, all within a half so nothing links across halftime:

1. De-spike. Link consecutive visible ball frames that are at most MAX_LINK_GAP frames apart. A link
   whose implied speed exceeds V_MAX is physically impossible -- no football moves faster than
   ~35 m/s -- so it is a violation. Cut the track at violations and at longer gaps; any short
   segment (<= SPIKE_MAX_LEN frames) touching a violation is a false detection and is removed.
   Repeat until nothing changes. A violation between two long segments is left alone and counted:
   it is a real jump or a persistent false track, and deleting the wrong side would be worse.
2. Bridge short gaps (<= INTERP_MAX_GAP frames) linearly, only when the bridge is physically
   plausible. Profiling showed short gaps are <1% of missing frames, so this is housekeeping.
3. For every frame with a ball position, distance from the ball to the nearest visible player of
   each team. Team-level minima are immune to the duplicate tracks and ID switches the profiling
   found, because neither changes which team is closest.

All thresholds are physical limits on a ball, not fitted to any labels.

Outputs (arrays over all 199319 video frames):
    xy      [n,2] cleaned ball position in metres, NaN where unknown
    src     [n]   0 missing, 1 observed, 2 bridged, 3 removed as a spike
    d_home  [n]   metres to nearest home player, NaN where no ball or no home player
    d_away  [n]   same for away
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, DATA_DIR, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from profile_noise import load_atd, true_runs, GT_PUBLIC  # noqa: E402

FPS = GAMES[GAME]["fps"]
V_MAX = 40.0          # m/s; above any real ball speed
MAX_LINK_GAP = 12     # frames; visible frames further apart than this are not compared
SPIKE_MAX_LEN = 5     # frames; segments this short that touch a violation are false detections
INTERP_MAX_GAP = 12   # frames (~0.5 s); longer gaps are left missing

MISSING, OBSERVED, BRIDGED, SPIKE = 0, 1, 2, 3
CACHE = os.path.join(DATA_DIR, "derived", "ball_signal.npz")


def _links(xy, a, b):
    """Visible frame indices in [a, b] and, per consecutive pair, (dt, violation, long_gap).
    idx- indices of visible frames
    violation- boolean mask for violations
    long_gap- boolean mask for long gaps
    """
    idx = np.where(~np.isnan(xy[a:b + 1, 0]))[0] + a
    dt = np.diff(idx)
    speed = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1) * FPS / np.maximum(dt, 1)
    long_gap = dt > MAX_LINK_GAP
    # these violations are spikes
    violation = (speed > V_MAX) & ~long_gap
    return idx, violation, long_gap


def despike(xy, blocks):
    """(cleaned copy, removed mask, remaining violations count)."""
    xy = xy.copy()
    removed = np.zeros(len(xy), dtype=bool)
    remaining = 0
    for _, a, b in blocks:
        while True:
            idx, violation, long_gap = _links(xy, a, b)
            if len(idx) < 2:
                break
            cut = violation | long_gap
            seg = np.concatenate(([0], np.cumsum(cut)))          
            # segment id per visible frame
            seg_len = np.bincount(seg)
            # does segment k touch a violation on its left or right boundary?
            touches = np.zeros(len(seg_len), dtype=bool)
            touches[seg[1:][violation]] = True                    
            # right side of a violation
            touches[seg[:-1][violation]] = True                   
            # left side of a violation
            bad = touches & (seg_len <= SPIKE_MAX_LEN)
            if not bad.any():
                remaining += int(violation.sum())
                break
            drop = idx[bad[seg]]
            xy[drop] = np.nan
            removed[drop] = True
    return xy, removed, remaining


def bridge(xy, blocks):
    """Linearly fill plausible short gaps. Returns (filled copy, bridged mask)."""
    xy = xy.copy()
    bridged = np.zeros(len(xy), dtype=bool)
    for _, a, b in blocks:
        starts, lens = true_runs(np.isnan(xy[a:b + 1, 0]))
        for s, n in zip(starts + a, lens):
            if n > INTERP_MAX_GAP or s == a or s + n > b:
                continue
            p0, p1 = xy[s - 1], xy[s + n]
            if np.linalg.norm(p1 - p0) * FPS / (n + 1) > V_MAX:
                continue
            w = (np.arange(1, n + 1) / (n + 1.0))[:, None]
            xy[s:s + n] = p0 + w * (p1 - p0)
            bridged[s:s + n] = True
    return xy, bridged


def team_distances(players, team, xy, chunk=20000):
    """(d_home, d_away): metres from ball to nearest visible player of each team."""
    n = len(xy)
    out = np.full((2, n), np.nan, dtype=np.float32)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        d = np.linalg.norm(players[i:j, :, :2] - xy[i:j, None, :], axis=2)
        for t in (0, 1):
            dt = d[:, team == t]
            has = ~np.all(np.isnan(dt), axis=1)
            out[t, i:j][has] = np.nanmin(dt[has], axis=1)
    return out[0], out[1]


def build(force=False):
    """Compute (or load cached) signal. Returns a dict of arrays."""
    if os.path.isfile(CACHE) and not force:
        return dict(np.load(CACHE))
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    players, ball = load_atd(raw_path, source_frames("tactical"))
    blocks = half_video_blocks(GAME)
    # blocks is (half, start_frame, end_frame) for each half; start/end inclusive

    # set raw_xy to nan outside the halves
    #raw_xy is [x, y] in metres, NaN where unknown; shape [n_frames, 2]
    raw_xy = ball[:, :2].astype(np.float64)
    # nothing outside the halves is graded; drop it so no step reaches into halftime
    in_play = np.zeros(len(raw_xy), dtype=bool)
    for _, a, b in blocks:
        in_play[a:b + 1] = True
    raw_xy[~in_play] = np.nan

    clean, removed, remaining = despike(raw_xy, blocks)
    clean, bridged = bridge(clean, blocks)
    d_home, d_away = team_distances(players, team, clean)
    raw_home, raw_away = team_distances(players, team, raw_xy)

    src = np.full(len(raw_xy), MISSING, dtype=np.int8)
    src[~np.isnan(raw_xy[:, 0])] = OBSERVED
    src[removed] = SPIKE
    src[bridged] = BRIDGED

    out = {"xy": clean.astype(np.float32), "src": src, "d_home": d_home, "d_away": d_away,
           "raw_d_home": raw_home, "raw_d_away": raw_away,
           "remaining_violations": np.array(remaining)}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def nearest_team(d_home, d_away):
    """0 home, 1 away, -1 no answer."""
    out = np.full(len(d_home), -1, dtype=np.int8)
    ok = ~np.isnan(d_home) | ~np.isnan(d_away)
    h = np.where(np.isnan(d_home), np.inf, d_home)
    w = np.where(np.isnan(d_away), np.inf, d_away)
    out[ok] = np.where(h[ok] <= w[ok], 0, 1)
    return out


if __name__ == "__main__":
    sig = build(force="--force" in sys.argv)
    src = sig["src"]
    graded = np.zeros(len(src), dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True

    print("graded frames {}".format(graded.sum()))
    for code, name in ((OBSERVED, "observed"), (SPIKE, "removed as spike"),
                       (BRIDGED, "bridged"), (MISSING, "missing")):
        print("  {:<18} {:>7}  {:5.1%}".format(name, int(np.sum(src[graded] == code)),
                                               np.mean(src[graded] == code)))
    print("  violations left between long segments: {}".format(int(sig["remaining_violations"])))

    nt = nearest_team(sig["d_home"], sig["d_away"])
    nt_raw = nearest_team(sig["raw_d_home"], sig["raw_d_away"])
    margin = np.abs(sig["d_home"] - sig["d_away"])
    has = nt >= 0
    print("\nnearest-team signal on {:.1%} of graded frames".format(np.mean(has[graded])))
    print("  home share of signalled frames: {:.1%}".format(np.mean(nt[graded & has] == 0)))
    print("  frames whose nearest team changed vs raw ball: {}".format(
        int(np.sum(graded & (nt != nt_raw) & (nt >= 0) & (nt_raw >= 0)))))
    print("  |d_home - d_away| median {:.2f} m; < 1 m on {:.1%} of signalled frames".format(
        np.nanmedian(margin[graded & has]), np.nanmean(margin[graded & has] < 1)))
    print("  nearest-team flips per minute (signalled frames): {:.1f}".format(
        np.sum((nt[1:] != nt[:-1]) & (nt[1:] >= 0) & (nt[:-1] >= 0) & graded[1:])
        / (graded.sum() / FPS / 60)))

    # Pipeline check only -- not a tuning signal (RULES.md).
    if os.path.isfile(GT_PUBLIC):
        gt = np.loadtxt(GT_PUBLIC, delimiter=",", skiprows=1, dtype=str, usecols=1)
        alive = (gt == "H") | (gt == "A")
        truth = np.where(gt == "H", 0, 1)
        for name, sig_nt in (("raw", nt_raw), ("de-spiked", nt)):
            m = alive & (sig_nt >= 0)
            print("\n[public window, pipeline check] {:<9} raw per-frame agreement with GT "
                  "team on alive frames: {:.1%} (coverage {:.1%})".format(
                      name, np.mean(sig_nt[m] == truth[m]), m.sum() / alive.sum()))
