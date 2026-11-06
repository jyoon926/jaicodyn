"""Clean ball track: remove teleports, then bridge short holes.

Works on any match: thresholds are in seconds and metres, not frames, and each period is cleaned
on its own so nothing links across halftime. All thresholds are physical limits on a ball, not
fitted to labels.

1. De-spike. Link consecutive visible frames at most MAX_LINK_S apart. A link faster than V_MAX
   is a violation. Cut the track at violations and longer gaps; a short segment (<= SPIKE_MAX_S)
   touching a violation is a false detection and is set to NaN. Repeat until stable. A violation
   between two long segments is left alone: deleting the wrong side would be worse.
2. Bridge holes of at most INTERP_MAX_S linearly, when the bridge itself is under V_MAX.
"""

import numpy as np

V_MAX = 40.0          # m/s; above any real ball speed
MAX_LINK_S = 0.5      # visible frames further apart are not compared
SPIKE_MAX_S = 0.2     # segments this short that touch a violation are false detections
INTERP_MAX_S = 1.0    # longer holes stay missing

MISSING, OBSERVED, BRIDGED, SPIKE = 0, 1, 2, 3


def true_runs(mask):
    """(starts, lengths) of runs of True."""
    m = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    d = np.diff(m)
    starts = np.where(d == 1)[0]
    return starts, np.where(d == -1)[0] - starts


def period_blocks(period):
    """(start, stop) row ranges, stop exclusive, one per run of equal period."""
    period = np.asarray(period)
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = np.concatenate(([0], cuts, [len(period)]))
    return list(zip(edges[:-1], edges[1:]))


def despike(xy, frame, fps, blocks):
    """(cleaned copy, removed mask, violations left between long segments)."""
    xy = xy.copy()
    removed = np.zeros(len(xy), dtype=bool)
    max_link = MAX_LINK_S * fps
    spike_max = max(1, int(round(SPIKE_MAX_S * fps)))
    remaining = 0
    for a, b in blocks:
        while True:
            idx = np.flatnonzero(~np.isnan(xy[a:b, 0])) + a
            if len(idx) < 2:
                break
            dt = np.diff(frame[idx])
            speed = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1) * fps / np.maximum(dt, 1)
            long_gap = dt > max_link
            violation = (speed > V_MAX) & ~long_gap
            seg = np.concatenate(([0], np.cumsum(violation | long_gap)))
            touches = np.zeros(seg[-1] + 1, dtype=bool)
            touches[seg[1:][violation]] = True
            touches[seg[:-1][violation]] = True
            bad = touches & (np.bincount(seg) <= spike_max)
            if not bad.any():
                remaining += int(violation.sum())
                break
            drop = idx[bad[seg]]
            xy[drop] = np.nan
            removed[drop] = True
    return xy, removed, remaining


def bridge(xy, frame, fps, blocks):
    """Linearly fill plausible short holes. Returns (filled copy, bridged mask)."""
    xy = xy.copy()
    bridged = np.zeros(len(xy), dtype=bool)
    for a, b in blocks:
        starts, lens = true_runs(np.isnan(xy[a:b, 0]))
        for s, n in zip(starts + a, lens):
            if s == a or s + n >= b:
                continue  # open at a period edge
            f0, f1 = frame[s - 1], frame[s + n]
            if (f1 - f0 - 1) > INTERP_MAX_S * fps:
                continue
            p0, p1 = xy[s - 1], xy[s + n]
            if np.linalg.norm(p1 - p0) * fps / (f1 - f0) > V_MAX:
                continue
            w = ((frame[s:s + n] - f0) / float(f1 - f0))[:, None]
            xy[s:s + n] = p0 + w * (p1 - p0)
            bridged[s:s + n] = True
    return xy, bridged


def clean_ball(frames, fps, pitch):
    """Cleaned ball in metres for a frames table.

    Returns (xy [n,2] metres NaN where unknown, src [n] int8 of MISSING/OBSERVED/BRIDGED/SPIKE,
    violations left). Rows follow `frames` order, which must be sorted by frame.
    """
    length, width = pitch
    raw = np.column_stack([frames["ball_x"].to_numpy(float) * length,
                           frames["ball_y"].to_numpy(float) * width])
    frame = frames["frame"].to_numpy(np.int64)
    blocks = period_blocks(frames["period"].to_numpy())
    xy, removed, remaining = despike(raw, frame, fps, blocks)
    xy, bridged = bridge(xy, frame, fps, blocks)
    src = np.full(len(raw), MISSING, dtype=np.int8)
    src[~np.isnan(raw[:, 0])] = OBSERVED
    src[removed] = SPIKE
    src[bridged] = BRIDGED
    return xy, src, remaining
