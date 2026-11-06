"""Possession spells: maximal runs of H or A within a period."""

import numpy as np
import pandas as pd


def _runs(values):
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.r_[0, breaks], np.r_[breaks, len(values)]


def spells(frames):
    """Team spells of a frames table as start, stop (exclusive), team and period."""
    rows = []
    for period, group in frames.groupby("period", sort=True):
        labels, numbers = group["label"].values, group["frame"].values
        for start, stop in zip(*_runs(labels)):
            if labels[start] in "HA":
                rows.append((numbers[start], numbers[stop - 1] + 1, labels[start], period))
    return pd.DataFrame(rows, columns=["start", "stop", "team", "period"])


def absorb_short_spells(frames, min_frames):
    """Labels with team spells under min_frames merged, if one team holds on both sides."""
    out = frames["label"].values.copy()
    for rows in frames.groupby("period").indices.values():
        window = out[rows]
        while True:
            starts, stops = _runs(window)
            kinds = window[starts]
            lengths = stops - starts
            short = [i for i in range(1, len(starts) - 1) if lengths[i] < min_frames
                     and kinds[i] in "HA" and kinds[i - 1] == kinds[i + 1] and kinds[i - 1] in "HA"]
            if not short:
                break
            pick = min(short, key=lambda i: lengths[i])
            window[starts[pick]:stops[pick]] = kinds[pick - 1]
        out[rows] = window
    return out
