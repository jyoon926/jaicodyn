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


# The challenge's "controlled possession" (challenge/CHALLENGE.md): a change of possession needs
# the other team to gain control -- "a player keeps the ball for a few seconds" or "a team
# completes more than two consecutive passes". Three passes take longer than a few seconds, so
# "a few seconds" is the lower bar of the two, read as 3. Set from the definition, not from any
# score.
CONTROLLED_S = 3.0


def fold_uncontrolled(frames, min_frames):
    """Labels relabelled to the challenge's definition of possession.

    A team spell shorter than min_frames that interrupts the other team's possession is a touch
    without control, so it takes the label of the team that last held the ball: "in phases where
    the ball is contested or loose, possession stays with the last team that had it". A spell
    right after a dead ball or the start of a period is kept whatever its length -- "whenever
    play restarts after a stoppage, there is a clear possessor".

    Unlike absorb_short_spells, the other team need not hold on both sides: a short touch before
    the ball goes out is folded too. Each period is relabelled on its own.
    """
    out = frames["label"].values.copy()
    for rows in frames.groupby("period").indices.values():
        window = out[rows]
        holder = None  # team that last had controlled possession; None after a stoppage
        for start, stop in zip(*_runs(window)):
            kind = window[start]
            if kind not in ("H", "A"):
                holder = None
            elif holder is not None and kind != holder and stop - start < min_frames:
                window[start:stop] = holder
            else:
                holder = kind
        out[rows] = window
    return out


def target_labels(match, scheme="strict"):
    """Labels to score against, sorted by frame.

    raw     as the provider labels them.
    strict  idsse relabelled to the challenge's controlled possession (fold_uncontrolled with
            CONTROLLED_S). SkillCorner's phases of play and Metrica's event-derived labels already
            require control, so they are unchanged.
    """
    frames = match.frames.sort_values("frame", ignore_index=True)
    if scheme == "raw" or not match.source.startswith("idsse"):
        return frames["label"].to_numpy()
    if scheme != "strict":
        raise ValueError("unknown label scheme {!r}".format(scheme))
    return fold_uncontrolled(frames, int(round(CONTROLLED_S * match.fps)))
