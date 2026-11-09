"""Decoder inputs from the automatic smart tagging (competition match only).

The labelled open games have no tags, so nothing here is tuned on labels: every setting comes from
a label-free check on the competition match itself, recorded next to it. Without tags (every
labelled game) each function returns None, which the decoder treats as no input.

Set pieces
    A SET PIECES tag's `Team` is the team taking it: on the competition match, every corner whose
    ball is visible is tagged with the team attacking that end (9 of 9), and every goal kick with
    the team defending it (9 of 9), using the attacking directions read from the team centroids.
    Throw-ins and kickoffs carry no team (`N/A`) and are skipped. Free kicks have a team but no
    end to check against; they are treated like the others.

    SET_PIECE_P = (18 + 1) / (18 + 2): the geometry agreement with Laplace's rule, so the evidence
    stays strong but not absolute.

    The decoder's restart falls 2-14 s after the tag's start on all 27 set pieces with a team, so
    the team covers [start - BEFORE_S, start + AFTER_S] and the decoder reads it at whichever
    restart it places there (restart_windows in __init__.py).
"""

import numpy as np

SET_PIECE_P = 19 / 20
BEFORE_S = 1.0
AFTER_S = 15.0


def set_piece_restart_p(feats, tags, fps, p=SET_PIECE_P, before_s=BEFORE_S, after_s=AFTER_S):
    """P(home restarts) per row around each set piece with a team; NaN elsewhere. None without
    tags."""
    if tags is None or not len(tags):
        return None
    frame = feats["frame"].to_numpy()
    out = np.full(len(feats), np.nan)
    pieces = tags[(tags["code"].str.upper() == "SET PIECES") & tags["team"].isin(["H", "A"])]
    for start, team in zip(pieces["start"], pieces["team"]):
        a, b = np.searchsorted(frame, [start - before_s * fps, start + after_s * fps])
        out[a:b] = p if team == "H" else 1 - p
    return out if np.isfinite(out).any() else None
