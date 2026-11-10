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


# Phase tags as possession evidence inside ball gaps.
#
# Each phase code names a team; for these it is the team in possession, and for its mirror code
# (same intervals, always the other team) the team without the ball, so mirrors are flipped and a
# pair counts as one source. GOALS (about 169 s, running on into the celebrations and mostly dead
# ball) and SET PIECES (handled above) are left out.
#
# Accuracy per source, label-free: if tags and decoder erred independently, their agreement on
# visible frames would be p_d * p_t + (1 - p_d)(1 - p_t), with p_d = 0.852 the decoder's team
# accuracy on visible frames of the degraded idsse train games. Solving for p_t gives the values
# below (python -m scripts.tag_reliability). They are upper bounds: Metrica derives its tags from
# the same tracking, so errors are shared and agreement is inflated -- two estimates came out above
# 1 and are capped at P_CAP.
DECODER_VISIBLE_ACCURACY = 0.852
P_CAP = 0.95
PHASES = {  # code: (flip, estimated accuracy as the possessing team)
    "PROGRESSION": (False, 0.95),          # agreement 0.847 -> 0.99, capped
    "DEFENDING IN MIDDLE THIRD": (True, 0.95),
    "BUILD UP": (False, 0.95),             # agreement 0.900 -> 1.07, capped
    "DEFENDING IN ATTACKING THIRD": (True, 0.95),
    "LONG BALL": (False, 0.80),            # agreement 0.714
    "BALL IN FINAL THIRD": (False, 0.78),  # agreement 0.695
    "ATTACKING TRANSITION": (False, 0.76),  # agreement 0.683
    "DEFENSIVE TRANSITION": (True, 0.76),
    "CREATING CHANCES": (False, 0.77),     # agreement 0.688
    "DEFENDING IN DEFENSIVE THIRD": (True, 0.77),
    "BALL IN THE BOX": (False, 0.68),      # agreement 0.626
    "PLAYERS IN THE BOX": (False, 0.60),   # agreement 0.570
}


def _logit(p):
    return np.log(p / (1 - p))


def phase_team_evidence(feats, tags, phases=PHASES, scale=1.0):
    """Per-second evidence for home (+) or away (-) from phase tags, on frames without a ball.

    Where tags overlap, the most reliable one speaks (they are nested and correlated, so summing
    would count the same information twice). Strength is calibrated against the nearest-player
    cue, whose up-to-1-per-second home-vs-away evidence is right DECODER_VISIBLE_ACCURACY of the
    time: a tag of accuracy p gives scale * logit(p) / logit(DECODER_VISIBLE_ACCURACY) per second.
    None without tags.
    """
    if tags is None or not len(tags):
        return None
    frame = feats["frame"].to_numpy()
    best_p = np.zeros(len(feats))
    team = np.zeros(len(feats))                      # +1 home in possession, -1 away
    codes = tags["code"].str.upper()
    for code, (flip, p) in phases.items():
        rows = tags[codes == code]
        for start, stop, t in zip(rows["start"], rows["stop"], rows["team"]):
            if t not in ("H", "A"):
                continue
            side = (1.0 if t == "H" else -1.0) * (-1.0 if flip else 1.0)
            a, b = np.searchsorted(frame, [start, stop])
            better = best_p[a:b] < p
            best_p[a:b][better] = p
            team[a:b][better] = side
    missing = feats["ball_x"].isna().to_numpy()
    strength = np.where(best_p > 0.5, _logit(np.clip(best_p, 0.5 + 1e-9, P_CAP))
                        / _logit(DECODER_VISIBLE_ACCURACY), 0.0)
    out = np.where(missing, scale * team * strength, 0.0)
    return out if np.any(out != 0) else None
