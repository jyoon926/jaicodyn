"""Synthetic smart tags on a labelled game, to test the phase-tag evidence where labels exist.

Nothing here is copied from the competition's real tags -- not their counts, lengths or coverage.
Each code fires where its definition applies, on the clean tracking and true (strict) labels of
the same game, at most once per true possession:

    BUILD UP              the possession starts in the team's own third
    PROGRESSION           the ball moves from the team's own third into the middle third
    BALL IN FINAL THIRD   the ball enters the attacking third
    BALL IN THE BOX       the ball enters the opponent's penalty box
    PLAYERS IN THE BOX    PLAYERS_IN_BOX or more attackers in the opponent's penalty box
    LONG BALL             the ball travels more than LONG_BALL_M within LONG_BALL_S
    ATTACKING TRANSITION  the possession starts with a turnover in open play (no stoppage)

Mirror codes (DEFENDING IN ATTACKING THIRD, DEFENDING IN MIDDLE THIRD, DEFENSIVE TRANSITION) get
the same interval and the other team, as real ones do. CREATING CHANCES is left out: it has no
clean definition from tracking alone.

Each interval runs either a fixed `window_s` from its trigger, or to the end of its true possession
plus `overrun_s` -- either way it can run past a turnover. Its team is set per interval:

    with probability r       copied from the decoder: its team at the interval start (or its first
                             H/A inside the interval, if it says dead there), held throughout --
                             a tag fooled by the same tracking as the decoder
    otherwise                the true possessing team at the trigger, flipped with probability q

Accuracy A, the realism setting, is measured where the evidence acts: frames inside an interval
with the ball missing and play truly live. Held through the window, a copied tag is right on a
share p_dec of those frames and an unflipped truth-based tag on p_true (below 1, since a turnover
inside the window makes it wrong for the rest). q is solved so that

    A = r * p_dec + (1 - r) * [(1 - q) * p_true + q * (1 - p_true)]

and a combination needing q outside [0, 1] is impossible (returns None).
"""

import zlib

import numpy as np
import pandas as pd

from ..data.labels import _runs
from ..features.team import dense_positions
from .gaps import home_direction, periods

BOX_DEPTH, BOX_WIDTH = 16.5, 40.3
PLAYERS_IN_BOX = 3
LONG_BALL_M, LONG_BALL_S = 30.0, 3.0
MIRRORS = {"BUILD UP": "DEFENDING IN ATTACKING THIRD", "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
           "ATTACKING TRANSITION": "DEFENSIVE TRANSITION"}
OTHER = {"H": "A", "A": "H"}


def possessions(truth, period):
    """[(team, start, stop)] maximal runs of H or A within a period."""
    out = []
    for a, b in zip(*_runs(truth)):
        if truth[a] in ("H", "A"):
            cuts = np.flatnonzero(period[a + 1:b] != period[a:b - 1]) + 1
            for s, e in zip(np.r_[0, cuts], np.r_[cuts, b - a]):
                out.append((truth[a], a + s, a + e))
    return out


def triggers(clean, clean_players, truth, fps, pitch):
    """[(code, row, true team)] where each code's definition fires, once per true possession."""
    length, width = pitch
    period = clean["period"].to_numpy()
    x, y = clean["ball_x"].to_numpy(float), clean["ball_y"].to_numpy(float)
    home_dir = home_direction(clean, pitch, periods(clean))
    index = pd.Index(clean["frame"])
    pos = {t: dense_positions(clean_players, index, t, pitch) for t in ("H", "A")}
    lag = int(LONG_BALL_S * fps)
    out = []
    for team, a, b in possessions(truth, period):
        d = home_dir[a] * (1.0 if team == "H" else -1.0)       # +1: this team attacks +x
        u = np.where(d > 0, x[a:b], length - x[a:b])            # distance along its attack
        side = np.abs(y[a:b] - width / 2)
        seen = ~np.isnan(u)

        def first(mask):
            i = np.flatnonzero(mask & seen)
            return a + int(i[0]) if len(i) else None

        hits = {}
        if seen[0] and u[0] < length / 3:
            hits["BUILD UP"] = a
        own = np.flatnonzero(seen & (u < length / 3))
        if len(own):
            later = np.flatnonzero(seen & (u >= length / 3) & (np.arange(b - a) > own[0]))
            if len(later):
                hits["PROGRESSION"] = a + int(later[0])
        hits["BALL IN FINAL THIRD"] = first(u > 2 * length / 3)
        in_box = (u > length - BOX_DEPTH) & (side < BOX_WIDTH / 2)
        hits["BALL IN THE BOX"] = first(in_box)
        p = pos[team][a:b]
        pu = np.where(d > 0, p[:, :, 0], length - p[:, :, 0])
        crowd = np.sum((pu > length - BOX_DEPTH) & (np.abs(p[:, :, 1] - width / 2) < BOX_WIDTH / 2),
                       axis=1)
        i = np.flatnonzero(crowd >= PLAYERS_IN_BOX)
        hits["PLAYERS IN THE BOX"] = a + int(i[0]) if len(i) else None
        if b - a > lag:
            jump = np.hypot(x[a + lag:b] - x[a:b - lag], y[a + lag:b] - y[a:b - lag])
            i = np.flatnonzero(jump > LONG_BALL_M)
            hits["LONG BALL"] = a + int(i[0]) if len(i) else None
        if a > 0 and truth[a - 1] == OTHER[team] and period[a - 1] == period[a]:
            hits["ATTACKING TRANSITION"] = a
        out += [(code, row, team) for code, row in hits.items() if row is not None]
    return out


def _copied_team(decoded, a, b):
    """The decoder's team at the interval start, else its first H/A inside it."""
    live = np.flatnonzero(np.isin(decoded[a:b], ["H", "A"]))
    return decoded[a + live[0]] if len(live) else None


def _held_accuracy(team, a, b, truth, target):
    """(right, total) frames of [a, b) among `target` rows where `team` matches the truth."""
    rows = np.flatnonzero(target[a:b]) + a
    return int(np.sum(truth[rows] == team)), len(rows)


def intervals(trig, window_s, fps, n, period, truth=None, overrun_s=None):
    """[(code, start, stop, true team)] from each trigger, kept within its period.

    Two length rules, neither taken from the real tags:
        fixed      window_s from the trigger, whatever the possession does
        overrun    to the end of the true possession plus overrun_s (needs truth); a phase that
                   outlasts its possession by a set amount
    """
    out = []
    width = int(round(window_s * fps)) if window_s is not None else None
    for code, row, team in trig:
        if overrun_s is not None:
            end = row
            while end < n and truth[end] == team and period[end] == period[row]:
                end += 1
            stop = min(n, end + int(round(overrun_s * fps)))
        else:
            stop = min(n, row + width)
        same = np.flatnonzero(period[row:stop] != period[row])
        if len(same):
            stop = row + int(same[0])
        out.append((code, row, stop, team))
    return out


def held_accuracies(ivals, truth, decoded, missing):
    """(p_dec, p_true): accuracy of the copied and of the unflipped truth-based team, held over
    each interval, on its frames with the ball missing and play truly live."""
    target = missing & np.isin(truth, ["H", "A"])
    copy = [0, 0]
    true = [0, 0]
    for code, a, b, team in ivals:
        c = _copied_team(decoded, a, b)
        if c is not None:
            r, t = _held_accuracy(c, a, b, truth, target)
            copy[0] += r
            copy[1] += t
        r, t = _held_accuracy(team, a, b, truth, target)
        true[0] += r
        true[1] += t
    return copy[0] / max(copy[1], 1), true[0] / max(true[1], 1)


def solve_q(accuracy, r, p_dec, p_true):
    """Flip rate for truth-based tags giving overall `accuracy`; None if impossible."""
    if r >= 1:
        return None
    t = (accuracy - r * p_dec) / (1 - r)       # accuracy needed from the truth-based tags
    if p_true <= 0.5:
        return None
    q = (p_true - t) / (2 * p_true - 1)
    return float(q) if 0 <= q <= 1 else None


def make_tags(ivals, decoded, frame, r, q, seed):
    """Tags table (start, stop, code, team, type) in frame numbers, with mirror codes."""
    rng = np.random.default_rng(seed)
    rows = []
    for code, a, b, team in ivals:
        if rng.random() < r:
            t = _copied_team(decoded, a, b) or team
        else:
            t = OTHER[team] if rng.random() < q else team
        start, stop = int(frame[a]), int(frame[b - 1]) + 1
        rows.append((start, stop, code, t, ""))
        if code in MIRRORS:
            rows.append((start, stop, MIRRORS[code], OTHER[t], ""))
    return pd.DataFrame(rows, columns=["start", "stop", "code", "team", "type"])


def realised_accuracy(tags, frame, truth, missing, codes=None):
    """Accuracy actually reached, on frames inside a (non-mirror) tag with no ball and truly
    live play -- the check that the solve worked."""
    target = missing & np.isin(truth, ["H", "A"])
    right = total = 0
    for start, stop, code, team in zip(tags["start"], tags["stop"], tags["code"], tags["team"]):
        if code in MIRRORS.values() or (codes is not None and code not in codes):
            continue
        a, b = np.searchsorted(frame, [start, stop])
        r, t = _held_accuracy(team, a, b, truth, target)
        right += r
        total += t
    return right / max(total, 1)


def seed_for(key, *settings):
    return zlib.crc32("{}|{}".format(key, "|".join(map(str, settings))).encode())
