"""Restart model: P(home restarts) at each restart, from the laws of the game and the tracking.

A restart is where the decoder (a first pass without restart evidence) switches from dead ball to
live play. Each feature is antisymmetric -- positive favours home, negative away, and it flips sign
if the teams are swapped -- and the logistic regression has no intercept, so the model cannot learn
a home bias from a few hundred restarts. Directions come from the team centroids (gaps.py):
"home attacks this end" is +1, "away attacks it" -1.

    out-of-play rule  who was nearest the ball as play stopped (the other team restarts), split by
                      how play stopped: over a touchline, over a goal line, or on the pitch (a
                      foul: usually the team in possession restarts); and who had the ball before
    restart type      where the ball is when play resumes, else where it went out:
                      corner     -> the team attacking that end takes it
                      goal area  -> goal kick, the team defending that end
                      centre     -> kickoff after a goal: the team that conceded (the one defending
                                    the end the ball went out at)
    formation         each team's player nearest the restart spot (the taker stands at the ball),
                      and players in the penalty box at that end (attackers crowd it at corners)

Opening kickoffs carry no team information and get none (NaN -> 0.5 downstream).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..features.team import dense_positions
from .dead import off_pitch
from .gaps import home_direction, periods

FEATURES = ["last_touch_touchline", "last_touch_goalline", "last_touch_onpitch", "poss_before",
            "poss_before_onpitch", "corner", "goal_kick", "kickoff_conceded", "taker",
            "box_crowd"]

BOX_DEPTH, BOX_WIDTH = 16.5, 40.3
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.3
CORNER_M, TOUCHLINE_M, CENTRE_M = 3.0, 2.0, 3.0


def restart_rows(labels, period):
    """Rows where play restarts after a stoppage (D -> H/A within a period)."""
    live = np.isin(labels, ["H", "A"])
    return np.flatnonzero(live[1:] & (labels[:-1] == "D") & (period[1:] == period[:-1])) + 1


def _first_visible(x, rows):
    rows = [r for r in rows if 0 <= r < len(x) and not np.isnan(x[r])]
    return rows[0] if rows else None


def features(feats, labels, players, fps, pitch, rows=None):
    """[len(rows), len(FEATURES)] table for the restarts in `labels` (a first-pass decode)."""
    length, width = pitch
    period = feats["period"].to_numpy()
    rows = restart_rows(labels, period) if rows is None else np.asarray(rows)
    x, y = feats["ball_x"].to_numpy(float), feats["ball_y"].to_numpy(float)
    near_h, near_a = feats["near_H"].to_numpy(float), feats["near_A"].to_numpy(float)
    off = off_pitch(feats, pitch, 1.0)
    direction = home_direction(feats, pitch, periods(feats))   # +1: home attacks +x
    index = pd.Index(feats["frame"])
    pos = {t: dense_positions(players, index, t, pitch) for t in ("H", "A")}
    one, two, three = int(fps), int(2 * fps), int(3 * fps)

    out = np.zeros((len(rows), len(FEATURES)))
    for n, s in enumerate(rows):
        d0 = s - 1
        while d0 > 0 and labels[d0 - 1] == "D" and period[d0 - 1] == period[s]:
            d0 -= 1                                            # start of the stoppage
        before = labels[d0 - 1] if d0 > 0 and period[d0 - 1] == period[s] else None
        poss = {"H": 1.0, "A": -1.0}.get(before, 0.0)

        # how play stopped: the first sighting of the ball beyond a line, around the start of the
        # stoppage, else the last sighting before it (on the pitch: a foul)
        window = range(max(0, d0 - three), min(len(x), d0 + two))
        crossed = [r for r in window if not np.isnan(x[r]) and off[r]]
        if crossed:
            last = crossed[0]
            beyond_x = x[last] < 0 or x[last] > length
            how = "goalline" if beyond_x else "touchline"
        else:
            on = [r for r in range(max(0, d0 - two), d0) if not np.isnan(x[r])]
            last = on[-1] if on else None
            how = "onpitch"
        # the last touch: who was nearest 0.3-1.5 s BEFORE that moment, not the player who
        # walks to the ball after it (that is the taker, read below)
        who = np.nan
        if last is not None:
            touch = [r for r in range(max(0, last - int(1.5 * fps)), max(0, last - int(0.3 * fps)))
                     if not np.isnan(x[r])]
            if touch:
                who = float(np.mean(np.tanh(np.nan_to_num(near_a[touch], nan=50)
                                            - np.nan_to_num(near_h[touch], nan=50))))
        w = 0.0 if np.isnan(who) else who

        # where play resumes: first sighting from 1 s before to 3 s after, else where it went out
        spot = _first_visible(x, range(s - one, s + three)) or last
        f = dict.fromkeys(FEATURES, 0.0)
        f["last_touch_" + how] = w
        f["poss_before"] = poss
        f["poss_before_onpitch"] = poss if how == "onpitch" else 0.0
        if spot is not None:
            sx, sy = x[spot], y[spot]
            end = 1.0 if sx > length / 2 else -1.0
            home_attacks_end = direction[s] * end              # +1: home attacks this end
            to_line = min(sx, length - sx)
            from_mid = abs(sy - width / 2)
            if to_line < CORNER_M and from_mid > width / 2 - CORNER_M:
                f["corner"] = home_attacks_end
            elif to_line < GOAL_AREA_DEPTH + 1 and from_mid < GOAL_AREA_WIDTH / 2 + 1:
                f["goal_kick"] = -home_attacks_end
            elif np.hypot(sx - length / 2, sy - width / 2) < CENTRE_M and how == "goalline" \
                    and last is not None:
                out_end = 1.0 if x[last] > length / 2 else -1.0
                f["kickoff_conceded"] = -direction[s] * out_end  # the defenders of that end
            # formation: the taker at the spot, and who crowds the box at that end
            d = {t: np.nanmin(np.hypot(pos[t][s, :, 0] - sx, pos[t][s, :, 1] - sy))
                 if not np.all(np.isnan(pos[t][s, :, 0])) else 50.0 for t in pos}
            f["taker"] = float(np.tanh((d["A"] - d["H"]) / 2.0))
            gx = length if end > 0 else 0.0
            in_box = {t: np.sum((np.abs(pos[t][s, :, 0] - gx) < BOX_DEPTH)
                                & (np.abs(pos[t][s, :, 1] - width / 2) < BOX_WIDTH / 2))
                      for t in pos}
            # attackers crowd the box: weight by who attacks this end
            f["box_crowd"] = float(np.tanh((in_box["H"] - in_box["A"]) / 3.0)) * home_attacks_end
        out[n] = [f[k] for k in FEATURES]
    return pd.DataFrame(out, columns=FEATURES), rows


def true_team(truth, period, rows):
    """The team the true labels next give the ball to from each row, within its period."""
    out = []
    for s in rows:
        k = s
        while k < len(truth) and period[k] == period[s] and truth[k] not in ("H", "A"):
            k += 1
        out.append(truth[k] if k < len(truth) and period[k] == period[s] else None)
    return np.array(out, dtype=object)


def fit(X, y_home, C=1.0):
    """Logistic regression without intercept: symmetric in the teams by construction."""
    model = LogisticRegression(fit_intercept=False, C=C, max_iter=1000)
    model.fit(X.to_numpy(), y_home)
    return model


def p_home_rows(coef, X):
    return 1 / (1 + np.exp(-(X.to_numpy() @ np.asarray(coef))))


def p_home_array(n, rows, p):
    """Per-row array for decode(restart_p_home=...): p at each restart row, NaN elsewhere."""
    out = np.full(n, np.nan)
    out[rows] = p
    return out
