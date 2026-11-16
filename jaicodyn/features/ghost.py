"""Cross-team merges: a player's track vanishing into an opponent's, and the ghost that undoes it.

When two players of different teams stand close, the tracker can lose one of them into the other:
the vanished player's track ends right beside the opponent, and that team has nobody there until
the tracker separates them again. On the competition match, tracks end within NEAR_M of an opponent
4.9 times more often than a random player-frame is that close, about 2.1 such merges a minute, and
the vanished team is back beside the survivor after a median 2.1 s.

A merge here is a track ending within NEAR_M of an opponent, the opponent (the survivor) still
tracked on the next frame, and no player of the vanished team within BACK_M of the survivor on
that frame -- otherwise the track was only renumbered (an ID switch). It lasts until a player of
the vanished team is back within BACK_M of the survivor, the survivor's track ends, or MAX_S.

The ghost keeps the vanished team present at the survivor's position for the merge, so both teams
read as equally close to a contested ball instead of handing it to the survivor.
"""

import numpy as np
import pandas as pd

NEAR_M = 1.5
BACK_M = 2.0
MAX_S = 10.0


class Tracks:
    """Players in metres, sorted by frame, with fast per-frame slices."""

    def __init__(self, players, pitch):
        length, width = pitch
        p = players[players["x"].notna()].sort_values(["frame", "team", "track_id"])
        self.frame = p["frame"].to_numpy()
        self.team = p["team"].to_numpy()
        self.track = p["track_id"].astype(str).to_numpy()
        self.x = p["x"].to_numpy(float) * length
        self.y = p["y"].to_numpy(float) * width
        self.index = p.index.to_numpy()

    def at(self, f):
        a, b = np.searchsorted(self.frame, [f, f + 1])
        return slice(a, b)

    def run_ends(self):
        """Rows where a (team, track) is seen for the last time before a break."""
        order = np.lexsort((self.frame, self.track, self.team))
        f, t, k = self.frame[order], self.team[order], self.track[order]
        last = np.r_[(f[1:] != f[:-1] + 1) | (t[1:] != t[:-1]) | (k[1:] != k[:-1]), True]
        return order[last]


def merges(players, frames, pitch, fps, near_m=NEAR_M, back_m=BACK_M, max_s=MAX_S):
    """[(vanished team, survivor track, first missing frame, end frame exclusive, why)]."""
    tr = Tracks(players, pitch)
    period = dict(zip(frames["frame"], frames["period"]))
    out = []
    for r in tr.run_ends():
        f, team = tr.frame[r], tr.team[r]
        if period.get(f + 1) != period.get(f) or (f + 1) not in period:
            continue
        here = tr.at(f)
        opp = np.flatnonzero(tr.team[here] != team) + here.start
        if not len(opp):
            continue
        d = np.hypot(tr.x[opp] - tr.x[r], tr.y[opp] - tr.y[r])
        if d.min() >= near_m:
            continue
        survivor = tr.track[opp[np.argmin(d)]]
        end, why = None, "limit"
        for k in range(1, int(max_s * fps) + 1):
            s = tr.at(f + k)
            rows = np.arange(s.start, s.stop)
            surv = rows[(tr.team[rows] != team) & (tr.track[rows] == survivor)]
            if not len(surv) or period.get(f + k) != period.get(f):
                end, why = f + k, "survivor lost"
                break
            mine = rows[tr.team[rows] == team]
            if len(mine) and np.min(np.hypot(tr.x[mine] - tr.x[surv[0]],
                                             tr.y[mine] - tr.y[surv[0]])) < back_m:
                end, why = f + k, "back"
                break
        if end is None:
            end = f + int(max_s * fps) + 1
        if why == "back" and end == f + 1:
            continue                                   # replaced at once: an ID switch
        out.append((team, survivor, f + 1, end, why))
    return out


def ghost_rows(players, frames, pitch, fps, events=None):
    """Players-table rows placing each vanished team at its survivor's position during a merge."""
    length, width = pitch
    tr = Tracks(players, pitch)
    events = merges(players, frames, pitch, fps) if events is None else events
    rows = []
    for team, survivor, start, end, _ in events:
        for f in range(start, end):
            s = tr.at(f)
            idx = np.arange(s.start, s.stop)
            surv = idx[(tr.team[idx] != team) & (tr.track[idx] == survivor)]
            if len(surv):
                rows.append((f, team, "ghost", tr.x[surv[0]] / length, tr.y[surv[0]] / width))
    return pd.DataFrame(rows, columns=["frame", "team", "track_id", "x", "y"])
