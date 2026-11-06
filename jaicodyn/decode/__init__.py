"""Three-state possession decoder: H, A, D by Viterbi over per-frame evidence.

Everything is in seconds of evidence. Each frame contributes a score per state, at most about one
second's worth per second of play, and a switch between states costs a fixed number of seconds of
evidence, one cost per kind of switch. So "TURNOVER_S = 1.0" reads: the other team must be clearly
nearest the ball for about a second before possession changes hands -- the challenge's
"controlled possession" (a player keeps the ball, rather than a touch or deflection). Contested
and loose balls give little evidence either way, so possession stays with the last team, as the
rules ask.

Evidence per frame (features from jaicodyn.features; per second, divided by fps per frame):

    who    tanh((near_other - near_team) / CLOSER_M)   +1 when the team is clearly nearer
    gate   1 when someone is within GATE_NEAR_M of the ball, fading to 0 at GATE_FAR_M:
           a ball in flight or far from everyone says nothing about who has it
    live   gate * LIVE_W: a player at the ball is weak evidence the ball is in play
    off    OFF_W when the ball is off the pitch by more than OFF_MARGIN_M: strong evidence dead

    score H = live - off + gate * who / 2     score A = live - off - gate * who / 2     score D = 0

A frame with no ball position scores 0 everywhere, so the state simply holds through gaps.

The constants are set from what they mean, not fitted to the provided game's labels (RULES.md).
Tune them on other labelled games.
"""

from dataclasses import dataclass

import numpy as np

STATES = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2


@dataclass(frozen=True)
class Params:
    closer_m: float = 1.0        # nearest-player margin that counts as clearly nearer
    gate_near_m: float = 1.5     # someone this close to the ball: full evidence
    gate_far_m: float = 4.0      # nobody this close: no evidence
    live_w: float = 0.25         # evidence per second that play is live, player at ball
    off_w: float = 2.0           # evidence per second that play is dead, ball off the pitch
    off_margin_m: float = 1.0    # beyond the lines by this much counts as off
    turnover_s: float = 1.0      # cost of H <-> A
    stop_s: float = 1.0          # cost of H/A -> D
    restart_s: float = 1.0       # cost of D -> H/A


def emissions(feats, fps, pitch, p=Params()):
    """[n, 3] per-frame scores for H, A, D."""
    length, width = pitch
    near_h = feats["near_H"].to_numpy(float)
    near_a = feats["near_A"].to_numpy(float)
    x, y = feats["ball_x"].to_numpy(float), feats["ball_y"].to_numpy(float)
    has_ball = ~np.isnan(x)

    big = 1e3  # a team with no visible player is never nearest
    nh, na = np.nan_to_num(near_h, nan=big), np.nan_to_num(near_a, nan=big)
    who = np.tanh((na - nh) / p.closer_m)
    gate = np.clip((p.gate_far_m - np.minimum(nh, na)) / (p.gate_far_m - p.gate_near_m), 0, 1)
    off = (x < -p.off_margin_m) | (x > length + p.off_margin_m) \
        | (y < -p.off_margin_m) | (y > width + p.off_margin_m)
    gate[off] = 0  # a player beside a ball out of play is taking a restart, not possessing it

    live = gate * p.live_w - off * p.off_w
    score = np.zeros((len(feats), 3))
    score[:, H] = live + gate * who / 2
    score[:, A] = live - gate * who / 2
    score[~has_ball] = 0
    return score / fps


def switch_costs(p=Params()):
    """[3, 3] cost of moving from row state to column state, in seconds of evidence."""
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = p.turnover_s
    c[H, D] = c[A, D] = p.stop_s
    c[D, H] = c[D, A] = p.restart_s
    return c


def viterbi(score, cost):
    """Best state path maximising summed score minus switch costs. score [n, 3], cost [3, 3]."""
    n, k = score.shape
    back = np.zeros((n, k), dtype=np.int8)
    best = score[0].copy()
    for i in range(1, n):
        cand = best[:, None] - cost          # cand[from, to]
        back[i] = np.argmax(cand, axis=0)
        best = cand[back[i], np.arange(k)] + score[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(best))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(feats, fps, pitch, p=Params()):
    """H/A/D label per row of a features table, each period decoded on its own."""
    score = emissions(feats, fps, pitch, p)
    cost = switch_costs(p)
    period = feats["period"].to_numpy()
    out = np.empty(len(feats), dtype=np.int8)
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(feats)]):
        out[a:b] = viterbi(score[a:b], cost)
    return STATES[out]
