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
    dead   the dead-ball evidence from dead.py: off the pitch, stationary ball, players walking,
           long ball gaps, set-piece tags

    score H = live + gate * who / 2     score A = live - gate * who / 2     score D = dead

With no ball position the team scores are 0, so possession holds through gaps unless the
dead-ball cues say play has stopped.

Inside long gaps, `gap_probs` from the gap classifier (gaps.py) add to these scores when given:
    score s += GAP_W * (log P(s) - mean log P), per second.
Adding rather than replacing keeps the hold-through-gaps behaviour and the dead-ball cues.

`posterior` gives each frame's probability of H, A and D under the same scores and costs.

The constants are set from what they mean, not fitted to the provided game's labels (RULES.md).
Tune them on other labelled games.
"""

from dataclasses import dataclass

import numpy as np

from .dead import DeadParams, cues, off_pitch

STATES = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2


@dataclass(frozen=True)
class Params:
    closer_m: float = 1.0        # nearest-player margin that counts as clearly nearer
    gate_near_m: float = 1.5     # someone this close to the ball: full evidence
    gate_far_m: float = 4.0      # nobody this close: no evidence
    live_w: float = 0.25         # evidence per second that play is live, player at ball
    turnover_s: float = 1.0      # cost of H <-> A
    stop_s: float = 1.0          # cost of H/A -> D
    restart_s: float = 1.0       # cost of D -> H/A
    gap_w: float = 1.0           # evidence per second from the gap classifier's log-odds
    nats_per_s: float = 4.0      # posterior only: log-odds per second of evidence (calibrate)
    dead: DeadParams = DeadParams()


def emissions(feats, fps, pitch, tags=None, p=Params(), gap_probs=None):
    """[n, 3] per-frame scores for H, A, D."""
    near_h = feats["near_H"].to_numpy(float)
    near_a = feats["near_A"].to_numpy(float)
    has_ball = ~np.isnan(feats["ball_x"].to_numpy(float))

    big = 1e3  # a team with no visible player is never nearest
    nh, na = np.nan_to_num(near_h, nan=big), np.nan_to_num(near_a, nan=big)
    who = np.tanh((na - nh) / p.closer_m)
    gate = np.clip((p.gate_far_m - np.minimum(nh, na)) / (p.gate_far_m - p.gate_near_m), 0, 1)
    # a player beside a ball out of play is taking a restart, not possessing it
    gate[off_pitch(feats, pitch, p.dead.off_margin_m)] = 0
    gate[~has_ball] = 0

    score = np.zeros((len(feats), 3))
    score[:, H] = gate * (p.live_w + who / 2)
    score[:, A] = gate * (p.live_w - who / 2)
    score[:, D] = cues(feats, fps, pitch, tags, p.dead)["dead"].to_numpy()
    if gap_probs is not None:
        rows = ~np.isnan(gap_probs[:, 0])
        logp = np.log(np.clip(gap_probs[rows], 1e-3, 1))
        score[rows] += p.gap_w * (logp - logp.mean(axis=1, keepdims=True))
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


def forward_backward(score, cost):
    """[n, 3] state probabilities, treating exp(score - cost) as unnormalised potentials."""
    n, k = score.shape
    trans = -cost
    fwd = np.empty((n, k))
    bwd = np.zeros((n, k))
    fwd[0] = score[0] - np.logaddexp.reduce(score[0])
    for i in range(1, n):
        f = np.logaddexp.reduce(fwd[i - 1][:, None] + trans, axis=0) + score[i]
        fwd[i] = f - np.logaddexp.reduce(f)
    for i in range(n - 2, -1, -1):
        b = np.logaddexp.reduce(trans + score[i + 1] + bwd[i + 1], axis=1)
        bwd[i] = b - np.logaddexp.reduce(b)
    post = fwd + bwd
    post -= np.logaddexp.reduce(post, axis=1)[:, None]
    return np.exp(post)


def _periods(feats):
    period = feats["period"].to_numpy()
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    return zip(np.r_[0, cuts], np.r_[cuts, len(feats)])


def decode(feats, fps, pitch, tags=None, p=Params(), gap_probs=None):
    """H/A/D label per row of a features table, each period decoded on its own."""
    score = emissions(feats, fps, pitch, tags, p, gap_probs)
    cost = switch_costs(p)
    out = np.empty(len(feats), dtype=np.int8)
    for a, b in _periods(feats):
        out[a:b] = viterbi(score[a:b], cost)
    return STATES[out]


def posterior(feats, fps, pitch, tags=None, p=Params(), gap_probs=None):
    """[n, 3] probability of H, A, D per row, each period on its own.

    Scores and costs are scaled by nats_per_s to read as log-odds. That scaling leaves the
    Viterbi path unchanged; it only sets how confident the probabilities are.
    """
    score = emissions(feats, fps, pitch, tags, p, gap_probs) * p.nats_per_s
    cost = switch_costs(p) * p.nats_per_s
    out = np.empty((len(feats), 3))
    for a, b in _periods(feats):
        out[a:b] = forward_backward(score[a:b], cost)
    return out
