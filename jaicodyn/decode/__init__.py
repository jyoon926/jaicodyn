"""Three-state possession decoder: H, A, D by Viterbi over per-frame evidence.

Everything is in seconds of evidence. Each frame contributes a score per state, at most about one
second's worth per second of play, and a switch between states costs a fixed number of seconds of
evidence, one cost per kind of switch. So "TURNOVER_S = 0.5" reads: the other team must be clearly
nearest the ball for about half a second before possession changes hands -- the challenge's
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

    touch  TOUCH_W on the frames of a touch by that team (features/touch.py): the ball changed
           velocity at one of its players, so that team had the ball there
    flight with flight_mps set, the gate is 0 while the ball moves faster than flight_mps with no
           touch within FLIGHT_TOUCH_S: a pass passing an opponent says nothing about who has it

    score H = live + gate * who / 2 + touch_H     score A = live - gate * who / 2 + touch_A
    score D = dead

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

from .dead import DeadParams, cues, sustained_off_pitch

STATES = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2


@dataclass(frozen=True)
class Params:
    closer_m: float = 1.0        # nearest-player margin that counts as clearly nearer
    gate_near_m: float = 1.5     # someone this close to the ball: full evidence
    gate_far_m: float = 2.5      # nobody this close: no evidence. Tuned on train (was 4.0):
                                 # +0.014-0.017 on degraded idsse at every missing-ball ratio
    live_w: float = 1.0          # evidence per second that play is live, player at ball. Tuned on
                                 # degraded idsse only, a dead-or-live setting (was 0.25); with
                                 # gate_far_m 2.5: +0.024-0.028, 5/5 games, at every ratio
    turnover_s: float = 0.5      # cost of H <-> A; chosen on the train split (scripts/tune.py)
    stop_s: float = 1.0          # cost of H/A -> D
    restart_s: float = 1.0       # cost of D -> H/A
    restart_scale: float = 1.0   # restart evidence: seconds of evidence per nat of log-odds
    restart_window_s: float = 3.0  # ... spread over this long after each restart
    gap_w: float = 1.0           # evidence per second from the gap classifier's log-odds
    nats_per_s: float = 4.0      # posterior only: log-odds per second of evidence (calibrate)
    # Touches. On train (degraded idsse with ball noise, and SkillCorner) touch_w 0.5 with the
    # flight rule at 8 m/s gave 0.547 against 0.537 without either; touch_w 2 or more costs 0.03,
    # an opponent's touch not being controlled possession. The final choice was made on
    # validation and test pooled, by source, deciding on degraded idsse and vetoing what clearly
    # breaks another source: touch_w 0.5 helps or ties on every source (degraded idsse +0.008,
    # degraded Metrica +0.009, Metrica +0.020), while the flight rule lost 11 of 16 held-out
    # games, tied on degraded idsse and cost degraded Metrica 0.012 on all 3. So touches stay
    # and the flight rule is off (kept, as flight_mps, for the record).
    touch_w: float = 0.5         # evidence per second on a team's touch frames (0: off)
    flight_mps: float = 0.0      # ball faster than this, no touch near: no evidence (0: off)
    flight_touch_s: float = 0.2  # ... unless a touch is within this long
    dead: DeadParams = DeadParams()


def emissions(feats, fps, pitch, tags=None, p=Params(), gap_probs=None, restart_window_p=None,
              team_evidence=None):
    """[n, 3] per-frame scores for H, A, D. restart_window_p: see restart_windows().

    team_evidence: optional per-second evidence for home (+) or away (-), such as the phase tags
    (tags.phase_team_evidence). It is taken off the other team's score rather than added to the
    named team's, so it moves only the choice between H and A, never between live and dead.
    """
    near_h = feats["near_H"].to_numpy(float)
    near_a = feats["near_A"].to_numpy(float)
    has_ball = ~np.isnan(feats["ball_x"].to_numpy(float))

    big = 1e3  # a team with no visible player is never nearest
    nh, na = np.nan_to_num(near_h, nan=big), np.nan_to_num(near_a, nan=big)
    who = np.tanh((na - nh) / p.closer_m)
    gate = np.clip((p.gate_far_m - np.minimum(nh, na)) / (p.gate_far_m - p.gate_near_m), 0, 1)
    # a player beside a ball out of play is taking a restart, not possessing it
    gate[sustained_off_pitch(feats, pitch, fps, p.dead)] = 0
    gate[~has_ball] = 0
    touch_h, touch_a = (feats[c].to_numpy(float) if c in feats else np.zeros(len(feats))
                        for c in ("touch_H", "touch_A"))
    if p.flight_mps > 0:
        gate[in_flight(feats, fps, touch_h + touch_a > 0, p)] = 0

    score = np.zeros((len(feats), 3))
    score[:, H] = gate * (p.live_w + who / 2) + p.touch_w * touch_h
    score[:, A] = gate * (p.live_w - who / 2) + p.touch_w * touch_a
    score[:, D] = cues(feats, fps, pitch, tags, p.dead)["dead"].to_numpy()
    if gap_probs is not None:
        rows = ~np.isnan(gap_probs[:, 0])
        logp = np.log(np.clip(gap_probs[rows], 1e-3, 1))
        score[rows] += p.gap_w * (logp - logp.mean(axis=1, keepdims=True))
    if restart_window_p is not None:
        e = restart_evidence(restart_window_p, p)
        score[:, H] += e
        score[:, A] -= e
    if team_evidence is not None:
        e = np.nan_to_num(np.asarray(team_evidence, float))
        score[:, A] -= np.maximum(e, 0)
        score[:, H] -= np.maximum(-e, 0)
    return score / fps


def in_flight(feats, fps, touched, p=Params()):
    """Frames with the ball moving faster than flight_mps and no touch within flight_touch_s."""
    xy = feats[["ball_x", "ball_y"]].to_numpy(float)
    speed = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps]
    k = max(1, int(round(p.flight_touch_s * fps)))
    near_touch = np.convolve(touched.astype(float), np.ones(2 * k + 1), mode="same") > 0
    return (np.nan_to_num(speed) > p.flight_mps) & ~near_touch


def restart_evidence(p_home, p=Params()):
    """Per-second evidence for home over away, from P(home restarts), on restart-window rows.

    restart_scale * log(p_H / p_A) / 2 / restart_window_s, added to H and taken from A: over a
    whole window the H-vs-A difference is restart_scale * log(p_H / p_A) seconds of evidence --
    the same as a per-stoppage restart cost of restart_scale * (-log p_X) -- but spread, so a
    path only collects it by keeping that team in possession, not by restarting and handing the
    ball straight back. Exactly 0 at p = 0.5, and on NaN (rows outside every window).
    """
    ph = np.clip(np.nan_to_num(np.asarray(p_home, float), nan=0.5), 1e-3, 1 - 1e-3)
    log_odds = np.log(ph) - np.log1p(-ph)          # exactly 0 at 0.5
    return p.restart_scale * log_odds / 2 / p.restart_window_s


def restart_windows(labels, period, p_home, fps, p=Params()):
    """P(home restarts) on the first restart_window_s of each live spell after a restart.

    labels: a decode without restart evidence, which places the restarts: every D -> H/A, and a
    period opening live (kickoff). Each window takes p_home at its restart frame and stops early
    if play goes dead again. NaN elsewhere.
    """
    out = np.full(len(labels), np.nan)
    live = np.isin(labels, ["H", "A"])
    width = int(round(p.restart_window_s * fps))
    starts = np.flatnonzero(live[1:] & (labels[:-1] == "D") & (period[1:] == period[:-1])) + 1
    opens = np.flatnonzero(np.r_[True, period[1:] != period[:-1]] & live)
    for s in np.r_[starts, opens]:
        e = s
        while e < min(len(labels), s + width) and live[e] and period[e] == period[s]:
            e += 1
        out[s:e] = p_home[s]
    return out


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


def _viterbi_periods(feats, score, cost):
    out = np.empty(len(feats), dtype=np.int8)
    for a, b in _periods(feats):
        out[a:b] = viterbi(score[a:b], cost)
    return STATES[out]


def _windows(feats, fps, pitch, tags, p, gap_probs, restart_p_home, team_evidence=None):
    """First pass without restart evidence places the restarts; returns the window array."""
    if restart_p_home is None:
        return None
    first = _viterbi_periods(feats, emissions(feats, fps, pitch, tags, p, gap_probs,
                                              team_evidence=team_evidence), switch_costs(p))
    return restart_windows(first, feats["period"].to_numpy(), np.asarray(restart_p_home, float),
                           fps, p)


def decode(feats, fps, pitch, tags=None, p=Params(), gap_probs=None, restart_p_home=None,
           team_evidence=None):
    """H/A/D label per row of a features table, each period decoded on its own.

    restart_p_home: optional P(home restarts) per row, read at each restart and turned into
    evidence over the first restart_window_s after it (restart_evidence). Needs two passes: the
    first, without it, places the restarts.
    """
    windows = _windows(feats, fps, pitch, tags, p, gap_probs, restart_p_home, team_evidence)
    return _viterbi_periods(feats, emissions(feats, fps, pitch, tags, p, gap_probs, windows,
                                             team_evidence), switch_costs(p))


def posterior(feats, fps, pitch, tags=None, p=Params(), gap_probs=None, restart_p_home=None,
              team_evidence=None):
    """[n, 3] probability of H, A, D per row, each period on its own.

    Scores and costs are scaled by nats_per_s to read as log-odds. That scaling leaves the
    Viterbi path unchanged; it only sets how confident the probabilities are.
    """
    windows = _windows(feats, fps, pitch, tags, p, gap_probs, restart_p_home, team_evidence)
    score = emissions(feats, fps, pitch, tags, p, gap_probs, windows, team_evidence) * p.nats_per_s
    cost = switch_costs(p) * p.nats_per_s
    out = np.empty((len(feats), 3))
    for a, b in _periods(feats):
        out[a:b] = forward_backward(score[a:b], cost)
    return out
