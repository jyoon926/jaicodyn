"""Gap classifier: P(H), P(A), P(D) for frames where the ball is not seen.

Almost all missing ball frames sit in gaps longer than 5 s, where the decoder otherwise only holds
its last state. This model reads the context of a gap instead:

    gap        length, position of the frame inside it, seconds since start / to end
    ends       who was nearest the ball in the CTX_S before and after, where the ball was last and
               next seen (in the home team's attacking direction), whether it was off the pitch
    players    median speed relative to the match's, over the gap, its first and last seconds,
               and at the frame
    shape      each team's centroid at the frame and its advance over the gap, in each team's
               attacking direction (read from the centroids: goalkeepers pull a team's centroid
               towards its own goal)
    tags       seconds since the last and until the next automatic set-piece tag, any tag near
               the gap

Training labels are pseudo-labels, no human labels: the base decoder's confident states with the
ball visible. Gaps of SIM_MIN_S to SIM_MAX_S are simulated by hiding the ball on fully visible
stretches. Confident frames inside real gaps ("hidden-ball anchors") can be mixed in at HIDDEN_W,
but are off by default: on BAR-ATH only dead-ball frames clear the confidence bar inside real
gaps, so the anchors taught "unseen means dead" and halved held-out agreement. Labels from other
games can replace the pseudo-labels unchanged.

Predictions are cross-fitted by period: frames in one half come from a model trained on the other,
so no frame is predicted by a model that saw its own pseudo-label.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..features.ball import true_runs
from .dead import off_pitch

CLASSES = np.array(["H", "A", "D"])
BALL_COLUMNS = ["ball_x", "ball_y", "near_H", "near_A", "cnt2_H", "cnt5_H", "cnt10_H",
                "cnt2_A", "cnt5_A", "cnt10_A"]


@dataclass(frozen=True)
class GapParams:
    min_gap_s: float = 5.0       # gaps shorter than this are left to the decoder
    sim_min_s: float = 5.0
    sim_max_s: float = 30.0
    ctx_s: float = 2.0           # visible context needed, and read, on each side
    step: int = 5                # training rows every this many frames
    rounds: int = 4              # simulated gap layouts per period
    tries: int = 3000            # placement attempts per round and period
    conf: float = 0.9            # pseudo-label confidence for simulated gaps
    hidden_conf: float = 0.95    # ... and for hidden-ball anchors
    hidden_w: float = 0.0        # sample weight of hidden-ball anchors; off: they are almost all
                                 # dead ball, and teach "unseen means dead" (see the docstring)
    seed: int = 0


def periods(feats):
    period = feats["period"].to_numpy()
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    return list(zip(np.r_[0, cuts], np.r_[cuts, len(feats)]))


def gap_runs(hidden, blocks, min_len):
    """(start, stop) row ranges of hidden runs at least min_len long, never across a period."""
    out = []
    for pa, pb in blocks:
        for s, n in zip(*true_runs(hidden[pa:pb])):
            if n >= min_len:
                out.append((pa + s, pa + s + n))
    return out


def home_direction(feats, pitch, blocks):
    """+1 per row where home attacks towards +x, -1 where towards -x."""
    out = np.ones(len(feats))
    for a, b in blocks:
        h = np.nanmedian(feats["cx_H"].to_numpy(float)[a:b])
        w = np.nanmedian(feats["cx_A"].to_numpy(float)[a:b])
        out[a:b] = 1.0 if w >= h else -1.0
    return out


def mask_ball(feats, hidden):
    """Copy of feats with every ball-derived column blanked on hidden rows."""
    out = feats.copy()
    for c in BALL_COLUMNS:
        if c in out:
            out.loc[hidden, c] = np.nan
    if "ball_src" in out:
        out.loc[hidden, "ball_src"] = 0
    return out


def _rolling(x, n):
    return pd.Series(x).rolling(max(int(n), 1), center=True, min_periods=1).mean().to_numpy()


def _tag_rows(feats, tags):
    if tags is None or not len(tags):
        return np.array([], dtype=int)
    starts = tags.loc[tags["code"].str.upper() == "SET PIECES", "start"].to_numpy()
    return np.searchsorted(feats["frame"].to_numpy(), starts)


class Context:
    """Per-row arrays shared by every gap of a match."""

    def __init__(self, feats, fps, pitch, tags):
        self.fps, self.pitch = fps, pitch
        self.blocks = periods(feats)
        length, width = pitch
        self.dir = home_direction(feats, pitch, self.blocks)
        speed = feats["speed_med"].to_numpy(float)
        self.sr = _rolling(speed, fps) / np.nanmedian(speed)
        self.cxh = self.dir * (feats["cx_H"].to_numpy(float) - length / 2)
        self.cxa = self.dir * (feats["cx_A"].to_numpy(float) - length / 2)
        self.x = feats["ball_x"].to_numpy(float)
        self.y = feats["ball_y"].to_numpy(float)
        self.near_h = feats["near_H"].to_numpy(float)
        self.near_a = feats["near_A"].to_numpy(float)
        self.off = off_pitch(feats, pitch, 1.0)
        self.period_of = np.empty(len(feats), dtype=int)
        for k, (a, b) in enumerate(self.blocks):
            self.period_of[a:b] = k
        n = len(feats)
        tag = _tag_rows(feats, tags)
        idx = np.arange(n)
        last = np.full(n, -10 ** 9)
        nxt = np.full(n, 10 ** 9)
        if len(tag):
            pos = np.searchsorted(tag, idx, side="right")
            has_last = pos > 0
            last[has_last] = tag[pos[has_last] - 1]
            has_next = pos < len(tag)
            nxt[has_next] = tag[pos[has_next]]
        self.since_tag = np.minimum((idx - last) / fps, 120.0)
        self.until_tag = np.minimum((nxt - idx) / fps, 120.0)

    def rows(self, hidden, runs, step):
        """Feature rows for frames of each run, every `step` frames. Returns (table, row index)."""
        fps, length, width = self.fps, *self.pitch
        ctx = int(round(2.0 * fps))
        three = int(round(3.0 * fps))
        who = np.tanh((self.near_a - self.near_h) / 1.0)  # +1 home nearer
        who[hidden] = np.nan
        parts, index = [], []
        for a, b in runs:
            pa, pb = self.blocks[self.period_of[a]]
            pre, post = slice(max(pa, a - ctx), a), slice(b, min(pb, b + ctx))
            vis_pre = np.flatnonzero(~hidden[pre] & ~np.isnan(self.x[pre])) + pre.start
            vis_post = np.flatnonzero(~hidden[post] & ~np.isnan(self.x[post])) + post.start
            last = vis_pre[-1] if len(vis_pre) else None
            first = vis_post[0] if len(vis_post) else None
            one = max(1, int(fps))

            def ball_at(i, dirn):
                if i is None:
                    return np.nan, np.nan, np.nan
                return dirn * (self.x[i] - length / 2), abs(self.y[i] - width / 2), float(self.off[i])

            bx0, by0, off0 = ball_at(last, self.dir[a])
            bx1, by1, off1 = ball_at(first, self.dir[a])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # empty context gives NaN
                run = {
                    "gap_s": (b - a) / fps,
                    "who_before": np.nanmean(who[vis_pre]) if len(vis_pre) else np.nan,
                    "who_after": np.nanmean(who[vis_post]) if len(vis_post) else np.nan,
                    "ball_x_before": bx0, "ball_y_before": by0, "off_before": off0,
                    "ball_x_after": bx1, "ball_y_after": by1, "off_after": off1,
                    "sr_gap": np.nanmean(self.sr[a:b]),
                    "sr_first": np.nanmean(self.sr[a:a + three]),
                    "sr_last": np.nanmean(self.sr[max(a, b - three):b]),
                    "sr_before": np.nanmean(self.sr[max(pa, a - three):a]) if a > pa else np.nan,
                    "adv_H": np.nanmean(self.cxh[max(a, b - one):b]) - np.nanmean(self.cxh[a:a + one]),
                    "adv_A": -(np.nanmean(self.cxa[max(a, b - one):b])
                               - np.nanmean(self.cxa[a:a + one])),
                    "tag_near": float(np.any((self.since_tag[a:b] < 5) | (self.until_tag[a:b] < 5))),
                }
            i = np.arange(a, b, step)
            frame = {
                "u": (i - a) / (b - a), "t_since": (i - a) / fps, "t_to": (b - i) / fps,
                "sr": self.sr[i], "cx_H": self.cxh[i], "cx_A": self.cxa[i],
                "since_tag": self.since_tag[i], "until_tag": self.until_tag[i],
            }
            part = pd.DataFrame(frame)
            for k, v in run.items():
                part[k] = v
            parts.append(part)
            index.append(i)
        if not parts:
            return pd.DataFrame(), np.array([], dtype=int)
        return pd.concat(parts, ignore_index=True), np.concatenate(index)


def simulate(hidden, blocks, fps, rng, p):
    """A mask of non-overlapping simulated gaps on fully visible stretches, inside each period."""
    sim = np.zeros(len(hidden), dtype=bool)
    ctx = int(round(p.ctx_s * fps))
    lo, hi = int(p.sim_min_s * fps), int(p.sim_max_s * fps)
    blocked = hidden.copy()
    for pa, pb in blocks:
        for _ in range(p.tries):
            n = int(rng.integers(lo, hi + 1))
            if pb - pa < n + 2 * ctx:
                break
            s = int(rng.integers(pa + ctx, pb - n - ctx))
            if blocked[s - ctx:s + n + ctx].any():
                continue
            sim[s:s + n] = True
            blocked[s - ctx:s + n + ctx] = True
    return sim


def training_rows(feats, fps, pitch, tags, labels, conf, p=GapParams()):
    """(X, y, weight, period, kind) for simulated gaps and hidden-ball anchors.

    labels: H/A/D per row (pseudo or true). conf: per-row confidence in them (1 for true labels).
    """
    ctx = Context(feats, fps, pitch, tags)
    missing = feats["ball_x"].isna().to_numpy()
    min_len = int(p.min_gap_s * fps)
    rng = np.random.default_rng(p.seed)
    X, y, w, per, kind = [], [], [], [], []

    def add(table, idx, threshold, weight, name):
        keep = conf[idx] >= threshold
        X.append(table[keep])
        y.append(labels[idx][keep])
        w.append(np.full(keep.sum(), weight))
        per.append(ctx.period_of[idx][keep])
        kind.append(np.full(keep.sum(), name))

    for _ in range(p.rounds):
        sim = simulate(missing, ctx.blocks, fps, rng, p)
        hidden = missing | sim
        table, idx = ctx.rows(hidden, gap_runs(sim, ctx.blocks, min_len), p.step)
        add(table, idx, p.conf, 1.0, "simulated")
    if p.hidden_w > 0:
        table, idx = ctx.rows(missing, gap_runs(missing, ctx.blocks, min_len), p.step)
        add(table, idx, p.hidden_conf, p.hidden_w, "hidden")
    return (pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(w),
            np.concatenate(per), np.concatenate(kind))


def make_model(seed=0):
    # Rows of one gap share most features, so the effective sample is a few hundred gaps: small,
    # well-populated leaves keep the model from memorising whole gaps.
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def fit(X, y, w, seed=0):
    model = make_model(seed)
    model.fit(X, y, sample_weight=w)
    return model


def predict(model, X):
    """[n, 3] probabilities in CLASSES order, whatever classes the model saw."""
    out = np.full((len(X), 3), 1e-3)
    if len(X):
        proba = model.predict_proba(X)
        for j, c in enumerate(model.classes_):
            out[:, list(CLASSES).index(c)] = proba[:, j]
    return out / out.sum(axis=1, keepdims=True)


def gap_probs(feats, fps, pitch, tags, labels, conf, hidden=None, p=GapParams()):
    """[n, 3] cross-fitted probabilities on frames of gaps >= min_gap_s, NaN elsewhere.

    hidden: which rows to treat as unseen (default: where the ball is missing). Each period is
    predicted by a model trained on the other periods' rows.
    """
    X, y, w, per, _ = training_rows(feats, fps, pitch, tags, labels, conf, p)
    ctx = Context(feats, fps, pitch, tags)
    if hidden is None:
        hidden = feats["ball_x"].isna().to_numpy()
    table, idx = ctx.rows(hidden, gap_runs(hidden, ctx.blocks, int(p.min_gap_s * fps)), 1)
    out = np.full((len(feats), 3), np.nan)
    for k in range(len(ctx.blocks)):
        train = per != k
        target = ctx.period_of[idx] == k
        if not train.any() or not target.any():
            continue
        model = fit(X[train], y[train], w[train], p.seed)
        out[idx[target]] = predict(model, table[target])
    return out
