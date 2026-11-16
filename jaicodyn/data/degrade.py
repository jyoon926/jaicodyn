"""Make a clean labelled game look like the competition tracking: hide the ball in long gaps.

idsse tracks the ball on every frame; the competition match loses it on about 41% of frames,
almost all in gaps over 5 s, and far more often while the ball is dead. Tuning on clean data fits
weights to conditions the test match does not have, so this hides the ball to match:

    missing        share of graded frames hidden: the competition match's, from its tracking alone
    gap_lengths_s  gap lengths, drawn from the competition match's own gaps of 5 s or more
    dead_ratio     how many times likelier a dead frame is hidden than a live one, in the
                   competition match: estimated from its player speeds (with P(dead | speed)
                   from idsse train labels) and checked against its automatic set-piece tags
    line_weights   how much likelier a gap starts with the ball near or beyond a line: the ball
                   leaving the competition's wide shot. Applied as line_weights_applied, the
                   weights that reproduce this pattern on the idsse train games (placing gaps over
                   idsse's own stoppages already puts many starts at the lines)

    merge_*        cross-team merges: how often a player next to an opponent vanishes into them
                   (per second of being within 1.5 m), and for how long -- from the competition
                   match's tracking (features/ghost.py)

    ball_noise_*   a slowly drifting error on the ball position, so the ball sits further from the
                   player at it, as in the competition match. With the ball moving at dribbling
                   speed (1-6 m/s) it is 0.3-0.4 m further from its nearest player there than in
                   idsse at the median and quartile, yet its track is no rougher than idsse's at
                   any lag from 1 frame to 1.3 s (the tracker smooths). So the error is a smooth
                   offset: per axis, white noise smoothed by a Gaussian of ball_noise_tau_s and
                   scaled to ball_noise_sigma_m, fitted by adding it to the idsse train games
                   until those quantiles match (scripts.degrade --measure). It has to be smooth: a
                   rough error (an Ornstein-Uhlenbeck drift of the same size) gives a still ball
                   over 1 m/s of false speed, which hides every stoppage from the still-ball cue.
                   The whole error goes on the ball: the features see only the ball-to-player
                   offset, and the competition's players are no rougher than idsse's. False
                   detections (15 short runs in the competition match) are too rare to add.

All are measured by `python -m scripts.degrade --measure` into config/degrade.json. No
competition labels are used. SkillCorner's own ratio (11x) is not: its TV feed cuts to replays
during stoppages, a different reason for losing the ball.

Each gap is centred on a frame drawn from the dead or the live frames, whichever is further below
its target, and kept with probability proportional to the line weight where it starts. A gap over
a stoppage can be cut to the stoppage plus stoppage_spill_s of play each side; that was needed
at SkillCorner's 11x, and makes no difference at the competition's ratio, so it is off. The seed
comes from the match id, so the output is the same every run.
"""

import json
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..features.ball import true_runs
from . import paths
from .schema import Tracking

CONFIG = paths.ROOT / "config" / "degrade.json"  # written by python -m scripts.degrade --measure
MARGIN_S = 1.0  # visible ball kept between gaps
SPILL_S = 1.5   # a stoppage gap may start / end this far into play around the stoppage
MIN_GAP_S = 5.0  # the competition match's missing ball is almost all in gaps this long or more


def settings():
    """{"missing", "dead_ratio", "gap_lengths_s"} as measured; see scripts/degrade.py."""
    return json.loads(CONFIG.read_text("utf-8"))


def targets(labels, missing, dead_ratio):
    """(hidden share of dead frames, of live frames) giving `missing` overall at `dead_ratio`."""
    graded = labels != "X"
    dead = np.mean(labels[graded] == "D")
    live_rate = missing / (dead * dead_ratio + (1 - dead))
    dead_rate = min(0.95, live_rate * dead_ratio)
    if dead_rate == 0.95:  # capped: give the rest to live frames to keep the overall share
        live_rate = (missing - dead * dead_rate) / (1 - dead)
    return dead_rate, live_rate


def hide_mask(labels, period, fps, seed, missing, dead_ratio, lengths, line_dist=None,
              bands=None, weights=None, spill_s=SPILL_S, already=None):
    """Boolean mask of frames whose ball to hide.

    already: frames whose ball the source itself already lacks. They count towards the targets,
    so only the shortfall is hidden; gaps may run into them. Metrica blanks nearly every dead
    frame, so there only live ball gets hidden, at the competition's live rate.

    line_dist: metres from the ball to the nearest line per frame (negative off the pitch).
    With bands and weights, a proposed gap is kept with probability proportional to the weight
    of the band the ball was in just before it. spill_s None leaves stoppage gaps uncut.
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    graded = labels != "X"
    is_dead = graded & (labels == "D")
    is_live = graded & np.isin(labels, ["H", "A"])
    dead_rate, live_rate = targets(labels, missing, dead_ratio)
    want = {"dead": dead_rate * is_dead.sum(), "live": live_rate * is_live.sum()}
    pools = {"dead": np.flatnonzero(is_dead), "live": np.flatnonzero(is_live)}
    hidden = np.zeros(n, dtype=bool)
    already = np.zeros(n, dtype=bool) if already is None else np.asarray(already, bool)
    margin = int(MARGIN_S * fps)
    spill = None if spill_s is None else int(spill_s * fps)
    if weights is not None:
        weights = np.asarray(weights, float) / max(weights)
    min_len = int(MIN_GAP_S * fps)
    stop_start, stop_end = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for s, k in zip(*true_runs(is_dead)):
        stop_start[s:s + k], stop_end[s:s + k] = s, s + k
    misses = tries = 0
    while misses < 2000 and tries < 200000:
        tries += 1
        gone = hidden | already
        need = {k: want[k] - (gone & (is_dead if k == "dead" else is_live)).sum() for k in want}
        if max(need.values()) <= 0:
            break
        kind = max(need, key=lambda k: need[k] / max(want[k], 1))  # furthest below, relatively
        length = int(round(rng.choice(lengths) * fps))
        centre = int(rng.choice(pools[kind]))
        a, b = centre - length // 2, centre - length // 2 + length
        if kind == "dead" and spill is not None:  # cover the stoppage, spill at most SPILL_S
            a = max(a, stop_start[centre] - spill)
            b = min(b, stop_end[centre] + spill)
            if b - a < min_len:
                misses += 1
                continue
        ok = (a - margin >= 0 and b + margin <= n
              and (period[a:b] == period[centre]).all()
              and not hidden[a - margin:b + margin].any()
              and graded[a:b].all())
        if not ok:
            misses += 1
            continue
        if weights is not None:  # the ball leaving the shot near a line
            d = line_dist[a - 1]
            band = next((i for i, (lo, hi) in enumerate(bands) if lo <= d < hi), len(bands) - 1)
            if rng.random() > weights[band]:
                continue
        hidden[a:b] = True
    return hidden


def close_rows(players, pitch, near_m=1.5, chunk=20000):
    """Index labels of player rows whose nearest opponent in the same frame is within near_m."""
    length, width = pitch
    p = players[players["x"].notna()].sort_values(["frame", "team"])
    frame_ids, inv = np.unique(p["frame"].to_numpy(), return_inverse=True)
    out = []
    side = {}
    for t in ("H", "A"):
        q = p[p["team"] == t]
        rows = np.searchsorted(frame_ids, q["frame"].to_numpy())
        slot = q.groupby("frame").cumcount().to_numpy()
        k = int(slot.max()) + 1 if len(slot) else 1
        pos = np.full((len(frame_ids), k, 2), np.nan, dtype=np.float32)
        lab = np.full((len(frame_ids), k), -1, dtype=np.int64)
        pos[rows, slot, 0] = q["x"].to_numpy(float) * length
        pos[rows, slot, 1] = q["y"].to_numpy(float) * width
        lab[rows, slot] = q.index.to_numpy()
        side[t] = (pos, lab)
    (ph, lh), (pa, la) = side["H"], side["A"]
    for a in range(0, len(frame_ids), chunk):
        b = min(len(frame_ids), a + chunk)
        d = np.sqrt(((ph[a:b, :, None, :] - pa[a:b, None, :, :]) ** 2).sum(-1))
        with np.errstate(invalid="ignore"):
            near_h = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2) < near_m
            near_a = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) < near_m
        out.append(lh[a:b][near_h & (lh[a:b] >= 0)])
        out.append(la[a:b][near_a & (la[a:b] >= 0)])
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def hide_players(players, frames, pitch, fps, seed, hazard_per_s, lengths_s, near_m=1.5):
    """Players with cross-team merges simulated: a player next to an opponent vanishes.

    Every player-frame with an opponent within near_m starts a merge with probability
    hazard_per_s / fps; the player's track is then dropped for a length drawn from lengths_s,
    starting on the next frame and kept within the period. Overlapping merges of the same track
    are skipped.
    """
    rng = np.random.default_rng(seed)
    p = players
    cand = np.sort(close_rows(p, pitch, near_m))
    start = cand[rng.random(len(cand)) < hazard_per_s / fps]
    period = dict(zip(frames["frame"], frames["period"]))
    key = p["team"].astype(str) + "|" + p["track_id"].astype(str)
    by_track = {k: g for k, g in p.groupby(key).groups.items()}
    hidden = set()
    hidden_until = {}
    for r in start:
        f, k = p.at[r, "frame"], key.at[r]
        if hidden_until.get(k, -1) > f:
            continue
        n = int(round(rng.choice(lengths_s) * fps))
        rows = by_track[k]
        fr = p.loc[rows, "frame"]
        sel = rows[(fr > f).to_numpy() & (fr <= f + n).to_numpy()
                   & (fr.map(period) == period.get(f)).to_numpy()]
        hidden.update(sel.tolist())
        hidden_until[k] = f + n + 1
    return p.drop(index=list(hidden))


def drift(n, sigma, tau_s, fps, rng, period=None):
    """[n, 2] smooth stationary noise: std sigma per axis, white noise under a Gaussian of
    tau_s. Drawn afresh in each period, so nothing carries across halftime."""
    out = np.zeros((n, 2))
    if sigma <= 0 or n == 0:
        return out
    width = tau_s * fps
    # the filter's own gain on unit white noise, so the output std is sigma
    gain = np.sqrt(np.sum(gaussian_filter1d(np.eye(1, 8 * int(width) + 1, 4 * int(width))[0],
                                            width) ** 2))
    blocks = [(0, n)] if period is None else _blocks(np.asarray(period))
    for lo, hi in blocks:
        e = rng.normal(0, 1, (hi - lo, 2))
        out[lo:hi] = gaussian_filter1d(e, width, axis=0, mode="wrap") * sigma / gain
    return out


def _blocks(period):
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = np.r_[0, cuts, len(period)]
    return list(zip(edges[:-1], edges[1:]))


def add_ball_noise(frames, pitch, fps, seed, sigma_m, tau_s):
    """Copy of `frames` (sorted by frame) with drift() added to the visible ball, in metres."""
    length, width = pitch
    rng = np.random.default_rng([seed, 1])  # its own stream: the gap draws stay as they were
    e = drift(len(frames), sigma_m, tau_s, fps, rng, frames["period"].to_numpy())
    frames = frames.copy()
    frames["ball_x"] = frames["ball_x"].to_numpy(float) + e[:, 0] / length
    frames["ball_y"] = frames["ball_y"].to_numpy(float) + e[:, 1] / width
    return frames


def degrade(match, config=None):
    """Tracking of `match` with the ball hidden and its position blurred, as described above."""
    c = config or settings()
    frames = match.frames.sort_values("frame", ignore_index=True)
    seed = zlib.crc32(match.key.encode())
    length, width = match.pitch
    x, y = frames["ball_x"].to_numpy(float) * length, frames["ball_y"].to_numpy(float) * width
    line_dist = np.nan_to_num(np.minimum.reduce([x, length - x, y, width - y]), nan=1000.0)
    mask = hide_mask(frames["label"].to_numpy(), frames["period"].to_numpy(), match.fps, seed,
                     c["missing"], c["dead_ratio"], np.asarray(c["gap_lengths_s"]), line_dist,
                     c.get("line_bands_m"), c.get("line_weights_applied", c.get("line_weights")),
                     c.get("stoppage_spill_s", SPILL_S), already=np.isnan(x))
    frames = frames.copy()
    frames.loc[mask, ["ball_x", "ball_y"]] = np.nan
    if c.get("ball_noise_sigma_m"):
        frames = add_ball_noise(frames, match.pitch, match.fps, seed, c["ball_noise_sigma_m"],
                                c["ball_noise_tau_s"])
    players = match.players
    if c.get("merge_hazard_per_s"):
        players = hide_players(players, frames, match.pitch, match.fps, seed,
                               c["merge_hazard_per_s"], np.asarray(c["merge_lengths_s"]))
    return Tracking(frames, players, match.fps, match.pitch, match.home, match.away)
