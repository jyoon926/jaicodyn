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
              bands=None, weights=None, spill_s=SPILL_S):
    """Boolean mask of frames whose ball to hide.

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
        need = {k: want[k] - (hidden & (is_dead if k == "dead" else is_live)).sum() for k in want}
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


def degrade(match, config=None):
    """Tracking of `match` with the ball hidden as described above."""
    c = config or settings()
    frames = match.frames.sort_values("frame", ignore_index=True)
    seed = zlib.crc32(match.key.encode())
    length, width = match.pitch
    x, y = frames["ball_x"].to_numpy(float) * length, frames["ball_y"].to_numpy(float) * width
    line_dist = np.nan_to_num(np.minimum.reduce([x, length - x, y, width - y]), nan=1000.0)
    mask = hide_mask(frames["label"].to_numpy(), frames["period"].to_numpy(), match.fps, seed,
                     c["missing"], c["dead_ratio"], np.asarray(c["gap_lengths_s"]), line_dist,
                     c.get("line_bands_m"), c.get("line_weights_applied", c.get("line_weights")),
                     c.get("stoppage_spill_s", SPILL_S))
    frames = frames.copy()
    frames.loc[mask, ["ball_x", "ball_y"]] = np.nan
    return Tracking(frames, match.players, match.fps, match.pitch, match.home, match.away)
