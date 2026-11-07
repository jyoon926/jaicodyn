"""Make a clean labelled game look like the competition tracking: hide the ball in long gaps.

idsse tracks the ball on every frame; the competition match loses it on about 41% of frames,
almost all in gaps over 5 s, and far more often while the ball is dead. Tuning on clean data fits
weights to conditions the test match does not have, so this hides the ball to match:

    missing        share of graded frames hidden: the competition match's, from its tracking alone
    gap_lengths_s  gap lengths, drawn from the competition match's own gaps of 5 s or more
    dead_ratio     how many times likelier a dead frame is hidden than a live one, measured on the
                   SkillCorner train games, which have real gaps and labels

All three are measured by `python -m scripts.degrade --measure` into config/degrade.json.

No competition labels are used. Each gap is centred on a frame drawn from the dead or the live
frames, whichever is further below its target. A gap over a stoppage is cut to the stoppage plus
at most SPILL_S of play on each side, as when the camera loses the ball just before it goes out
and finds it just after the restart; without the cut, long gaps spill far into play and the
dead/live ratio comes out at 3-4x instead of the measured one. The seed comes from the match id,
so the output is the same every run.
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


def hide_mask(labels, period, fps, seed, missing, dead_ratio, lengths):
    """Boolean mask of frames whose ball to hide."""
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
    spill = int(SPILL_S * fps)
    min_len = int(MIN_GAP_S * fps)
    stop_start, stop_end = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for s, k in zip(*true_runs(is_dead)):
        stop_start[s:s + k], stop_end[s:s + k] = s, s + k
    misses = 0
    while misses < 2000:
        need = {k: want[k] - (hidden & (is_dead if k == "dead" else is_live)).sum() for k in want}
        if max(need.values()) <= 0:
            break
        kind = max(need, key=lambda k: need[k] / max(want[k], 1))  # furthest below, relatively
        length = int(round(rng.choice(lengths) * fps))
        centre = int(rng.choice(pools[kind]))
        a, b = centre - length // 2, centre - length // 2 + length
        if kind == "dead":  # a stoppage gap covers the stoppage, spilling at most SPILL_S into play
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
        hidden[a:b] = True
    return hidden


def degrade(match, config=None):
    """Tracking of `match` with the ball hidden as described above."""
    c = config or settings()
    frames = match.frames.sort_values("frame", ignore_index=True)
    seed = zlib.crc32(match.key.encode())
    mask = hide_mask(frames["label"].to_numpy(), frames["period"].to_numpy(), match.fps, seed,
                     c["missing"], c["dead_ratio"], np.asarray(c["gap_lengths_s"]))
    frames = frames.copy()
    frames.loc[mask, ["ball_x", "ball_y"]] = np.nan
    return Tracking(frames, match.players, match.fps, match.pitch, match.home, match.away)
