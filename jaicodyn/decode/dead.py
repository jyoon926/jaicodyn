"""Dead-ball evidence per frame, in seconds of evidence like the rest of the decoder.

Positive means the ball is out of play. Each cue is a physical observation about a stoppage:

    off       ball beyond the lines, ignoring brief exits that come    +OFF_W
              straight back (sustained_off_pitch)
    still     ball visible and not moving for a second              +STILL_W
    moving    ball visible and moving at pass speed                 -MOVING_W
    slow      players walking together, relative to the match's own median speed:
              +SLOW_W at two thirds of it, -SLOW_W at one and a half times it
    gap       ball unseen for a long stretch: 0 up to GAP_FROM_S, +GAP_W from GAP_FULL_S
    tag       automatic smart tagging marks a set piece starting: +TAG_W from TAG_PRE_S before
              its start to TAG_POST_S after. Tags are Metrica's automatic output, not human data.
    bias      -BIAS_W: a lean towards live when no cue fires. Tuned to 0: the cues alone are
              balanced, and a lean either way costs spellF1 (scripts/robustness.py)

Speed thresholds are relative to the match so they carry across providers whose speed scales
differ. Set from what each cue means, then STILL_W and BIAS_W tuned on the idsse train games
degraded like the competition tracking, at missing-ball ratios 1.75x, 2.8x and 5x, strict labels,
and confirmed once on the validation games (+0.023 to +0.044 spellF1). Never fitted to the
provided game's labels (RULES.md).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.ball import true_runs


@dataclass(frozen=True)
class DeadParams:
    off_w: float = 2.0
    off_margin_m: float = 1.0
    off_min_s: float = 1.0       # an exit this short that comes straight back is tracker noise
    still_w: float = 2.0         # tuned: x2 won 5/5 train games at every ratio, confirmed on validation
    still_mps: float = 0.3
    moving_w: float = 0.5
    moving_mps: float = 3.0
    slow_w: float = 1.0
    slow_ratio: float = 3.0      # a factor of this below/above the match median is one unit.
                                 # Tuned on degraded idsse (was 1.5): +0.018-0.020, 5/5 games at
                                 # every ratio; it weakens the cue both ways, so the decoder calls
                                 # somewhat more dead ball (49% -> 52%, true 46%) and fewer, cleaner
                                 # possessions
    gap_w: float = 0.5
    gap_from_s: float = 10.0
    gap_full_s: float = 30.0
    tag_w: float = 2.0
    tag_pre_s: float = 3.0
    tag_post_s: float = 2.0
    bias_w: float = 0.0          # tuned: flat from -0.125 to +0.25 with still_w 2; -0.5 clearly worse
    smooth_s: float = 1.0


def _rolling(x, n):
    return pd.Series(x).rolling(max(int(n), 1), center=True, min_periods=1).mean().to_numpy()


def off_pitch(feats, pitch, margin):
    length, width = pitch
    x, y = feats["ball_x"].to_numpy(float), feats["ball_y"].to_numpy(float)
    return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def gap_seconds(feats, fps):
    """Length in seconds of the ball gap each frame sits in; 0 where the ball is known."""
    out = np.zeros(len(feats))
    missing = feats["ball_x"].isna().to_numpy()
    period = feats["period"].to_numpy()
    for start, n in zip(*true_runs(missing)):
        # a gap never spans halftime: split at period changes
        seg = period[start:start + n]
        cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, n]):
            out[start + a:start + b] = (b - a) / fps
    return out


def tag_mask(feats, tags, fps, pre_s, post_s):
    """Frames around the start of each automatic set-piece tag."""
    mask = np.zeros(len(feats), dtype=bool)
    if tags is None or not len(tags):
        return mask
    frame = feats["frame"].to_numpy()
    starts = tags.loc[tags["code"].str.upper() == "SET PIECES", "start"].to_numpy()
    for s in starts:
        a, b = np.searchsorted(frame, [s - pre_s * fps, s + post_s * fps])
        mask[a:b] = True
    return mask


def sustained_off_pitch(feats, pitch, fps, p=DeadParams()):
    """Off the pitch, ignoring brief exits that come straight back.

    A run of off-pitch frames no longer than off_min_s that is followed directly by the ball seen
    back on the pitch is treated as tracker noise near the line: on the competition match 19 of 56
    exits are like that, and on the clean idsse train games every such exit (5 of 5) is live play,
    while every longer one (70) is dead. Exits that last longer, or end with the ball out of sight,
    still count.
    """
    off = off_pitch(feats, pitch, p.off_margin_m)
    x = feats["ball_x"].to_numpy(float)
    limit = p.off_min_s * fps
    for start, n in zip(*true_runs(off)):
        end = start + n
        comes_back = end < len(x) and not np.isnan(x[end]) and not off[end]
        if n <= limit and comes_back:
            off[start:end] = False
    return off


def cues(feats, fps, pitch, tags=None, p=DeadParams()):
    """Table of each cue's evidence per second, and their sum in `dead`."""
    n_smooth = p.smooth_s * fps
    xy = feats[["ball_x", "ball_y"]].to_numpy(float)
    step = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps]
    visible = ~np.isnan(xy[:, 0])
    ball_speed = pd.Series(step).rolling(int(n_smooth), center=True, min_periods=int(n_smooth) // 2
                                         ).median().to_numpy()
    seen = visible & ~np.isnan(ball_speed)

    off = sustained_off_pitch(feats, pitch, fps, p)
    speed = feats["speed_med"].to_numpy(float)
    ratio = _rolling(speed, n_smooth) / np.nanmedian(speed)
    with np.errstate(divide="ignore", invalid="ignore"):  # everyone still: ratio 0, clipped
        slow = np.clip(-np.log(ratio) / np.log(p.slow_ratio), -1, 2)
    gap = np.clip((gap_seconds(feats, fps) - p.gap_from_s) / (p.gap_full_s - p.gap_from_s), 0, 1)

    out = pd.DataFrame({
        "off": p.off_w * off,
        "still": p.still_w * (seen & (ball_speed < p.still_mps) & ~off),
        "moving": -p.moving_w * (seen & (ball_speed > p.moving_mps) & ~off),
        "slow": p.slow_w * np.nan_to_num(slow),
        "gap": p.gap_w * gap,
        "tag": p.tag_w * tag_mask(feats, tags, fps, p.tag_pre_s, p.tag_post_s),
    })
    out["dead"] = out.sum(axis=1) - p.bias_w
    return out
