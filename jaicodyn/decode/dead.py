"""Dead-ball evidence per frame, in seconds of evidence like the rest of the decoder.

Positive means the ball is out of play. Each cue is a physical observation about a stoppage:

    off       ball beyond the lines                                 +OFF_W
    still     ball visible and not moving for a second              +STILL_W
    moving    ball visible and moving at pass speed                 -MOVING_W
    slow      players walking together, relative to the match's own median speed:
              +SLOW_W at two thirds of it, -SLOW_W at one and a half times it
    gap       ball unseen for a long stretch: 0 up to GAP_FROM_S, +GAP_W from GAP_FULL_S
    tag       automatic smart tagging marks a set piece starting: +TAG_W from TAG_PRE_S before
              its start to TAG_POST_S after. Tags are Metrica's automatic output, not human data.
    bias      -BIAS_W: the ball is in play most of the time, so with no cue the answer is live

Speed thresholds are relative to the match so they carry across providers whose speed scales
differ. Set from what each cue means, not fitted to the provided game's labels (RULES.md); tune
on other labelled games.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.ball import true_runs


@dataclass(frozen=True)
class DeadParams:
    off_w: float = 2.0
    off_margin_m: float = 1.0
    still_w: float = 1.0
    still_mps: float = 0.3
    moving_w: float = 0.5
    moving_mps: float = 3.0
    slow_w: float = 1.0
    slow_ratio: float = 1.5      # a factor of this below/above the match median is one unit
    gap_w: float = 0.5
    gap_from_s: float = 10.0
    gap_full_s: float = 30.0
    tag_w: float = 2.0
    tag_pre_s: float = 3.0
    tag_post_s: float = 2.0
    bias_w: float = 0.5
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


def cues(feats, fps, pitch, tags=None, p=DeadParams()):
    """Table of each cue's evidence per second, and their sum in `dead`."""
    n_smooth = p.smooth_s * fps
    xy = feats[["ball_x", "ball_y"]].to_numpy(float)
    step = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps]
    visible = ~np.isnan(xy[:, 0])
    ball_speed = pd.Series(step).rolling(int(n_smooth), center=True, min_periods=int(n_smooth) // 2
                                         ).median().to_numpy()
    seen = visible & ~np.isnan(ball_speed)

    off = off_pitch(feats, pitch, p.off_margin_m)
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
