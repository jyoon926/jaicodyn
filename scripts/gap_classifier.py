"""Check the gap classifier: does it recover what the ball would have shown?

    python -m scripts.gap_classifier BAR-ATH

1. Decode the match with the ball visible: these confident labels are the reference.
2. Hide the ball on a fresh layout of simulated 5-30 s test gaps. Everything downstream -- the
   pseudo-labels the classifier trains on included -- sees only the masked match, so nothing
   under a test gap leaks into training.
3. On the test-gap frames, compare with the reference:
       hold          the label just before the gap, carried through
       decoder       the base decoder on the masked match (holds state, dead-ball cues)
       classifier    the gap classifier's most likely state alone
       decoder+gaps  the decoder with the classifier's probabilities plugged in

Pseudo-labels measure how well the visible-ball decoder is recovered, not the truth. With true
labels from other games the same comparison measures accuracy.
"""

import argparse
from dataclasses import replace

import numpy as np

from jaicodyn.data import load
from jaicodyn.decode import Params, decode, posterior
from jaicodyn.decode.gaps import (CLASSES, GapParams, gap_probs, gap_runs, mask_ball, periods,
                                  simulate)
from jaicodyn.features import build
from scripts.build_features import resolve

EVAL_SEED = 12345  # a layout the training rounds (seed 0) never use
CONF = 0.9


def pseudo(feats, match):
    labels = decode(feats, match.fps, match.pitch, match.tags)
    conf = posterior(feats, match.fps, match.pitch, match.tags).max(axis=1)
    return labels, conf


def report(name, pred, ref, rows):
    acc = np.mean(pred[rows] == ref[rows])
    recall = ["{} {:.0%}".format(c, np.mean(pred[rows][ref[rows] == c] == c))
              if np.any(ref[rows] == c) else "{} -".format(c) for c in CLASSES]
    print("  {:<14} agreement {:5.1%}   recall {}".format(name, acc, "  ".join(recall)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", help="match id or source/id")
    parser.add_argument("--gap-w", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                        help="classifier weights in the decoder to compare")
    args = parser.parse_args()
    match = load(resolve(args.match))
    feats = build(match)
    fps, p = match.fps, GapParams()

    ref, ref_conf = pseudo(feats, match)
    missing = feats["ball_x"].isna().to_numpy()
    blocks = periods(feats)
    test = simulate(missing, blocks, fps, np.random.default_rng(EVAL_SEED), p)
    masked = mask_ball(feats, test)
    rows = test & (ref_conf >= CONF)
    runs = gap_runs(test, blocks, int(p.min_gap_s * fps))
    print("{}: {} test gaps, {:.0f} s hidden, {} confident reference frames ({})".format(
        match.key, len(runs), test.sum() / fps, rows.sum(),
        ", ".join("{} {:.0%}".format(c, np.mean(ref[rows] == c)) for c in CLASSES)))

    labels_m, conf_m = pseudo(masked, match)
    probs = gap_probs(masked, fps, match.pitch, match.tags, labels_m, conf_m, p=p)
    hold = ref.copy()
    for a, b in runs:
        hold[a:b] = labels_m[a - 1]
    cls = CLASSES[np.nanargmax(np.nan_to_num(probs, nan=-1), axis=1)]

    print("\nrecovering the visible-ball labels on test-gap frames")
    report("hold", hold, ref, rows)
    report("decoder", labels_m, ref, rows)
    report("classifier", cls, ref, rows)
    for gw in args.gap_w:
        with_gaps = decode(masked, fps, match.pitch, match.tags, replace(Params(), gap_w=gw), probs)
        report("+gaps w={:g}".format(gw), with_gaps, ref, rows)

    print("\nwhole match, real gaps")
    base = decode(feats, fps, match.pitch, match.tags)
    real = gap_probs(feats, fps, match.pitch, match.tags, ref, ref_conf, p=p)
    full = decode(feats, fps, match.pitch, match.tags, gap_probs=real)
    for name, lab in (("decoder", base), ("decoder+gaps", full)):
        n = np.sum(np.isin(lab[1:], ["H", "A"]) & (lab[1:] != lab[:-1])) + int(lab[0] in "HA")
        print("  {:<14} possession starts {:>4}   dead {:5.1%}   in gaps: H {:.0%} A {:.0%} D {:.0%}"
              .format(name, n, np.mean(lab == "D"),
                      *(np.mean(lab[missing] == c) for c in CLASSES)))


if __name__ == "__main__":
    main()
