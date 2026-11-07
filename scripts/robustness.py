"""Do the decoder constants hold up if the competition's missing-ball pattern is not what we measured?

    python -m scripts.robustness                       # ratios 1.75, 2.8, 5
    python -m scripts.robustness --ratios 1.75 2.8 5 8

The dead/live missing ratio behind idsse_degraded (2.8x) is an estimate; the automatic tags only
bound it from below (1.75x). So the idsse train games are degraded at several ratios, in memory,
everything else in config/degrade.json unchanged, and each constant is swept one at a time:

    turnover_s           the turnover cost
    dead.<weight>        each dead-ball cue weight at x0, x0.5, x1, x2

Scored on strict labels (the challenge's controlled possession). For each setting: spellF1 per
ratio, and regret -- how far below the best setting at that ratio. A constant holds up if the
default's worst regret is small, i.e. no other value is clearly better at any plausible ratio.
Train split only. SkillCorner is left out: its TV feed shows mostly players near the ball and
cuts to replays during stoppages, so its player-speed and long-gap cues behave unlike the
competition's wide shot.
"""

import argparse
import json
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.degrade import degrade, settings
from jaicodyn.data.labels import target_labels
from jaicodyn.decode import Params, decode
from jaicodyn.decode.dead import DeadParams
from jaicodyn.features.team import team_features
from scripts.decode import score

DEAD_WEIGHTS = ["off_w", "still_w", "moving_w", "slow_w", "gap_w", "bias_w"]  # no tags in idsse
SCALES = [0.0, 0.5, 1.0, 2.0]


def degraded_features(matches, config, ratio):
    """{key: (features, strict labels)} for each match degraded at `ratio`."""
    out = {}
    for m in matches:
        t = degrade(m, dict(config, dead_ratio=ratio))
        view = SimpleNamespace(frames=t.frames, players=t.players, fps=m.fps, pitch=m.pitch)
        out[m.key] = (team_features(view), target_labels(m, "strict"))
    return out


def mean_f1(matches, cache, params):
    return float(np.mean([score(cache[m.key][1], decode(cache[m.key][0], m.fps, m.pitch, None,
                                                        params))["spell_f1_mean"]
                          for m in matches]))


def sweep(name, settings_, matches, caches, ratios, default_label):
    """Print spellF1 per ratio and regret for each (label, params); return the default's regret."""
    scores = {label: [mean_f1(matches, caches[r], p) for r in ratios] for label, p in settings_}
    best = np.max(np.array(list(scores.values())), axis=0)
    print("\n{}".format(name))
    print("  {:<16}".format("") + "".join("  {:>6}x".format(r) for r in ratios) + "   worst regret")
    for label, row in scores.items():
        regret = best - np.array(row)
        mark = "  <- default" if label == default_label else ""
        print("  {:<16}".format(label) + "".join("  {:7.3f}".format(v) for v in row)
              + "   {:6.3f}{}".format(regret.max(), mark))
    return float((best - np.array(scores[default_label])).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.75, 2.8, 5.0])
    parser.add_argument("--turnover", type=float, nargs="+", default=[0.25, 0.375, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse"]
    matches = [load("idsse/{}".format(i)) for i in train]
    config = settings()
    caches = {r: degraded_features(matches, config, r) for r in args.ratios}
    base = Params()
    print("idsse train, {} games, degraded at ratios {}; strict labels".format(
        len(matches), args.ratios))

    regrets = {}
    turnover = [("{:g}".format(t), replace(base, turnover_s=t)) for t in args.turnover]
    regrets["turnover_s"] = sweep("turnover_s", turnover, matches, caches, args.ratios,
                                  "{:g}".format(base.turnover_s))
    defaults = {f.name: getattr(base.dead, f.name) for f in fields(DeadParams)}
    for w in DEAD_WEIGHTS:
        options = [("x{:g} = {:g}".format(s, defaults[w] * s),
                    replace(base, dead=replace(base.dead, **{w: defaults[w] * s}))) for s in SCALES]
        regrets["dead." + w] = sweep("dead." + w, options, matches, caches, args.ratios,
                                     "x1 = {:g}".format(defaults[w]))

    print("\ndefault's worst regret per constant (spellF1 below the best value at some ratio)")
    for k, v in sorted(regrets.items(), key=lambda kv: -kv[1]):
        print("  {:<16} {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
