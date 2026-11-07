"""Grid-search decoder constants on the train split, then report validation once.

    python -m scripts.tune                                   # the default grid
    python -m scripts.tune --turnover 0.25 0.5 1 --stop 0.5 1 --restart 0.5 1
    python -m scripts.tune --sources idsse skillcorner       # tune on clean idsse instead

Choose on train against the challenge's definition of possession (strict labels: idsse's short
uncontrolled spells folded back, see jaicodyn/data/labels.py), read validation once against both
strict and raw labels, never touch test (config/splits.json). Sources label possession
differently -- idsse's DFL flag flips far more often than SkillCorner's phases -- so the mean
spellF1 is shown per source as well as overall, and the choice is made on the mean of the source
means, so 14 SkillCorner games do not outvote 5 idsse ones.
"""

import argparse
import itertools
import json
from dataclasses import replace

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.labels import target_labels
from jaicodyn.data.match import is_built
from jaicodyn.decode import Params, decode
from jaicodyn.features import build
from scripts.decode import score


def games(split, sources):
    table = json.loads(paths.SPLITS.read_text("utf-8"))["splits"][split]
    return [load("{}/{}".format(s, i)) for s, ids in table.items() if s in sources
            for i in ids if is_built(s, i)]


def evaluate(matches, cache, params, scheme="strict"):
    """{source: [spellF1, ...]} and {source: [possession count ratio, ...]}."""
    f1, ratio = {}, {}
    for m in matches:
        feats, gt = cache[m.key]
        r = score(gt[scheme], decode(feats, m.fps, m.pitch, m.tags, params))
        f1.setdefault(m.source, []).append(r["spell_f1_mean"])
        ratio.setdefault(m.source, []).append(r["n_pred_spells"] / max(r["n_gt_spells"], 1))
    return f1, ratio


def summary(f1, ratio):
    by_source = {s: np.mean(v) for s, v in f1.items()}
    return np.mean(list(by_source.values())), by_source, {s: np.mean(v) for s, v in ratio.items()}


def line(label, f1, ratio):
    overall, by, rat = summary(f1, ratio)
    parts = "  ".join("{} {:.3f} (count x{:.2f})".format(s, by[s], rat[s]) for s in sorted(by))
    print("{:<36} spellF1 {:.3f}   {}".format(label, overall, parts))
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["idsse_degraded", "skillcorner"],
                        help="sources to tune on; clean idsse never loses the ball, so the "
                             "default uses its degraded copy")
    parser.add_argument("--turnover", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--stop", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--restart", type=float, nargs="+", default=[0.5, 1.0])
    args = parser.parse_args()

    train, val = games("train", args.sources), games("validation", args.sources)
    cache = {}
    for m in train + val:
        cache[m.key] = (build(m), {k: target_labels(m, k) for k in ("strict", "raw")})
    print("train {} games, validation {} games\n".format(len(train), len(val)))

    base = Params()
    results = []
    for t, s, r in itertools.product(args.turnover, args.stop, args.restart):
        p = replace(base, turnover_s=t, stop_s=s, restart_s=r)
        label = "turnover {:g}  stop {:g}  restart {:g}".format(t, s, r)
        results.append((line(label, *evaluate(train, cache, p, "strict")), p, label))

    best, p, label = max(results, key=lambda x: x[0])
    print("\nbest on train: {}  (spellF1 {:.3f})".format(label, best))
    print("\nvalidation, read once")
    for scheme in ("strict", "raw"):
        print("against {} labels".format(scheme))
        line("  current defaults", *evaluate(val, cache, base, scheme))
        line("  " + label, *evaluate(val, cache, p, scheme))


if __name__ == "__main__":
    main()
