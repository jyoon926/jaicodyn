"""Grid-search decoder constants on the train split; report validation only when asked.

    python -m scripts.tune                                   # the default grid, train only
    python -m scripts.tune --validate                        # ... and read validation once
    python -m scripts.tune --turnover 0.25 0.5 1 --stop 0.5 1 --restart 0.5 1
    python -m scripts.tune --sources idsse skillcorner       # tune on clean idsse instead
    python -m scripts.tune --grid closer_m=0.5,1,2 live_w=0,0.25   # any Params / dead.* fields

Choose on train against the challenge's definition of possession (strict labels: idsse's short
uncontrolled spells folded back, see jaicodyn/data/labels.py). Validation is read only with
--validate, against both strict and raw labels: every read makes it less independent, so read it
once, for a frozen choice. Never touch test (config/splits.json). Sources label possession
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


def with_field(p, field, value):
    if field.startswith("dead."):
        return replace(p, dead=replace(p.dead, **{field[5:]: value}))
    return replace(p, **{field: value})


def settings(base, args):
    """[(Params, label)] to try: the --grid product, else the switch-cost grid."""
    if not args.grid:
        return [(replace(base, turnover_s=t, stop_s=s, restart_s=r),
                 "turnover {:g}  stop {:g}  restart {:g}".format(t, s, r))
                for t, s, r in itertools.product(args.turnover, args.stop, args.restart)]
    fields = [g.split("=")[0] for g in args.grid]
    values = [[float(v) for v in g.split("=")[1].split(",")] for g in args.grid]
    out = []
    for combo in itertools.product(*values):
        p = base
        for f, v in zip(fields, combo):
            p = with_field(p, f, v)
        if p.gate_far_m <= p.gate_near_m:
            continue
        out.append((p, "  ".join("{} {:g}".format(f, v) for f, v in zip(fields, combo))))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["idsse_degraded", "skillcorner"],
                        help="sources to tune on; clean idsse never loses the ball, so the "
                             "default uses its degraded copy")
    parser.add_argument("--turnover", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--stop", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--restart", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--grid", nargs="+", metavar="FIELD=V1,V2",
                        help="sweep these Params fields (dead.<field> for DeadParams) instead of "
                             "the switch costs; combinations with gate_far_m <= gate_near_m are "
                             "skipped")
    parser.add_argument("--validate", action="store_true",
                        help="also read validation for the best setting (read it once)")
    args = parser.parse_args()

    train = games("train", args.sources)
    val = games("validation", args.sources) if args.validate else []
    cache = {}
    for m in train + val:
        cache[m.key] = (build(m), {k: target_labels(m, k) for k in ("strict", "raw")})
    print("train {} games{}\n".format(
        len(train), ", validation {} games".format(len(val)) if args.validate else ""))

    base = Params()
    results = []
    for p, label in settings(base, args):
        results.append((line(label, *evaluate(train, cache, p, "strict")), p, label))
    print("\ncurrent defaults on train: spellF1 {:.3f}".format(
        summary(*evaluate(train, cache, base, "strict"))[0]))

    best, p, label = max(results, key=lambda x: x[0])
    print("\nbest on train: {}  (spellF1 {:.3f})".format(label, best))
    if not args.validate:
        print("validation not read (pass --validate to read it, once, for a frozen choice)")
        return
    print("\nvalidation, read once")
    for scheme in ("strict", "raw"):
        print("against {} labels".format(scheme))
        line("  current defaults", *evaluate(val, cache, base, scheme))
        line("  " + label, *evaluate(val, cache, p, scheme))


if __name__ == "__main__":
    main()
