"""Oracle test: which dead-ball error is worth fixing? Fix each kind perfectly, one at a time.

    python -m scripts.oracle_dead                       # degraded idsse and SkillCorner, train
    python -m scripts.oracle_dead --sources idsse_degraded

The true labels decide only where dead ball is; who has the ball stays the decoder's call, so
each line isolates one kind of dead-ball error:

    snap ends       every predicted stoppage that overlaps a true one ends when the true one ends:
                    frames it ran on into play take the team the decoder gives after it; if it
                    ended early, it is extended as dead
    delete false    predicted stoppages that overlap no true stoppage become live, possession
                    carrying on with the decoder's team before them
    insert missed   true stoppages that no predicted stoppage overlaps become dead
    all three       the three together
    perfect dead    dead exactly where true, teams still the decoder's: the ceiling for any
                    dead-ball fix

Scored with strict spellF1 (the challenge's controlled possession). Train split only: an oracle
uses labels, so it must never touch validation or test.
"""

import argparse
import json

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.labels import target_labels
from jaicodyn.data.match import is_built
from jaicodyn.decode import decode
from jaicodyn.features import build
from jaicodyn.features.ball import true_runs
from scripts.decode import score

FIXES = ["snap ends", "delete false", "insert missed", "all three", "perfect dead"]


def fill_live(pred, rows):
    """Give `rows` (currently D) the decoder's team: the last live label before, else the next."""
    out = pred.copy()
    for i in rows:
        j = i - 1
        while j >= 0 and out[j] not in ("H", "A"):
            j -= 1
        if j < 0:
            k = i + 1
            while k < len(out) and out[k] not in ("H", "A"):
                k += 1
            out[i] = out[k] if k < len(out) else "H"
        else:
            out[i] = out[j]
    return out


def team_after(pred, stop):
    """The decoder's first team label at or after row `stop`."""
    k = stop
    while k < len(pred) and pred[k] not in ("H", "A"):
        k += 1
    return pred[k] if k < len(pred) else None


def snap_ends(pred, truth):
    out = pred.copy()
    t = truth == "D"
    for s, n in zip(*true_runs(pred == "D")):
        e = s + n
        overlap = np.flatnonzero(t[s:e])
        if not len(overlap):
            continue
        last = s + overlap[-1]                    # last true-dead frame inside the prediction
        true_end = last
        while true_end < len(t) and t[true_end]:  # the true stoppage may run on past it
            true_end += 1
        if true_end < e:                          # ran on into play: hand back to the restart team
            team = team_after(pred, e)
            out[true_end:e] = team or out[true_end - 1]
        else:                                     # ended early: extend as dead
            out[e:true_end] = "D"
    return out


def delete_false(pred, truth):
    t = truth == "D"
    rows = [i for s, n in zip(*true_runs(pred == "D")) if not t[s:s + n].any()
            for i in range(s, s + n)]
    return fill_live(pred, rows)


def insert_missed(pred, truth):
    out = pred.copy()
    p = pred == "D"
    for s, n in zip(*true_runs(truth == "D")):
        if not p[s:s + n].any():
            out[s:s + n] = "D"
    return out


def perfect_dead(pred, truth):
    out = pred.copy()
    out[truth == "D"] = "D"
    wrong = np.flatnonzero((out == "D") & (truth != "D"))
    return fill_live(out, wrong)


def apply(fix, pred, truth):
    if fix == "snap ends":
        return snap_ends(pred, truth)
    if fix == "delete false":
        return delete_false(pred, truth)
    if fix == "insert missed":
        return insert_missed(pred, truth)
    if fix == "all three":
        return insert_missed(delete_false(snap_ends(pred, truth), truth), truth)
    return perfect_dead(pred, truth)


def per_period(fn, pred, truth, period):
    """Apply fn within each period, so no fix reaches across halftime."""
    out = pred.copy()
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(pred)]):
        out[a:b] = fn(pred[a:b], truth[a:b])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["idsse_degraded", "skillcorner"])
    args = parser.parse_args()
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]

    for source in args.sources:
        base, gains = [], {f: [] for f in FIXES}
        for match_id in train.get(source, []):
            if not is_built(source, match_id):
                continue
            m = load("{}/{}".format(source, match_id))
            feats = build(m)
            truth = target_labels(m, "strict")
            pred = decode(feats, m.fps, m.pitch, m.tags)
            period = feats["period"].to_numpy()
            b = score(truth, pred)["spell_f1_mean"]
            base.append(b)
            for f in FIXES:
                fixed = per_period(lambda p, t: apply(f, p, t), pred, truth, period)
                gains[f].append(score(truth, fixed)["spell_f1_mean"] - b)
        n = len(base)
        print("{} train, {} games: decoder spellF1 {:.3f}".format(source, n, np.mean(base)))
        for f in FIXES:
            g = np.array(gains[f])
            print("  {:<14} {:+.3f}   (better in {}/{} games, range {:+.3f} to {:+.3f})".format(
                f, g.mean(), int(np.sum(g > 1e-9)), n, g.min(), g.max()))
        print()


if __name__ == "__main__":
    main()
