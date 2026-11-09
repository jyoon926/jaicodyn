"""Oracle test: which possession (team) error is worth fixing? Fix each kind perfectly, one at a time.

    python -m scripts.oracle_team                       # degraded idsse and SkillCorner, train
    python -m scripts.oracle_team --tol 3

The decoder's dead ball is kept exactly; the true labels only correct who has the ball, so each
line isolates one kind of team error. A turnover is a change H -> A or A -> H between live frames.
Predicted and true turnovers are matched one to one, same direction, nearest first, within TOL_S.

    snap turnovers    matched predicted turnovers move to the true frame
    delete false      unmatched predicted turnovers are removed: the previous team keeps the ball
    insert missed     unmatched true turnovers are added at the true frame
    restart team      after each predicted stoppage (and at kickoff), a spell given to the wrong
                      team is given to the true one, up to the first true turnover inside it
    all four          the four together, in that order
    perfect teams     the true team wherever both say live: the ceiling for any team fix

Scored with strict spellF1. Train split only: an oracle uses labels.
"""

import argparse
import json

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.labels import target_labels
from jaicodyn.data.match import is_built
from jaicodyn.decode import decode
from jaicodyn.features import build
from scripts.decode import score
from scripts.oracle_dead import per_period

FIXES = ["snap turnovers", "delete false", "insert missed", "restart team", "all four",
         "perfect teams"]
LIVE = ("H", "A")


def turnovers(labels):
    """[(frame, to_team)] where a live label changes team on consecutive live frames."""
    i = np.flatnonzero(np.isin(labels[1:], LIVE) & np.isin(labels[:-1], LIVE)
                       & (labels[1:] != labels[:-1])) + 1
    return [(int(k), labels[k]) for k in i]


def match(pred_t, true_t, tol):
    """(pairs [(p, t)], unmatched predicted frames, unmatched true frames)."""
    cands = sorted((abs(p - t), p, t) for p, pt in pred_t for t, tt in true_t
                   if pt == tt and abs(p - t) <= tol)
    used_p, used_t, pairs = set(), set(), []
    for _, p, t in cands:
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
            pairs.append((p, t))
    return (pairs, [p for p, _ in pred_t if p not in used_p],
            [(t, tt) for t, tt in true_t if t not in used_t])


def spell_end(labels, start):
    """End (exclusive) of the run of labels[start] beginning at start."""
    k = start
    while k < len(labels) and labels[k] == labels[start]:
        k += 1
    return k


def spell_start(labels, end):
    """Start of the run of labels[end - 1] ending at end."""
    k = end - 1
    while k > 0 and labels[k - 1] == labels[end - 1]:
        k -= 1
    return k


def snap(pred, truth, tol):
    out = pred.copy()
    pairs, _, _ = match(turnovers(pred), turnovers(truth), tol)
    for p, t in sorted(pairs):
        old, new = out[p - 1], out[p]
        if old not in LIVE or new not in LIVE:
            continue  # an earlier edit already changed this boundary
        if t < p:   # the change came earlier: the new team takes [t, p), within the old spell
            a = max(t, spell_start(out, p))
            out[a:p] = new
        elif t > p:  # it came later: the old team keeps [p, t), within the new spell
            b = min(t, spell_end(out, p))
            out[p:b] = old
    return out


def delete_false(pred, truth, tol):
    out = pred.copy()
    _, false, _ = match(turnovers(pred), turnovers(truth), tol)
    for p in sorted(false):
        if out[p - 1] in LIVE and out[p] in LIVE and out[p] != out[p - 1]:
            out[p:spell_end(out, p)] = out[p - 1]
    return out


def insert_missed(pred, truth, tol):
    out = pred.copy()
    _, _, missed = match(turnovers(pred), turnovers(truth), tol)
    for t, team in sorted(missed):
        if out[t] in LIVE and out[t] != team:
            out[t:spell_end(out, t)] = team
    return out


def restart_team(pred, truth, tol=None):
    """Spells that start the period or follow a predicted stoppage get the true team."""
    out = pred.copy()
    starts = [0] + [int(k) for k in np.flatnonzero((pred[:-1] == "D") & np.isin(pred[1:], LIVE)) + 1]
    for s in starts:
        if out[s] not in LIVE:
            continue
        e = spell_end(out, s)
        live = [k for k in range(s, e) if truth[k] in LIVE]
        if not live or truth[live[0]] == out[s]:
            continue
        team, k = truth[live[0]], live[0]
        while k < e and truth[k] in (team, "D"):  # only up to the first true turnover
            k += 1
        out[s:k] = team
    return out


def perfect_teams(pred, truth, tol=None):
    out = pred.copy()
    both = np.isin(pred, LIVE) & np.isin(truth, LIVE)
    out[both] = truth[both]
    return out


def apply(fix, pred, truth, tol):
    if fix == "snap turnovers":
        return snap(pred, truth, tol)
    if fix == "delete false":
        return delete_false(pred, truth, tol)
    if fix == "insert missed":
        return insert_missed(pred, truth, tol)
    if fix == "restart team":
        return restart_team(pred, truth)
    if fix == "all four":
        out = snap(pred, truth, tol)
        out = delete_false(out, truth, tol)
        out = insert_missed(out, truth, tol)
        return restart_team(out, truth)
    return perfect_teams(pred, truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["idsse_degraded", "skillcorner"])
    parser.add_argument("--tol", type=float, default=5.0, help="matching tolerance, seconds")
    args = parser.parse_args()
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]

    for source in args.sources:
        base, gains, counts = [], {f: [] for f in FIXES}, []
        for match_id in train.get(source, []):
            if not is_built(source, match_id):
                continue
            m = load("{}/{}".format(source, match_id))
            feats = build(m)
            truth = target_labels(m, "strict")
            pred = decode(feats, m.fps, m.pitch, m.tags)
            period = feats["period"].to_numpy()
            tol = int(round(args.tol * m.fps))
            pairs, false, missed = [], [], []
            cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(pred)]):
                pr, fa, mi = match(turnovers(pred[a:b]), turnovers(truth[a:b]), tol)
                pairs += pr
                false += fa
                missed += mi
            offsets = [abs(p - t) / m.fps for p, t in pairs]
            counts.append((len(pairs), len(false), len(missed),
                           np.median(offsets) if offsets else np.nan))
            b0 = score(truth, pred)["spell_f1_mean"]
            base.append(b0)
            for f in FIXES:
                fixed = per_period(lambda p, t: apply(f, p, t, tol), pred, truth, period)
                gains[f].append(score(truth, fixed)["spell_f1_mean"] - b0)
        n = len(base)
        c = np.array(counts)
        print("{} train, {} games: decoder spellF1 {:.3f}".format(source, n, np.mean(base)))
        print("  turnovers per game: {:.0f} matched (median {:.1f}s off), {:.0f} false, "
              "{:.0f} missed".format(c[:, 0].mean(), np.nanmedian(c[:, 3]), c[:, 1].mean(),
                                    c[:, 2].mean()))
        for f in FIXES:
            g = np.array(gains[f])
            print("  {:<15} {:+.3f}   (better in {}/{} games, range {:+.3f} to {:+.3f})".format(
                f, g.mean(), int(np.sum(g > 1e-9)), n, g.min(), g.max()))
        print()


if __name__ == "__main__":
    main()
