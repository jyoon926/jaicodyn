"""Does phase-tag evidence help? Test it on labelled games with synthetic tags.

    python -m scripts.synthetic_tags                # sweep on the degraded idsse train games
    python -m scripts.synthetic_tags --validate     # also read validation once for the choice

Tags are made per game by jaicodyn/decode/synthetic_tags.py from each code's definition -- nothing
copied from the competition's real tags. Two kinds of settings:

    realism (unknowable; the answer must hold across them)
        length      fixed 10 or 20 s from the trigger, or to the possession's end + 0, 5 or 10 s
        accuracy    0.6, 0.75 or 0.9 on frames inside a tag with no ball and live play
        r           0, 0.3 or 0.6 of tags copying the decoder (errors shared with it)
    decision (what we would ship)
        sources     all phase codes / without the two box codes / build up and progression only
        scale       0.5, 1 or 2 times the evidence strength

Each feasible realism combination is one world. Each decision is scored in every world: the gain
in strict spellF1 over the decoder without tags, averaged over the games. The chosen decision is
the one with the best worst-case gain over the worlds (robust, not best in one world). The shipped
decision is SHIPPED (all codes, scale 0.5; see PHASE_SCALE in jaicodyn/decode/tags.py), which trades
a slightly worse worst case for a larger mean gain.

Validation bar (fixed before reading): in the validation games, the shipped decision's mean gain
over the worlds is at least 0, and its worst world loses no more than 0.01.
"""

import argparse
import json

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.labels import target_labels
from jaicodyn.decode import decode
from jaicodyn.decode import synthetic_tags as st
from jaicodyn.decode.tags import PHASE_SCALE, PHASES, phase_team_evidence
from jaicodyn.features import build
from scripts.decode import score

LENGTHS = [("fixed", 10), ("fixed", 20), ("overrun", 0), ("overrun", 5), ("overrun", 10)]
ACCURACIES = [0.6, 0.75, 0.9]
SHARED = [0.0, 0.3, 0.6]
BOX = {"BALL IN THE BOX", "PLAYERS IN THE BOX"}
STRONG = {"PROGRESSION", "DEFENDING IN MIDDLE THIRD", "BUILD UP", "DEFENDING IN ATTACKING THIRD"}
SOURCES = {"all": dict(PHASES),
           "no box codes": {k: v for k, v in PHASES.items() if k not in BOX},
           "build up + progression": {k: v for k, v in PHASES.items() if k in STRONG}}
SCALES = [0.5, 1.0, 2.0]
SHIPPED = ("all", PHASE_SCALE)          # what scripts/decode.py uses


def game(match_id):
    clean_m, m = load("idsse/{}".format(match_id)), load("idsse_degraded/{}".format(match_id))
    clean, feats = build(clean_m), build(m)
    truth = target_labels(m, "strict")
    decoded = decode(feats, m.fps, m.pitch, None)
    return {"m": m, "feats": feats, "truth": truth, "decoded": decoded,
            "missing": feats["ball_x"].isna().to_numpy(), "period": feats["period"].to_numpy(),
            "frame": feats["frame"].to_numpy(),
            "trig": st.triggers(clean, clean_m.players, truth, m.fps, m.pitch),
            "base": score(truth, decoded)["spell_f1_mean"]}


def ivals(g, rule):
    kind, value = rule
    return st.intervals(g["trig"], value if kind == "fixed" else None, g["m"].fps,
                        len(g["feats"]), g["period"], g["truth"],
                        value if kind == "overrun" else None)


def worlds(games):
    """[(rule, accuracy, r, q, {game key: intervals})] for every feasible combination."""
    out = []
    for rule in LENGTHS:
        iv = {g["m"].key: ivals(g, rule) for g in games}
        p = np.mean([st.held_accuracies(iv[g["m"].key], g["truth"], g["decoded"], g["missing"])
                     for g in games], axis=0)
        for acc in ACCURACIES:
            for r in SHARED:
                q = st.solve_q(acc, r, p[0], p[1])
                if q is not None:
                    out.append((rule, acc, r, q, iv))
    return out


def gains(games, world, decisions):
    """{decision: mean spellF1 gain over the games} in one world, and the realised accuracy."""
    rule, acc, r, q, iv = world
    out = {d: [] for d in decisions}
    realised = []
    for g in games:
        m = g["m"]
        tags = st.make_tags(iv[m.key], g["decoded"], g["frame"], r, q,
                            st.seed_for(m.key, rule, acc, r))
        realised.append(st.realised_accuracy(tags, g["frame"], g["truth"], g["missing"]))
        for source, scale in decisions:
            e = phase_team_evidence(g["feats"], tags, SOURCES[source], scale)
            labels = decode(g["feats"], m.fps, m.pitch, None, team_evidence=e)
            out[(source, scale)].append(score(g["truth"], labels)["spell_f1_mean"] - g["base"])
    return {d: float(np.mean(v)) for d, v in out.items()}, float(np.mean(realised))


def name(world):
    (kind, value), acc, r, q, _ = world
    rule = "fixed {}s".format(value) if kind == "fixed" else "to end +{}s".format(value)
    return "{:<12} A {:.2f} r {:.1f}".format(rule, acc, r)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="read validation once")
    args = parser.parse_args()
    splits = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]
    decisions = [(s, k) for s in SOURCES for k in SCALES]

    train = [game(i) for i in splits["train"]["idsse_degraded"]]
    ws = worlds(train)
    print("{} train games, {} feasible worlds, {} decisions\n".format(len(train), len(ws),
                                                                     len(decisions)))
    table = {}
    for w in ws:
        g, realised = gains(train, w, decisions)
        table[name(w)] = g
        best = max(g, key=g.get)
        print("  {}  (accuracy reached {:.2f})  best: {} x{:g} {:+.3f}   all x1: {:+.3f}".format(
            name(w), realised, best[0], best[1], g[best], g[("all", 1.0)]), flush=True)

    print("\n{:<26} {:>9} {:>9} {:>14}".format("decision", "worst", "mean", "worlds > 0"))
    summary = {}
    for d in decisions:
        v = np.array([table[w][d] for w in table])
        summary[d] = v.min()
        print("{:<26} {:+9.3f} {:+9.3f} {:>9}/{}".format("{} x{:g}".format(*d), v.min(), v.mean(),
                                                         int(np.sum(v > 1e-9)), len(v)))
    choice = max(summary, key=summary.get)
    print("\nmost robust: {} x{:g} (worst-case gain {:+.3f})".format(*choice, summary[choice]))
    print("shipped:     {} x{:g} (worst-case gain {:+.3f})".format(*SHIPPED, summary[SHIPPED]))

    if args.validate:
        val = [game(i) for i in splits["validation"]["idsse_degraded"]]
        vw = worlds(val)
        v = np.array([gains(val, w, [SHIPPED])[0][SHIPPED] for w in vw])
        ok = v.mean() >= 0 and v.min() >= -0.01
        print("\nvalidation, read once, shipped decision ({} worlds): mean gain {:+.3f}, worst {:+.3f}, best {:+.3f}"
              " -> bar {}".format(len(vw), v.mean(), v.min(), v.max(),
                                  "passed" if ok else "FAILED"))


if __name__ == "__main__":
    main()
