"""Fit and check the restart model (jaicodyn/decode/restart.py).

    python -m scripts.restart_model                # leave-one-game-out on train, then fit and save
    python -m scripts.restart_model --validate     # also read validation, once

Restarts are where the decoder, without restart evidence, goes from dead ball to live play; the
target is the team the true (strict) labels next give the ball to. Trained on the degraded idsse
train games (all players tracked, ball hidden like the competition's). For each game:

    restart accuracy   share of restarts given to the right team: decoder alone vs the model
    spellF1            the decoder alone vs the decoder with the model's restart evidence

The fitted coefficients go to config/restart_model.json, which the submission reads.
"""

import argparse
import json

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.labels import target_labels
from jaicodyn.data.match import is_built
from jaicodyn.decode import decode
from jaicodyn.decode.restart import (FEATURES, features, fit, p_home_array, p_home_rows,
                                     true_team)
from jaicodyn.features import build
from scripts.decode import score

MODEL = paths.ROOT / "config" / "restart_model.json"
SOURCE = "idsse_degraded"


def games(split):
    ids = json.loads(paths.SPLITS.read_text("utf-8"))["splits"][split].get(SOURCE, [])
    return [load("{}/{}".format(SOURCE, i)) for i in ids if is_built(SOURCE, i)]


def dataset(m):
    feats = build(m)
    truth = target_labels(m, "strict")
    first = decode(feats, m.fps, m.pitch, m.tags)
    X, rows = features(feats, first, m.players, m.fps, m.pitch)
    y = true_team(truth, feats["period"].to_numpy(), rows)
    keep = y != None  # noqa: E711 -- restarts with no live play after them in the period
    return {"m": m, "feats": feats, "truth": truth, "first": first, "X": X[keep].reset_index(
        drop=True), "rows": rows[keep], "y": y[keep], "all_rows": rows, "X_all": X}


def evaluate(d, coef):
    """(decoder restart acc, model restart acc, decoder spellF1, with-model spellF1)."""
    p_all = p_home_rows(coef, d["X_all"])
    p = p_home_rows(coef, d["X"])
    m, feats = d["m"], d["feats"]
    with_model = decode(feats, m.fps, m.pitch, m.tags,
                        restart_p_home=p_home_array(len(feats), d["all_rows"], p_all))
    return (np.mean(d["first"][d["rows"]] == d["y"]), np.mean(np.where(p > 0.5, "H", "A") == d["y"]),
            score(d["truth"], d["first"])["spell_f1_mean"],
            score(d["truth"], with_model)["spell_f1_mean"])


def report(title, results):
    r = np.array(results, dtype=float)
    print(title)
    print("  restart team right: decoder {:.0%}  model {:.0%}".format(r[:, 0].mean(), r[:, 1].mean()))
    gain = r[:, 3] - r[:, 2]
    print("  spellF1: decoder {:.3f}  with model {:.3f}  ({:+.3f}, better in {}/{} games)".format(
        r[:, 2].mean(), r[:, 3].mean(), gain.mean(), int(np.sum(gain > 1e-9)), len(gain)))


def stack(ds):
    import pandas as pd
    X = pd.concat([d["X"] for d in ds], ignore_index=True)
    y = np.concatenate([d["y"] for d in ds]) == "H"
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="read validation once")
    args = parser.parse_args()

    train = [dataset(m) for m in games("train")]
    print("{} train games, {} restarts".format(len(train), sum(len(d["y"]) for d in train)))
    held_out = []
    for k, d in enumerate(train):
        X, y = stack([o for j, o in enumerate(train) if j != k])
        held_out.append(evaluate(d, fit(X, y).coef_[0]))
    report("\nleave one game out (train)", held_out)

    X, y = stack(train)
    coef = fit(X, y).coef_[0]
    MODEL.write_text(json.dumps({"features": FEATURES, "coef": [round(float(c), 4) for c in coef],
                                 "trained_on": "{} train, strict labels".format(SOURCE)},
                                indent=1) + "\n", "utf-8")
    print("\nfitted on all train -> {}".format(MODEL))
    for name, c in sorted(zip(FEATURES, coef), key=lambda t: -abs(t[1])):
        print("  {:<22} {:+.2f}".format(name, c))

    if args.validate:
        report("\nvalidation, read once", [evaluate(dataset(m), coef) for m in games("validation")])


if __name__ == "__main__":
    main()
