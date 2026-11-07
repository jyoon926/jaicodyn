"""Decode possession for a match, and score it where labels exist.

    python -m scripts.decode BAR-ATH                    # writes jaicodyn.csv, the submission
    python -m scripts.decode BAR-ATH --public           # ... and checks it on the public window
    python -m scripts.decode idsse/J03WMX               # a labelled game: scores it
    python -m scripts.decode --split train validation   # labelled games in those splits
    python -m scripts.decode --all-labelled             # every labelled processed game, test too
    python -m scripts.decode BAR-ATH --probs            # also save P(H), P(A), P(D) per frame

The public window is a pipeline check -- frames aligned, labels meaning what we think -- never a
tuning signal (challenge/RULES.md). Tune on the other labelled games, using train and
validation only: config/splits.json keeps test for one run after everything is frozen.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from challenge.evaluation import possession_metrics as pm
from challenge.games import GAMES
from jaicodyn.data import available, load, paths
from jaicodyn.data.match import is_built
from jaicodyn.decode import decode, posterior
from jaicodyn.features import build
from scripts.build_features import resolve

PUBLIC_GT = paths.ROOT / "challenge" / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
SUBMISSION = paths.ROOT / "jaicodyn.csv"


def predict(match):
    feats = build(match)
    return feats["frame"].to_numpy(), decode(feats, match.fps, match.pitch, match.tags)


def save_probs(match):
    """P(H), P(A), P(D) per frame, next to the features. Returns the path."""
    feats = build(match)
    post = posterior(feats, match.fps, match.pitch, match.tags)
    path = paths.DATA / "decoded" / match.source / "{}_probs.parquet".format(match.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": feats["frame"], "p_H": post[:, 0], "p_A": post[:, 1],
                  "p_D": post[:, 2]}).astype({"p_H": "float32", "p_A": "float32",
                                              "p_D": "float32"}).to_parquet(path, index=False)
    return path


def submission(match, frame, labels):
    """Full-length frame,label table: one row per tactical video frame, X off the graded halves."""
    full = np.full(GAMES[match.id]["n_video_frames"], "X", dtype=object)
    full[frame] = labels
    return pd.DataFrame({"frame": np.arange(len(full)), "label": full})


def stats(labels):
    codes = pm.label_codes(np.asarray(labels, dtype=object))
    return pm.possession_stats(codes, pm.valid_runs(codes))


def score(gt_labels, pred_labels):
    gt = pm.label_codes(np.asarray(gt_labels, dtype=object))
    pred = pm.label_codes(np.asarray(pred_labels, dtype=object))
    return pm.evaluate(gt, pred, bootstrap_n=0)


def show(key, r):
    print("{:<24} spellF1 {:.3f}  macroF1 {:.3f}  possessions {:>4} (gt {:>4})  "
          "mean {:5.1f}s (gt {:5.1f}s)  f1 H/A/D {:.2f}/{:.2f}/{:.2f}".format(
              key, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
              r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"],
              *(r["per_class"][k]["f1"] for k in "HAD")))


def competition(match, public, out):
    frame, labels = predict(match)
    table = submission(match, frame, labels)
    reason = pm.validate_submission(table, len(table))
    if reason:
        sys.exit("submission invalid: " + reason)
    table.to_csv(out, index=False)
    s = stats(table["label"])
    print("{}: wrote {} ({} rows, valid)".format(match.key, out, len(table)))
    print("  {} possessions, mean {:.1f}s, home share {:.1%}, dead {:.1%} of graded frames".format(
        s["n_spells"], s["mean_duration_s"], s["home_share"], np.mean(labels == "D")))
    if public:
        gt = pm.read_label_csv(str(PUBLIC_GT))["label"]
        print("\n[public window -- pipeline check only, do not tune on this]")
        show(match.key, score(gt, table["label"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matches", nargs="*", help="match ids or source/id")
    parser.add_argument("--split", nargs="+", choices=["train", "validation", "test"],
                        help="labelled games in these splits that are processed")
    parser.add_argument("--all-labelled", action="store_true", help="every labelled game")
    parser.add_argument("--public", action="store_true", help="check BAR-ATH on the public window")
    parser.add_argument("--out", default=str(SUBMISSION), help="submission path")
    parser.add_argument("--probs", action="store_true", help="save per-frame state probabilities")
    args = parser.parse_args()

    matches = [load(resolve(name)) for name in args.matches]
    if args.all_labelled:
        matches += [m for m in available() if m.source != "competition"]
    for name in args.split or []:
        table = json.loads(paths.SPLITS.read_text("utf-8"))["splits"][name]
        matches += [load("{}/{}".format(s, i)) for s, ids in table.items() for i in ids
                    if is_built(s, i)]
    if not matches:
        parser.error("give a match, --split, or --all-labelled")

    results = []
    for match in matches:
        if args.probs:
            print("{}: probabilities in {}".format(match.key, save_probs(match)))
        if match.source == "competition":
            competition(match, args.public, args.out)
            continue
        frame, labels = predict(match)
        r = score(match.frames.sort_values("frame")["label"], labels)
        show(match.key, r)
        results.append(r)
    if len(results) > 1:
        print("\nmean over {} games: spellF1 {:.3f}  macroF1 {:.3f}".format(
            len(results), np.mean([r["spell_f1_mean"] for r in results]),
            np.mean([r["macro_f1"] for r in results])))


if __name__ == "__main__":
    main()
