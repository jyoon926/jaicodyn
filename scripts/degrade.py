"""Build idsse_degraded: idsse games with the ball hidden to look like the competition tracking.

    python -m scripts.degrade --measure     # measure the settings into config/degrade.json
    python -m scripts.degrade               # build data/processed/idsse_degraded/*, and report

--measure reads the competition match's ball track (no labels) for how much of it is missing and
how long its gaps are, and the SkillCorner train games' labels for how much more often the ball
is missing while dead. See jaicodyn/data/degrade.py.
"""

import argparse
import json

import numpy as np

from jaicodyn.data import available, load, paths
from jaicodyn.data.degrade import CONFIG, settings
from jaicodyn.data.match import write
from jaicodyn.data.sources import idsse_degraded
from jaicodyn.features.ball import true_runs
from scripts.build_features import resolve


def measure():
    comp = load(resolve("BAR-ATH"))
    f = comp.frames.sort_values("frame")
    missing = f["ball_x"].isna().to_numpy()
    starts, lengths = true_runs(missing)
    gaps = lengths / comp.fps
    long = np.sort(gaps[gaps >= 5])

    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["skillcorner"]
    dead, live = [], []
    for match_id in train:
        m = load("skillcorner/{}".format(match_id))
        t = m.frames.sort_values("frame")
        miss, lab = t["ball_x"].isna().to_numpy(), t["label"].to_numpy()
        dead.append(miss[lab == "D"].mean())
        live.append(miss[np.isin(lab, ["H", "A"])].mean())
    config = {
        "missing": round(float(missing.mean()), 4),
        "dead_ratio": round(float(np.mean(dead) / np.mean(live)), 2),
        "gap_lengths_s": [round(float(g), 2) for g in long],
        "sources": "missing and gap_lengths_s from competition/BAR-ATH tracking (no labels); "
                   "dead_ratio from the skillcorner train split",
    }
    CONFIG.write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    print("wrote {}: missing {:.1%}, dead_ratio {:.1f}, {} gap lengths (median {:.1f}s)".format(
        CONFIG, config["missing"], config["dead_ratio"], len(long), np.median(long)))


def report(match, c):
    t = match.frames.sort_values("frame")
    miss, lab = t["ball_x"].isna().to_numpy(), t["label"].to_numpy()
    graded = lab != "X"
    dead, live = miss[graded & (lab == "D")].mean(), miss[graded & np.isin(lab, ["H", "A"])].mean()
    _, lengths = true_runs(miss)
    gaps = lengths / match.fps
    print("{:<26} missing {:5.1%} (target {:.1%})  dead {:4.0%}  live {:4.0%}  ratio {:5.1f} "
          "(target {:.1f})  gaps {:>3}, median {:4.1f}s".format(
              match.key, miss[graded].mean(), c["missing"], dead, live, dead / max(live, 1e-9),
              c["dead_ratio"], len(gaps), np.median(gaps) if len(gaps) else 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true", help="measure into config/degrade.json")
    args = parser.parse_args()
    if args.measure:
        measure()
        return
    c = settings()
    for match in [m for m in available() if m.source == "idsse"]:
        write("idsse_degraded", match.id, idsse_degraded.read(match.id))
        report(load("idsse_degraded/{}".format(match.id)), c)


if __name__ == "__main__":
    main()
