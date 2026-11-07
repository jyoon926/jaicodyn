"""Build idsse_degraded: idsse games with the ball hidden to look like the competition tracking.

    python -m scripts.degrade --measure     # measure the settings into config/degrade.json
    python -m scripts.degrade               # build data/processed/idsse_degraded/*, and report

--measure reads the competition match's own tracking and automatic tags, no labels:
    missing, gap_lengths_s  how much of the ball track is missing, in how long gaps
    dead_ratio              how much likelier the ball is missing while dead (A below); the tags
                            give a lower bound to check it against (B)
    line_weights            how much likelier a gap starts near or beyond a line (C)
SkillCorner's own ratio (11x) is recorded but not used: its TV feed cuts to replays during
stoppages, which the competition's wide shot does not. See jaicodyn/data/degrade.py.
"""

import argparse
import json

import numpy as np
import pandas as pd

from jaicodyn.data import available, load, paths
from jaicodyn.data.degrade import CONFIG, degrade, settings
from jaicodyn.data.match import write
from jaicodyn.data.sources import idsse_degraded
from jaicodyn.features import build
from jaicodyn.features.ball import true_runs
from scripts.build_features import resolve


SPEED_BINS = [0, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 9]  # player speed / match median
LINE_BANDS = [(-1000, 0), (0, 3), (3, 10), (10, 1000)]       # metres from the nearest line


def speed_ratio(feats, fps, window_s=3.0):
    s = feats["speed_med"].to_numpy(float)
    smooth = pd.Series(s).rolling(int(window_s * fps), center=True, min_periods=1).mean()
    return smooth.to_numpy() / np.nanmedian(s)


def line_distance(feats, pitch):
    """Metres from the ball to the nearest touchline or goal line; negative off the pitch."""
    length, width = pitch
    x, y = feats["ball_x"].to_numpy(float), feats["ball_y"].to_numpy(float)
    return np.minimum.reduce([x, length - x, y, width - y])


def dead_given_speed():
    """P(dead | player speed ratio bin) on the clean idsse train games, where every player is
    tracked as in the competition's wide shot."""
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse"]
    num, den = np.zeros(len(SPEED_BINS) - 1), np.zeros(len(SPEED_BINS) - 1)
    for match_id in train:
        m = load("idsse/{}".format(match_id))
        lab = m.frames.sort_values("frame")["label"].to_numpy()
        b = np.clip(np.digitize(speed_ratio(build(m), m.fps), SPEED_BINS) - 1, 0, len(num) - 1)
        for k in range(len(num)):
            sel = (lab != "X") & (b == k)
            den[k] += sel.sum()
            num[k] += (sel & (lab == "D")).sum()
    return num / np.maximum(den, 1)


def ratio_from_speed(comp, feats, missing):
    """A: Bayes on the competition match. P(dead | speed) from idsse gives the dead share of its
    missing and visible frames; P(missing | dead) / P(missing | live) follows."""
    p_dead = dead_given_speed()
    b = np.clip(np.digitize(speed_ratio(feats, comp.fps), SPEED_BINS) - 1, 0, len(p_dead) - 1)
    q = p_dead[b]
    dead, dead_missing, share = q.mean(), q[missing].mean(), missing.mean()
    return (dead_missing * share / dead) / ((1 - dead_missing) * share / (1 - dead))


def ratio_from_tags(comp, feats, missing):
    """B: missing share just before automatic set-piece restarts vs far from any tag. A lower
    bound: untagged stoppages (fouls, most throw-ins) sit in the 'far' frames."""
    fps, frame = comp.fps, feats["frame"].to_numpy()
    tags = comp.tags
    starts = tags.loc[tags["code"].str.upper() == "SET PIECES", "start"].to_numpy()
    near, far = np.zeros(len(feats), bool), np.ones(len(feats), bool)
    for s in starts:
        a, b = np.searchsorted(frame, [s - 10 * fps, s + 2 * fps])
        near[a:b] = True
        a, b = np.searchsorted(frame, [s - 30 * fps, s + 30 * fps])
        far[a:b] = False
    return missing[near].mean() / missing[far].mean()


def line_weights(comp, feats, missing):
    """C: how much likelier a gap starts with the ball in each band from the lines than the
    visible ball is to be there -- the ball leaving the wide shot."""
    d = line_distance(feats, comp.pitch)
    starts, lengths = true_runs(missing)
    last = starts[(lengths >= 5 * comp.fps) & (starts > 0)] - 1
    last = last[~np.isnan(d[last])]
    seen = d[~missing]
    return [round(float(np.mean((d[last] >= lo) & (d[last] < hi))
                        / max(np.mean((seen >= lo) & (seen < hi)), 1e-9)), 2)
            for lo, hi in LINE_BANDS]


def skillcorner_ratio():
    """For reference only: TV replays during stoppages make this far higher than the wide shot's."""
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["skillcorner"]
    dead, live = [], []
    for match_id in train:
        t = load("skillcorner/{}".format(match_id)).frames
        miss, lab = t["ball_x"].isna().to_numpy(), t["label"].to_numpy()
        dead.append(miss[lab == "D"].mean())
        live.append(miss[np.isin(lab, ["H", "A"])].mean())
    return np.mean(dead) / np.mean(live)


def measure():
    comp = load(resolve("BAR-ATH"))
    feats = build(comp)
    missing = feats["ball_x"].isna().to_numpy()
    _, lengths = true_runs(missing)
    gaps = lengths / comp.fps
    long = np.sort(gaps[gaps >= 5])
    a = ratio_from_speed(comp, feats, missing)
    b = ratio_from_tags(comp, feats, missing)
    w = line_weights(comp, feats, missing)
    config = {
        "missing": round(float(missing.mean()), 4),
        "dead_ratio": round(float(a), 2),
        "line_bands_m": LINE_BANDS,
        "line_weights": w,
        "stoppage_spill_s": None,  # no cut: at this ratio cutting stoppage gaps changes nothing
        "gap_lengths_s": [round(float(g), 2) for g in long],
        "checks": {"dead_ratio_from_tags_lower_bound": round(float(b), 2),
                   "dead_ratio_skillcorner_not_used": round(float(skillcorner_ratio()), 2)},
        "sources": "all from competition/BAR-ATH tracking and its automatic smart tagging, no "
                   "labels; dead_ratio by Bayes with P(dead | player speed) from the idsse "
                   "train labels. SkillCorner's ratio is TV replays, so it is not used.",
    }
    config["line_weights_applied"] = calibrate(config)
    CONFIG.write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    print("wrote {}: missing {:.1%}, dead_ratio {:.2f} (tags lower bound {:.2f}, SkillCorner "
          "{:.1f} not used), line weights {}, {} gap lengths (median {:.1f}s)".format(
              CONFIG, config["missing"], a, b, config["checks"]["dead_ratio_skillcorner_not_used"],
              w, len(long), np.median(long)))


def start_weights(tracks, bands):
    """Gap-start weight per band (gap starts vs visible frames), over (frames, raw ball) pairs."""
    starts_in, seen_in = np.zeros(len(bands)), np.zeros(len(bands))
    for frames, d in tracks:
        miss = frames["ball_x"].isna().to_numpy()
        s, n = true_runs(miss)
        last = s[(n >= 5 * 25) & (s > 0)] - 1
        for k, (lo, hi) in enumerate(bands):
            starts_in[k] += np.sum((d[last] >= lo) & (d[last] < hi))
            seen_in[k] += np.sum((d[~miss] >= lo) & (d[~miss] < hi))
    return (starts_in / starts_in.sum()) / np.maximum(seen_in / seen_in.sum(), 1e-9)


def calibrate(c, rounds=3):
    """Weights that make degraded idsse train games reproduce the competition's gap-start pattern.

    The measured weights already contain the stoppages-happen-at-lines effect, which placing gaps
    over idsse's own stoppages adds again. Matched on shape -- each band's weight relative to
    mid-pitch -- since the visible ball sits in different places in idsse than in the competition,
    so the absolute weights cannot all match at once.
    """
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse"]
    matches = [load("idsse/{}".format(i)) for i in train]
    target = np.asarray(c["line_weights"])
    applied = target.copy()
    for _ in range(rounds):
        tracks = []
        for m in matches:
            frames = degrade(m, dict(c, line_weights=list(applied))).frames
            raw = m.frames.sort_values("frame", ignore_index=True)
            length, width = m.pitch
            x, y = raw["ball_x"].to_numpy() * length, raw["ball_y"].to_numpy() * width
            tracks.append((frames, np.minimum.reduce([x, length - x, y, width - y])))
        achieved = start_weights(tracks, c["line_bands_m"])
        shape, want = achieved / achieved[-1], target / target[-1]  # relative to mid-pitch
        print("  calibration: starts vs mid-pitch {} target {}".format(np.round(shape, 2),
                                                                     np.round(want, 2)))
        applied = applied * want / np.maximum(shape, 1e-9)
    return [round(float(v), 3) for v in applied]


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
