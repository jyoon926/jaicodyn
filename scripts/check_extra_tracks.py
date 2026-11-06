"""Do the extra tracks (more than 11 on a team) change the ball features?

    python -m scripts.check_extra_tracks BAR-ATH

On every frame where a team still has more than 11 players after dedup, the team's tracks are
ranked by how long their tracklet (a contiguous run of one track id) lives; tracks beyond the
11 longest-lived are the extras. Ghosts and false detections are short-lived, real players are
not. The team's ball features are then recomputed without the extras and compared:

    near     the extra was the team's nearest player, so near_{t} changes
    flip     ... and that changes which team is nearest the ball (the possession cue)
    cnt2/5   an extra was within 2 / 5 m of the ball, so the count is inflated

Where the extras sit -- distance to the ball, off the pitch, lifetime -- says what they are.
"""

import argparse

import numpy as np
import pandas as pd

from jaicodyn.data import load
from jaicodyn.features import build
from jaicodyn.features.team import TEAMS, ball_relative, deduplicate, dense_positions
from scripts.build_features import resolve


def tracklet_lifetimes(players, fps):
    """Seconds each row's tracklet lives, aligned to `players` rows."""
    p = players[["team", "track_id", "frame"]].sort_values(["team", "track_id", "frame"])
    new = ((p["frame"].diff() != 1) | (p["track_id"] != p["track_id"].shift())
           | (p["team"] != p["team"].shift()))
    run = new.cumsum()
    life = run.map(run.value_counts()) / fps
    return life.reindex(players.index)


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "  ".join("p{}={:.1f}".format(q, v) for q, v in zip(qs, np.percentile(x, qs))) \
        if len(x) else "n/a"


def check(match):
    feats = build(match)
    length, width = match.pitch
    players = match.players[match.players["x"].notna()].copy()
    players["life"] = tracklet_lifetimes(players, match.fps)
    ball = feats[["frame", "ball_x", "ball_y"]]
    print("{}  {} frames\n".format(match.key, len(feats)))

    near_other = {}
    rows = []
    for t in TEAMS:
        over = feats[feats["n_" + t] > 11]
        p = players[(players["team"] == t) & players["frame"].isin(over["frame"])].copy()
        p["rank"] = p.sort_values("life", ascending=False).groupby("frame").cumcount()
        p["extra"] = p["rank"] >= 11
        extras = p[p["extra"]].merge(ball, on="frame")
        dx = extras["x"] * length - extras["ball_x"]
        dy = extras["y"] * width - extras["ball_y"]
        extras["d_ball"] = np.hypot(dx, dy)

        # features for this team without the extras, on the affected frames
        index = pd.Index(over["frame"])
        pos = deduplicate(dense_positions(p[~p["extra"]], index, t, match.pitch))
        xy = over[["ball_x", "ball_y"]].to_numpy(float)
        near, counts = ball_relative(pos, xy)
        has_ball = ~np.isnan(xy[:, 0])
        old_near = over["near_" + t].to_numpy(float)
        changed = has_ball & ~np.isclose(near, old_near, atol=0.01, equal_nan=True)
        other = "A" if t == "H" else "H"
        o = over["near_" + other].to_numpy(float)
        flip = changed & ((old_near < o) != (np.nan_to_num(near, nan=np.inf) < o))
        near_other[t] = (over["frame"].to_numpy()[flip])

        n, nb = len(over), int(has_ball.sum())
        rows.append((t, n, n / len(feats), nb, changed.sum(), flip.sum(),
                     np.sum(has_ball & (over["cnt2_" + t].to_numpy() != counts[2])),
                     np.sum(has_ball & (over["cnt5_" + t].to_numpy() != counts[5]))))

        off = ((extras["x"] < 0) | (extras["x"] > 1) | (extras["y"] < 0) | (extras["y"] > 1))
        print("team {}: {} extra track-frames on {} frames".format(t, len(extras), n))
        print("  distance to ball (m): {}".format(pct(extras["d_ball"])))
        d = extras["d_ball"].dropna()  # shares over frames where the ball is known
        print("  within 2 m {:.1%}   within 5 m {:.1%}   over 20 m {:.1%}   off the pitch {:.1%}"
              .format(np.mean(d < 2), np.mean(d < 5), np.mean(d > 20), off.mean()))
        print("  tracklet lifetime (s): {}".format(pct(extras["life"])))
        print("  kept 11 lifetime (s):  {}\n".format(pct(p.loc[~p["extra"], "life"])))

    print("effect on ball features, frames where the team has >11 and the ball is known")
    print("{:<4} {:>7} {:>7} {:>7} {:>13} {:>13} {:>13} {:>13}".format(
        "team", "frames", "share", "w/ball", "near changes", "nearest flip", "cnt2 changes",
        "cnt5 changes"))
    for t, n, share, nb, ch, fl, c2, c5 in rows:
        f = lambda k: "{:>5} {:>6.1%}".format(int(k), k / max(nb, 1))  # noqa: E731
        print("{:<4} {:>7} {:>7.1%} {:>7} {:>13} {:>13} {:>13} {:>13}".format(
            t, n, share, nb, f(ch), f(fl), f(c2), f(c5)))
    graded = feats["ball_x"].notna().sum()
    flips = len(set(near_other["H"]) | set(near_other["A"]))
    print("\nnearest-team flips caused by extras: {} frames = {:.2%} of frames with a ball"
          .format(flips, flips / graded))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", help="match id or source/id")
    check(load(resolve(parser.parse_args().match)))


if __name__ == "__main__":
    main()
