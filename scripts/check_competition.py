"""Check the parsed competition match against published numbers and the official baseline."""

import numpy as np
import pandas as pd

from jaicodyn.data import load
from jaicodyn.data.paths import ROOT

BASELINE = ROOT / "challenge" / "evaluation" / "results" / "baseline_raw.csv"


def nearest_team(match):
    """Team of the player nearest the ball, per frame with a ball."""
    ball = match.frames.dropna(subset=["ball_x"])[["frame", "ball_x", "ball_y"]]
    merged = match.players.merge(ball, on="frame")
    length, width = match.pitch
    merged["dist"] = np.hypot((merged["x"].astype(float) - merged["ball_x"]) * length,
                              (merged["y"].astype(float) - merged["ball_y"]) * width)
    return merged.loc[merged.groupby("frame")["dist"].idxmin()].set_index("frame")["team"]


def main():
    match = load("competition/BAR-ATH")
    frames, players = match.frames, match.players
    print("{} v {}, {} fps, pitch {}".format(match.home, match.away, match.fps, match.pitch))
    print("{:<24} {:>10} {:>10}".format("", "ours", "expected"))
    print("{:<24} {:>10} {:>10}".format("graded frames", len(frames), 158857))
    print("{:<24} {:>10} {:>10}".format("frames with a ball", frames["ball_x"].notna().sum(), 94214))
    print("{:<24} {:>10.1f} {:>10}".format("players per frame", len(players) / len(frames), 20.5))

    official = pd.read_csv(BASELINE).set_index("frame")["label"].reindex(frames["frame"])
    seen = official != "D"
    ball_agrees = (frames["ball_x"].notna().values == seen.values).mean()
    ours = nearest_team(match)
    team_agrees = (ours == official.reindex(ours.index)).mean()
    print("{:<24} {:>10.2%} {:>10}".format("ball agrees w/ baseline", ball_agrees, "100%"))
    print("{:<24} {:>10.2%} {:>10}".format("team agrees w/ baseline", team_agrees, "~100%"))


if __name__ == "__main__":
    main()
