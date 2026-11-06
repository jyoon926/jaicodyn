"""Sanity-check processed matches.

team: nearest player to the ball is on the labelled team (under 50% suggests swapped sides).
dist: median metres from ball to nearest player (large suggests misaligned frames).
"""

import numpy as np

from jaicodyn.data import available


def check(match):
    frames, players = match.frames, match.players
    known = frames[frames["ball_x"].notna() & frames["label"].isin(["H", "A"])]
    merged = players.merge(known[["frame", "ball_x", "ball_y", "label"]], on="frame")
    length, width = match.pitch
    merged["dist"] = np.hypot((merged["x"] - merged["ball_x"]) * length,
                              (merged["y"] - merged["ball_y"]) * width)
    nearest = merged.loc[merged.groupby("frame")["dist"].idxmin()]
    possessions = match.possessions
    seconds = (possessions["stop"] - possessions["start"]) / match.fps
    return {"key": match.key, "frames": len(frames), "poss": len(possessions),
            "mean_s": seconds.mean(), "dead": (frames["label"] == "D").mean(),
            "ball": frames["ball_x"].notna().mean(),
            "team": (nearest["team"] == nearest["label"]).mean(),
            "dist": nearest["dist"].median(),
            "players": len(players) / len(frames)}


def main():
    print("{:<22} {:>7} {:>5} {:>6} {:>6} {:>6} {:>6} {:>5} {:>7}".format(
        "match", "frames", "poss", "mean_s", "dead", "ball", "team", "dist", "players"))
    for match in available():
        print("{key:<22} {frames:>7} {poss:>5} {mean_s:>6.1f} {dead:>6.1%} {ball:>6.1%} "
              "{team:>6.1%} {dist:>5.1f} {players:>7.1f}".format(**check(match)))


if __name__ == "__main__":
    main()
