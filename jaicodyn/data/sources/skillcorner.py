"""SkillCorner: 10 fps broadcast tracking. Labels come from phases of play; undetected is NaN."""

import csv
import json

import numpy as np

from .. import paths
from ..schema import Tracking, centred_to_unit, frames_table, players_table

FPS = 10


def read(match_id):
    folder = paths.RAW / "skillcorner" / "matches" / match_id
    info = json.loads((folder / "{}_match.json".format(match_id)).read_text("utf-8"))
    pitch = float(info["pitch_length"]), float(info["pitch_width"])
    sides = {info["home_team"]["id"]: "H", info["away_team"]["id"]: "A"}
    team_of = {player["id"]: sides.get(player["team_id"]) for player in info["players"]}

    frame, period, ball_x, ball_y = [], [], [], []
    p_frame, p_team, p_track, p_x, p_y = [], [], [], [], []
    with open(folder / "{}_tracking_extrapolated.jsonl".format(match_id)) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("period") not in (1, 2):
                continue
            ball = row.get("ball_data") or {}
            seen = ball.get("is_detected") is True
            frame.append(row["frame"])
            period.append(row["period"])
            ball_x.append(ball["x"] if seen else np.nan)
            ball_y.append(ball["y"] if seen else np.nan)
            for player in row.get("player_data") or []:
                team = team_of.get(player["player_id"])
                if team and player.get("is_detected") is True:
                    p_frame.append(row["frame"])
                    p_team.append(team)
                    p_track.append(str(player["player_id"]))
                    p_x.append(player["x"])
                    p_y.append(player["y"])

    frame = np.array(frame)
    label = np.full(len(frame), "D", dtype=object)
    with open(folder / "{}_phases_of_play.csv".format(match_id), encoding="utf-8") as handle:
        for phase in csv.DictReader(handle):
            team = sides.get(int(float(phase["team_in_possession_id"] or 0)))
            if team:
                start, stop = int(phase["frame_start"]), int(phase["frame_end"])
                label[(frame >= start) & (frame <= stop)] = team

    frames = frames_table(frame, period, *centred_to_unit(ball_x, ball_y, pitch), label)
    players = players_table(p_frame, p_team, p_track, *centred_to_unit(p_x, p_y, pitch))
    return Tracking(frames, players, FPS, pitch, info["home_team"]["name"], info["away_team"]["name"])
