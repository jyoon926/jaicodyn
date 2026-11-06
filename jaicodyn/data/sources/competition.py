"""The competition match. Frames are video frames, within the graded halves; labels are X."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from challenge.games import GAME, GAMES, asset_paths, half_of_video_frame
from challenge.starter import load_homography, load_smart_tagging
from challenge.starter.baseline import team_names, team_of_each_track

from .. import paths
from ..schema import Tracking, frames_table, players_table

MATCHES = [GAME]
TAG_TEAMS = {"Barcelona": "H", "Atletic Club": "A"}


def _files(match_id):
    return asset_paths(match_id, data_dir=str(paths.COMPETITION))


def is_ready(match_id):
    files = _files(match_id)
    return os.path.exists(files["atd_meta"]) and os.path.exists(files["atd_raw"])


def video(match_id):
    """The tactical video. Also accepts a download suffix, as in BAR-ATH_tactical-001.mp4."""
    path = Path(_files(match_id)["video"])
    if path.exists():
        return path
    matches = sorted(path.parent.glob(path.stem + "*" + path.suffix))
    return matches[0] if matches else None


def homography(match_id):
    if not Path(_files(match_id)["homography"]).exists():
        return None
    return load_homography.load(match_id, str(paths.COMPETITION))[0]


def tags(match_id):
    if not Path(_files(match_id)["smart_tagging"]).exists():
        return None
    return pd.DataFrame([
        {"start": event["start_frame"], "stop": event["end_frame"], "code": event["code"],
         "team": TAG_TEAMS.get(event["tags"].get("Team"), ""), "type": event["tags"].get("Type", "")}
        for event in load_smart_tagging.load(match_id, str(paths.COMPETITION))])


def _xy(group):
    if group.startswith("N"):  # missing, as in starter/baseline.py
        return np.nan, np.nan
    x, y = group.split(",")[:2]
    return float(x), float(y)


def read(match_id):
    files = _files(match_id)
    teams = np.array(["H", "A"])[team_of_each_track(files["atd_meta"])]

    frame, period, ball_x, ball_y = [], [], [], []
    p_frame, p_team, p_track, p_x, p_y = [], [], [], [], []
    with open(files["atd_raw"]) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            number, bodies, ball = parts
            video_frame = int(number) - 1
            half = half_of_video_frame(match_id, video_frame)
            if half is None:
                continue
            x, y = _xy(ball)
            frame.append(video_frame)
            period.append(half)
            ball_x.append(x)
            ball_y.append(y)
            for slot, body in enumerate(bodies.split(";")):
                x, y = _xy(body)
                if not np.isnan(x):
                    p_frame.append(video_frame)
                    p_team.append(teams[slot])
                    p_track.append(str(slot + 1))
                    p_x.append(x)
                    p_y.append(y)

    frames = frames_table(frame, period, ball_x, ball_y, np.full(len(frame), "X"))
    players = players_table(p_frame, p_team, p_track, p_x, p_y)
    info = GAMES[match_id]
    return Tracking(frames, players, info["fps"], info["pitch"], *team_names(files["atd_meta"]))
