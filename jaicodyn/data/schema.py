"""The one schema. Coordinates are 0-1, origin top left; NaN means not observed."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LABELS = {"H", "A", "D", "X"}
PERIODS = {1, 2}


@dataclass
class Tracking:
    """What a source produces for one match."""

    frames: pd.DataFrame
    players: pd.DataFrame
    fps: int
    pitch: tuple
    home: str
    away: str

    def validate(self):
        if not self.frames["frame"].is_unique:
            raise ValueError("duplicate frames")
        if not set(self.frames["label"]) <= LABELS:
            raise ValueError("unknown labels")
        if not set(self.frames["period"]) <= PERIODS:
            raise ValueError("unknown periods")
        for column in (self.frames["ball_x"], self.frames["ball_y"],
                       self.players["x"], self.players["y"]):
            if column.dropna().between(-0.15, 1.15).mean() < 0.99:
                raise ValueError("coordinates are not 0-1: {}".format(column.name))


def frames_table(frame, period, ball_x, ball_y, label):
    return pd.DataFrame({
        "frame": np.asarray(frame, dtype=np.int32),
        "period": np.asarray(period, dtype=np.int8),
        "ball_x": np.asarray(ball_x, dtype=np.float32),
        "ball_y": np.asarray(ball_y, dtype=np.float32),
        "label": np.asarray(label, dtype=object),
    }).sort_values("frame", ignore_index=True)


def players_table(frame, team, track_id, x, y):
    return pd.DataFrame({
        "frame": np.asarray(frame, dtype=np.int32),
        "team": np.asarray(team, dtype=object),
        "track_id": np.asarray(track_id, dtype=object),
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
    }).sort_values(["frame", "team", "track_id"], ignore_index=True)


def centred_to_unit(x, y, pitch):
    """Centre-origin, y-up metres to schema coordinates."""
    length, width = pitch
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (x + length / 2) / length, 1 - (y + width / 2) / width
