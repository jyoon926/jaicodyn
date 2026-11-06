"""Turn a Match into what the viewer needs: JSON-ready metadata and chunks."""

import numpy as np
import pandas as pd

from jaicodyn.data import paths

CHUNK = 1500
SPEED_WINDOW_S = 0.2
PUBLIC_TRUTH = paths.ROOT / "challenge" / "evaluation" / "gt"


def runs(frames, values):
    """[start, stop, value] for constant runs, broken at gaps in the frame numbers."""
    frames, values = np.asarray(frames), np.asarray(values)
    if not len(frames):
        return []
    breaks = np.flatnonzero((values[1:] != values[:-1]) | (np.diff(frames) != 1)) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, len(frames)]
    plain = values.tolist()
    return [[int(frames[a]), int(frames[b - 1]) + 1, plain[a]] for a, b in zip(starts, stops)]


def _speed(frame, x, y, fps, pitch):
    """Metres per second over SPEED_WINDOW_S, NaN across gaps. Rows are in time order."""
    step = max(1, int(round(SPEED_WINDOW_S * fps)))
    out = np.full(len(frame), np.nan)
    if len(frame) > step:
        distance = np.hypot((x[step:] - x[:-step]) * pitch[0], (y[step:] - y[:-step]) * pitch[1])
        out[step:] = np.where(frame[step:] - frame[:-step] == step, distance * fps / step, np.nan)
    return out


def _player_speeds(match):
    table = match.players
    frame, x, y = table["frame"].values, table["x"].values, table["y"].values
    out = np.full(len(table), np.nan)
    for rows in table.groupby("track_id").indices.values():
        out[rows] = _speed(frame[rows], x[rows].astype(float), y[rows].astype(float),
                           match.fps, match.pitch)
    return out


def _rounded(number, digits):
    return None if np.isnan(number) else round(float(number), digits)


def _label_rows(match):
    """Timeline rows: the match's labels, public truth, then saved predictions."""
    frames = match.frames
    rows = []
    if not (frames["label"] == "X").all():
        rows.append(("labels", frames["label"].values))
    files = []
    public = PUBLIC_TRUTH / "{}_possession_gt_public.csv".format(match.id)
    if match.source == "competition" and public.exists():
        files.append(("public truth", public))
    folder = paths.DATA / "predictions" / match.source / match.id
    files += [(path.stem, path) for path in sorted(folder.glob("*.csv"))]
    for name, path in files:
        table = pd.read_csv(path).set_index("frame")["label"]
        rows.append((name, table.reindex(frames["frame"]).fillna("X").values))
    return [{"name": name, "runs": [r for r in runs(frames["frame"].values, values) if r[2] != "X"]}
            for name, values in rows]


class View:
    """One match, indexed for fast chunk lookups."""

    def __init__(self, match):
        self.match = match
        frames = match.frames
        self.frames = frames.set_index("frame", drop=False)
        self.players = match.players.reset_index(drop=True)
        self.player_speed = _player_speeds(match)
        self.player_frames = self.players["frame"].values
        seen = frames["ball_x"].notna().values
        self.ball_speed = pd.Series(np.nan, index=frames["frame"].values)
        self.ball_speed[seen] = _speed(frames["frame"].values[seen],
                                       frames["ball_x"].values[seen].astype(float),
                                       frames["ball_y"].values[seen].astype(float),
                                       match.fps, match.pitch)

    def info(self):
        match, frames = self.match, self.match.frames
        periods = frames.groupby("period")["frame"].agg(["min", "max"])
        tags = match.tags
        return {
            "key": match.key, "fps": match.fps, "pitch": match.pitch,
            "home": match.home, "away": match.away, "chunk": CHUNK,
            "first": int(frames["frame"].min()), "last": int(frames["frame"].max()),
            "periods": [[int(p), int(row["min"]), int(row["max"]) + 1]
                        for p, row in periods.iterrows()],
            "video": match.video is not None, "homography": match.homography is not None,
            "rows": _label_rows(match),
            "seen": runs(frames["frame"].values, frames["ball_x"].notna().values),
            "tags": [] if tags is None else tags.to_dict("records"),
        }

    def chunk(self, index):
        start = index * CHUNK
        ball, players, matrices = [None] * CHUNK, [None] * CHUNK, [None] * CHUNK
        window = self.frames.loc[start:start + CHUNK - 1]
        for frame, x, y in zip(window["frame"], window["ball_x"], window["ball_y"]):
            players[frame - start] = []
            if not np.isnan(x):
                ball[frame - start] = [round(float(x), 5), round(float(y), 5),
                                       _rounded(self.ball_speed[frame], 1)]

        first, last = np.searchsorted(self.player_frames, [start, start + CHUNK])
        rows = self.players.iloc[first:last]
        for row, speed in zip(rows.itertuples(index=False), self.player_speed[first:last]):
            if players[row.frame - start] is not None:
                players[row.frame - start].append(
                    [row.team, row.track_id, round(float(row.x), 5), round(float(row.y), 5),
                     _rounded(speed, 1)])

        homography = self.match.homography
        if homography is not None:
            for frame in range(start, min(start + CHUNK, len(homography))):
                if not np.isnan(homography[frame, 0, 0]):
                    matrices[frame - start] = [round(float(v), 7) for v in homography[frame].ravel()]
        return {"start": start, "ball": ball, "players": players, "homography": matrices}
