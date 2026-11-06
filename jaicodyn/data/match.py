"""The Match object, and reading and writing matches on disk."""

import json
from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from . import paths
from .labels import spells


@dataclass(frozen=True)
class Match:
    """One match. Tables load on first use."""

    source: str
    id: str
    fps: int
    pitch: tuple
    home: str
    away: str

    @property
    def key(self):
        return "{}/{}".format(self.source, self.id)

    @property
    def folder(self):
        return paths.PROCESSED / self.source / self.id

    @cached_property
    def frames(self):
        return pd.read_parquet(self.folder / "frames.parquet")

    @cached_property
    def players(self):
        return pd.read_parquet(self.folder / "players.parquet")

    @cached_property
    def possessions(self):
        """One row per team spell: start and stop frames, team, period."""
        return spells(self.frames)


def load(key):
    """Load a match by key, such as "idsse/J03WMX"."""
    source, match_id = key.split("/")
    path = paths.PROCESSED / source / match_id / "match.json"
    info = json.loads(path.read_text("utf-8"))
    return Match(source, match_id, info["fps"], tuple(info["pitch"]), info["home"], info["away"])


def split(name):
    """Matches in "train", "validation" or "test"."""
    table = json.loads(paths.SPLITS.read_text("utf-8"))["splits"][name]
    return [load("{}/{}".format(source, match_id))
            for source, ids in table.items() for match_id in ids]


def available():
    """Every processed match."""
    found = sorted(paths.PROCESSED.glob("*/*/match.json"))
    return [load("{}/{}".format(path.parent.parent.name, path.parent.name)) for path in found]


def is_built(source, match_id):
    return (paths.PROCESSED / source / match_id / "match.json").exists()


def write(source, match_id, tracking):
    tracking.validate()
    folder = paths.PROCESSED / source / match_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "match.json").unlink(missing_ok=True)  # written last, marks completion
    tracking.frames.to_parquet(folder / "frames.parquet", index=False, compression="zstd")
    tracking.players.to_parquet(folder / "players.parquet", index=False, compression="zstd")
    info = {"fps": tracking.fps, "pitch": tracking.pitch,
            "home": tracking.home, "away": tracking.away}
    (folder / "match.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")
