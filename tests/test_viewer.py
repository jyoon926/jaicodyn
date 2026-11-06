import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.match import write
from jaicodyn.data.schema import Tracking, frames_table, players_table
from jaicodyn.viz.api import CHUNK, View, runs


def demo(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "DATA", tmp_path)
    numbers = np.arange(CHUNK - 2, CHUNK + 8)
    ball_x = np.linspace(0.1, 0.9, 10)
    ball_x[-1] = np.nan
    frames = frames_table(numbers, [1] * 10, ball_x, [0.5] * 10, list("HHHHHAAADD"))
    players = players_table(numbers, ["H"] * 10, ["7"] * 10, [0.2] * 10, [0.3] * 10)
    write("demo", "m1", Tracking(frames, players, 25, (105.0, 68.0), "Home", "Away"))
    return load("demo/m1")


def test_runs_break_at_label_changes_and_gaps():
    assert runs([1, 2, 3, 7], list("HHAA")) == [[1, 3, "H"], [3, 4, "A"], [7, 8, "A"]]
    assert runs([], []) == []


def test_info_lists_labels_and_predictions(tmp_path, monkeypatch):
    match = demo(tmp_path, monkeypatch)
    folder = tmp_path / "predictions" / "demo" / "m1"
    folder.mkdir(parents=True)
    (folder / "mine.csv").write_text("frame,label\n{},A\n".format(CHUNK))

    info = View(match).info()
    assert [row["name"] for row in info["rows"]] == ["labels", "mine"]
    assert info["rows"][1]["runs"] == [[CHUNK, CHUNK + 1, "A"]]
    assert (info["video"], info["homography"], info["tags"]) == (False, False, [])


def test_chunks_align_to_frames_across_a_boundary(tmp_path, monkeypatch):
    view = View(demo(tmp_path, monkeypatch))
    first, second = view.chunk(0), view.chunk(1)
    assert first["players"][0] is None
    assert first["players"][CHUNK - 1][0][:2] == ["H", "7"]
    assert second["start"] == CHUNK
    assert second["ball"][0][0] == round(float(np.linspace(0.1, 0.9, 10)[2]), 5)
    assert second["ball"][7] is None
    assert second["players"][7][0][4] == 0.0
