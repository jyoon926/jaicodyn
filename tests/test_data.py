import numpy as np
import pandas as pd
import pytest

from jaicodyn.data import absorb_short_spells, available, load, paths
from jaicodyn.data.match import write
from jaicodyn.data.schema import Tracking, frames_table, players_table
from jaicodyn.data.sources import competition
from jaicodyn.data.sources.metrica import labels_from_events


def table(labels, periods=None):
    n = len(labels)
    return pd.DataFrame({"frame": np.arange(n), "period": periods or [1] * n, "label": list(labels)})


def event(kind, team, start, end=None, subtype="", period=1):
    return {"type": kind, "team": team, "start": start, "end": start if end is None else end,
            "subtype": subtype, "period": period}


@pytest.mark.parametrize("before, minimum, after", [
    ("AAAHAAA", 2, "AAAAAAA"),
    ("AAAHHAAA", 2, "AAAHHAAA"),
    ("AAAHDDAA", 2, "AAAHDDAA"),
    ("AAHAAHHAA", 3, "AAAAAAAAA"),
])
def test_absorb(before, minimum, after):
    assert "".join(absorb_short_spells(table(before), minimum)) == after


def test_absorb_stays_within_a_period():
    assert "".join(absorb_short_spells(table("AHA", [1, 1, 2]), 2)) == "AHA"


def test_events_hold_through_a_challenge_and_stop_at_ball_out():
    events = [event("SET PIECE", "H", 2), event("CHALLENGE", "A", 4), event("RECOVERY", "A", 6),
              event("BALL OUT", "A", 8)]
    assert "".join(labels_from_events(events, np.arange(10), np.ones(10, int))) == "DDHHHHAADD"


def test_events_inside_a_stop_are_not_a_restart():
    events = [event("RECOVERY", "H", 2), event("BALL OUT", "H", 2, 5), event("CARRY", "H", 2, 3),
              event("SET PIECE", "A", 8)]
    assert "".join(labels_from_events(events, np.arange(10), np.ones(10, int))) == "DDHHHDDDAD"


def test_events_restart_after_a_goal():
    events = [event("PASS", "H", 0), event("SHOT", "H", 2, 3, "ON TARGET-GOAL"),
              event("SET PIECE", "A", 7), event("PASS", "A", 8, 9)]
    assert "".join(labels_from_events(events, np.arange(10), np.ones(10, int))) == "HHHDDDDAAA"


def test_match_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    frames = frames_table([0, 1, 2, 3], [1, 1, 1, 2], [0.5] * 4, [np.nan] * 4, list("HHDA"))
    players = players_table([0], ["H"], ["7"], [0.1], [0.2])
    write("demo", "m1", Tracking(frames, players, 25, (105.0, 68.0), "Home", "Away"))

    match = load("demo/m1")
    assert (match.key, match.fps, match.pitch) == ("demo/m1", 25, (105.0, 68.0))
    assert len(match.players) == 1
    assert match.possessions.values.tolist() == [[0, 2, "H", 1], [3, 4, "A", 2]]
    assert [m.key for m in available()] == ["demo/m1"]


def test_write_rejects_unnormalised_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    frames = frames_table([0], [1], [40.0], [20.0], ["H"])
    players = players_table([0], ["H"], ["7"], [0.1], [0.2])
    with pytest.raises(ValueError):
        write("demo", "m1", Tracking(frames, players, 25, (105.0, 68.0), "Home", "Away"))


def test_competition_format(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "COMPETITION", tmp_path)
    folder = tmp_path / "tactical"
    folder.mkdir()
    (folder / "BAR-ATH_tactical_atd.xml").write_text(
        '<root><Team id="a"><Name>Barcelona</Name></Team><Team id="b"><Name>Athletic</Name></Team>'
        '<Player teamId="a"/><Player teamId="b"/></root>')
    (folder / "BAR-ATH_tactical_atd.txt").write_text(
        "5:0.1,0.1,1.0;0.2,0.2,1.0:0.3,0.3,5.0\n"          # before kickoff
        "\n"
        "11186:0.4,0.5,1.0;NaN:NaN,NaN,NaN\n"              # first graded frame
        "110809:0.6,0.7,1.0;0.8,0.9,1.0:0.3,0.4,5.0\n")    # second half

    tracking = competition.read("BAR-ATH")
    assert tracking.home == "Barcelona"
    assert tracking.frames["frame"].tolist() == [11185, 110808]
    assert tracking.frames["period"].tolist() == [1, 2]
    assert tracking.frames["label"].tolist() == ["X", "X"]
    assert np.isnan(tracking.frames["ball_x"][0])
    assert tracking.players["track_id"].tolist() == ["1", "2", "1"]
    assert sorted(tracking.players["team"]) == ["A", "H", "H"]


@pytest.mark.parametrize("before, minimum, after", [
    ("HHHHAHHHH", 2, "HHHHHHHHH"),     # a touch inside a possession
    ("HHHHADDHH", 2, "HHHHHDDHH"),     # a touch just before the ball goes out
    ("DDAHHHH", 2, "DDAHHHH"),         # a short spell after a restart is a clear possessor
    ("HHHAAHHH", 2, "HHHAAHHH"),       # long enough: a real change of possession
    ("HHHAHAHAAAA", 2, "HHHHHHHAAAA"), # contested: stays with H until A has control
])
def test_fold_uncontrolled(before, minimum, after):
    from jaicodyn.data.labels import fold_uncontrolled
    assert "".join(fold_uncontrolled(table(before), minimum)) == after


def test_fold_uncontrolled_resets_at_period_start():
    from jaicodyn.data.labels import fold_uncontrolled
    assert "".join(fold_uncontrolled(table("HHHAHH", [1, 1, 1, 2, 2, 2]), 2)) == "HHHAHH"
