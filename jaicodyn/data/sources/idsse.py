"""IDSSE: DFL XML with per-frame BallPossession (1 home, 2 away) and BallStatus (1 alive)."""

import numpy as np
from lxml import etree

from .. import paths
from ..schema import Tracking, centred_to_unit, frames_table, players_table

FPS = 25
PERIODS = {"firstHalf": 1, "secondHalf": 2}


def _file(kind, match_id):
    return next((paths.RAW / "idsse").glob("{}_*DFL-MAT-{}.xml".format(kind, match_id)))


def _teams(path):
    teams = {team.get("Role"): team for team in etree.parse(str(path)).getroot().iter("Team")}
    return teams["home"], teams["guest"]


def _positions(path, sides):
    ball = {key: [] for key in ("frame", "period", "x", "y", "owner", "status")}
    people = {key: [] for key in ("frame", "team", "track", "x", "y")}
    pitch = period = team = person = None

    for event, element in etree.iterparse(str(path), events=("start", "end"),
                                          tag=("FrameSet", "Frame", "PitchSize")):
        if event == "start":
            if element.tag == "FrameSet":
                period = PERIODS.get(element.get("GameSection"))
                team = sides.get(element.get("TeamId"))
                person = element.get("PersonId")
            continue
        if element.tag == "PitchSize":
            pitch = float(element.get("X")), float(element.get("Y"))
        elif element.tag == "Frame" and period:
            if "BallStatus" in element.attrib:
                ball["frame"].append(int(element.get("N")))
                ball["period"].append(period)
                ball["x"].append(float(element.get("X")))
                ball["y"].append(float(element.get("Y")))
                ball["owner"].append(element.get("BallPossession"))
                ball["status"].append(element.get("BallStatus"))
            elif team:
                people["frame"].append(int(element.get("N")))
                people["team"].append(team)
                people["track"].append(person)
                people["x"].append(float(element.get("X")))
                people["y"].append(float(element.get("Y")))
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    return ball, people, pitch


def read(match_id):
    home, away = _teams(_file("DFL_02_01_matchinformation", match_id))
    sides = {home.get("TeamId"): "H", away.get("TeamId"): "A"}
    ball, people, pitch = _positions(_file("DFL_04_03_positions_raw_observed", match_id), sides)

    owner = np.array(ball["owner"], dtype=object)
    alive = np.array(ball["status"], dtype=object) == "1"
    label = np.select([alive & (owner == "1"), alive & (owner == "2")], ["H", "A"], "D")

    frames = frames_table(ball["frame"], ball["period"],
                          *centred_to_unit(ball["x"], ball["y"], pitch), label)
    players = players_table(people["frame"], people["team"], people["track"],
                            *centred_to_unit(people["x"], people["y"], pitch))
    players = players[players["frame"].isin(frames["frame"])].reset_index(drop=True)
    return Tracking(frames, players, FPS, pitch, home.get("TeamName"), away.get("TeamName"))
