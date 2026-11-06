"""Metrica: games 1-2 are CSV, game 3 is EPTS. Labels are derived from events."""

import csv
import json

import numpy as np
import pandas as pd
from lxml import etree

from .. import paths
from ..schema import Tracking, frames_table, players_table

FPS = 25
PITCH = (105.0, 68.0)
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}


def labels_from_events(events, frames, periods):
    """The last on-ball team holds; dead from the end of a stop until the next on-ball action."""
    frames, periods = np.asarray(frames), np.asarray(periods)
    label = np.full(len(frames), "D", dtype=object)

    for period in np.unique(periods):
        rows = np.flatnonzero(periods == period)
        window = frames[rows]
        marks = [(window[0], "D")]
        last = window[0]
        for event in (e for e in events if e["period"] == period):
            since, state = marks[-1]
            if state == "D" and event["start"] < since:
                continue  # logged inside a stop, not a restart
            at = max(event["start"], since)
            stop = max(at, event["end"])
            if event["type"] in ON_BALL and event["team"]:
                marks.append((at, event["team"]))
                last = max(last, stop)
            if _stops_play(event):
                marks.append((stop, "D"))
                last = max(last, stop)
        marks.append((last + 1, "D"))
        for (start, value), (stop, _) in zip(marks, marks[1:]):
            label[rows[(window >= start) & (window < stop)]] = value
    return label


def _stops_play(event):
    if event["type"] == "SHOT":
        return "GOAL" in event["subtype"] or "OUT" in event["subtype"]
    return event["type"] in STOPS


def _csv_team(path, team):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader), next(reader)
        header = next(reader)
    table = pd.read_csv(path, skiprows=3, header=None, na_values=["NaN"])
    table = table[table[0].notna()]
    frame = table[1].values.astype(int)

    parts = []
    for index, name in enumerate(header):
        if name.startswith("Player"):
            part = pd.DataFrame({"frame": frame, "x": table[index].values,
                                 "y": table[index + 1].values})
            parts.append(part[part["x"].notna()].assign(team=team, track=team + name[6:]))
    ball = header.index("Ball")
    return frame, table[0].values.astype(int), table[ball].values, table[ball + 1].values, \
        pd.concat(parts, ignore_index=True)


def _csv_events(path):
    with open(path, newline="") as handle:
        return sorted(({"team": {"Home": "H", "Away": "A"}.get(row["Team"]), "type": row["Type"],
                        "subtype": row["Subtype"] or "", "period": int(row["Period"]),
                        "start": int(row["Start Frame"]), "end": int(row["End Frame"])}
                       for row in csv.DictReader(handle)), key=lambda e: e["start"])


def _csv_game(folder):
    game = folder.name
    frame, period, ball_x, ball_y, home = _csv_team(
        folder / "{}_RawTrackingData_Home_Team.csv".format(game), "H")
    away = _csv_team(folder / "{}_RawTrackingData_Away_Team.csv".format(game), "A")[4]
    events = _csv_events(folder / "{}_RawEventsData.csv".format(game))
    return frame, period, ball_x, ball_y, pd.concat([home, away], ignore_index=True), events, \
        ("Home", "Away")


def _subtype(value):
    values = value if isinstance(value, list) else [value] if value else []
    return "-".join(item["name"] for item in values)


def _epts_game(folder):
    stem = folder.name + "_"
    root = etree.parse(str(folder / (stem + "metadata.xml"))).getroot()
    etree.cleanup_namespaces(root)
    for element in root.iter(etree.Element):
        element.tag = etree.QName(element).localname

    teams = list(root.iter("Team"))
    sides = {teams[0].get("id"): "H", teams[1].get("id"): "A"}  # EPTS lists home first
    team_of = {player.get("id"): sides[player.get("teamId")] for player in root.iter("Player")}
    owner = {channel.get("id"): channel.get("playerId") for channel in root.iter("PlayerChannel")}
    layouts = [(int(spec.get("startFrame")), int(spec.get("endFrame")),
                [owner[ref.get("playerChannelId")] for ref in spec.iter("PlayerChannelRef")
                 if ref.get("playerChannelId").endswith("_x")])
               for spec in root.iter("DataFormatSpecification")]
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}

    frame, ball_x, ball_y = [], [], []
    rows = []
    with open(folder / (stem + "tracking.txt")) as handle:
        for line in handle:
            number, bodies, ball = line.strip().split(":")
            number = int(number)
            layout = next(order for start, stop, order in layouts if start <= number <= stop)
            bx, by = ball.split(",")[:2]
            frame.append(number)
            ball_x.append(float(bx))
            ball_y.append(float(by))
            for player, body in zip(layout, bodies.split(";")):
                x, y = body.split(",")[:2]
                if x != "NaN":
                    rows.append((number, float(x), float(y), team_of[player], player))

    people = pd.DataFrame(rows, columns=["frame", "x", "y", "team", "track"])
    period = np.where(np.array(frame) >= int(params["second_half_start"]), 2, 1)
    raw = json.loads((folder / (stem + "events.json")).read_text("utf-8"))["data"]
    events = sorted(({"team": sides.get((e.get("team") or {}).get("id")), "type": e["type"]["name"],
                      "subtype": _subtype(e.get("subtypes")), "period": int(e["period"]),
                      "start": int(e["start"]["frame"]),
                      "end": int(e["end"]["frame"] or e["start"]["frame"])}
                     for e in raw), key=lambda e: e["start"])
    names = tuple(team.findtext("Name") for team in teams)
    return frame, period, ball_x, ball_y, people, events, names


def read(match_id):
    number = match_id.replace("game", "")
    folder = paths.RAW / "metrica" / "Sample_Game_{}".format(number)
    frame, period, ball_x, ball_y, people, events, names = \
        (_epts_game if number == "3" else _csv_game)(folder)

    label = labels_from_events(events, frame, period)
    frames = frames_table(frame, period, ball_x, ball_y, label)
    players = players_table(people["frame"], people["team"], people["track"],
                            people["x"], people["y"])
    return Tracking(frames, players, FPS, PITCH, *names)
