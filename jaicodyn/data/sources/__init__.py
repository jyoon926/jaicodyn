"""Each source exposes read(match_id) -> Tracking."""

from . import competition, idsse, metrica, skillcorner

SOURCES = {"competition": competition, "idsse": idsse, "metrica": metrica,
           "skillcorner": skillcorner}
