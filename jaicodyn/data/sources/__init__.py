"""Each source exposes read(match_id) -> Tracking."""

from . import competition, idsse, idsse_degraded, metrica, skillcorner

SOURCES = {"competition": competition, "idsse": idsse, "idsse_degraded": idsse_degraded,
           "metrica": metrica, "skillcorner": skillcorner}
