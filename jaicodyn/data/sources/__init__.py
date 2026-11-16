"""Each source exposes read(match_id) -> Tracking."""

from . import competition, idsse, idsse_degraded, metrica, metrica_degraded, skillcorner

SOURCES = {"competition": competition, "idsse": idsse, "idsse_degraded": idsse_degraded,
           "metrica": metrica, "metrica_degraded": metrica_degraded, "skillcorner": skillcorner}
