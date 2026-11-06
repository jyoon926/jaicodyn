"""Every match, from any source, as one Match object. See docs/data.md."""

from .labels import absorb_short_spells
from .match import Match, available, load, split

__all__ = ["Match", "absorb_short_spells", "available", "load", "split"]
