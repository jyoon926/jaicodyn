"""Metrica games degraded like the competition tracking. See ../degrade.py.

Metrica already blanks the ball on nearly every dead frame, so only live ball is hidden, to the
competition's live rate; its dead ball stays more often missing than the competition's (the
dead-to-live ratio comes out higher than measured). Merges and ball noise are added as for idsse.
Built from the processed Metrica match, so run scripts.prepare_data for metrica first.
"""

from ..degrade import degrade
from ..match import load


def read(match_id):
    return degrade(load("metrica/{}".format(match_id)))
