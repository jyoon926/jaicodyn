"""idsse games with the ball hidden to look like the competition tracking. See ../degrade.py.

Built from the processed idsse match, so run scripts.prepare_data for idsse first.
"""

from ..degrade import degrade
from ..match import load


def read(match_id):
    return degrade(load("idsse/{}".format(match_id)))
