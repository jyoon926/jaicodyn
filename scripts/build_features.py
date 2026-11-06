"""Build team features for matches, and print a summary per match.

    python -m scripts.build_features                     # every processed match
    python -m scripts.build_features BAR-ATH             # one match, by id ...
    python -m scripts.build_features idsse/J03WMX        # ... or by source/id
    python -m scripts.build_features BAR-ATH --force     # rebuild even if cached

The competition match is read from MTAH_DATA_DIR (the folder holding tactical/ and broadcast/),
or data/competition/ if unset, and is processed on first use. Other sources come from
`python -m scripts.prepare_data`.

ball: share of frames with a ball position (observed or bridged).
over11: share of frames with more than 11 players on a team, before and after dedup.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from jaicodyn.data import available, load, paths
from jaicodyn.data.match import is_built, write
from jaicodyn.data.sources import SOURCES, competition
from jaicodyn.features import build, feature_path
from jaicodyn.features.ball import BRIDGED, OBSERVED, SPIKE
from jaicodyn.features.team import TEAMS, dense_positions


def resolve(name):
    """A match key ("source/id") for a key or a bare id, processing the competition match if needed."""
    if "/" in name:
        source, match_id = name.split("/", 1)
    else:
        found = [m.source for m in available() if m.id == name]
        if not found and name in competition.MATCHES:
            found = ["competition"]
        if len(found) != 1:
            sys.exit("{}: {}; have {}".format(
                name, "ambiguous, give source/id" if found else "no such match",
                " ".join(m.key for m in available()) or "none processed"))
        source, match_id = found[0], name
    if source not in SOURCES:
        sys.exit("{}: no such source; have {}".format(name, " ".join(sorted(SOURCES))))
    if not is_built(source, match_id):
        if source != "competition":
            sys.exit("{}/{}: not processed; run python -m scripts.prepare_data --sources {}".format(
                source, match_id, source))
        if not competition.is_ready(match_id):
            sys.exit("{}: no tracking under {}; set MTAH_DATA_DIR to the folder holding "
                     "tactical/".format(match_id, paths.COMPETITION))
        print("processing {}/{} from {} ...".format(source, match_id, paths.COMPETITION))
        write(source, match_id, competition.read(match_id))
    return "{}/{}".format(source, match_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matches", nargs="*", help="match ids or source/id; default all processed")
    parser.add_argument("--force", action="store_true", help="rebuild cached features")
    args = parser.parse_args()

    matches = [load(resolve(name)) for name in args.matches] or available()
    if not matches:
        sys.exit("no processed matches; give one, such as BAR-ATH, or run scripts.prepare_data")

    print("{:<22} {:>7} {:>6} {:>6} {:>6} {:>12} {:>12}".format(
        "match", "frames", "ball", "spike", "bridge", "over11 H", "over11 A"))
    for match in matches:
        table = build(match, force=args.force)
        src = table["ball_src"].to_numpy()
        before = {}
        for t in TEAMS:
            pos = dense_positions(match.players, pd.Index(table["frame"]), t, match.pitch)
            before[t] = np.mean(np.sum(~np.isnan(pos[:, :, 0]), axis=1) > 11)
        over = ["{:.1%}>{:.1%}".format(before[t], np.mean(table["n_" + t] > 11)) for t in TEAMS]
        print("{:<22} {:>7} {:>6.1%} {:>6.2%} {:>6.2%} {:>12} {:>12}".format(
            match.key, len(table), np.mean((src == OBSERVED) | (src == BRIDGED)),
            np.mean(src == SPIKE), np.mean(src == BRIDGED), *over))
    print("\nfeatures in {}".format(feature_path(matches[0]).parent.parent))


if __name__ == "__main__":
    main()
