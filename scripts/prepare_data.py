"""Build every match into external/processed. Safe to re-run."""

import argparse
import sys

from jaicodyn.data.download import download, load_lock
from jaicodyn.data.match import is_built, write
from jaicodyn.data.sources import SOURCES, competition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--limit", type=int, help="matches per source")
    parser.add_argument("--force", action="store_true", help="rebuild finished matches")
    args = parser.parse_args()

    lock = load_lock()
    failed = []
    for source in args.sources:
        entries = lock.get(source, [])
        ids = competition.MATCHES if source == "competition" \
            else sorted({entry["match"] for entry in entries})
        for match_id in ids[:args.limit]:
            if source == "competition" and not competition.is_ready(match_id):
                print("{}/{}: skipped, not in data/competition/".format(source, match_id))
                continue
            if is_built(source, match_id) and not args.force:
                continue
            try:  # a failure costs one match, not the run
                for entry in entries:
                    if entry["match"] == match_id:
                        download(entry, source)
                write(source, match_id, SOURCES[source].read(match_id))
                print("{}/{}".format(source, match_id))
            except Exception as error:
                failed.append("{}/{}".format(source, match_id))
                print("{}/{}: FAILED, {!r}".format(source, match_id, error)[:200])

    if failed:
        sys.exit("failed, re-run to retry: {}".format(" ".join(failed)))


if __name__ == "__main__":
    main()
