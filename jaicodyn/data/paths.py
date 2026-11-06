"""Repository paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("JAICODYN_DATA_DIR", ROOT / "data"))
# The challenge bundle (tactical/, broadcast/). MTAH_DATA_DIR is the official starter's variable,
# so one setting serves both their tools and ours.
COMPETITION = Path(os.environ.get("MTAH_DATA_DIR") or DATA / "competition").expanduser()
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LOCK = ROOT / "config" / "datasets.lock.json"
SPLITS = ROOT / "config" / "splits.json"
