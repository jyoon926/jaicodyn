"""Repository paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("JAICODYN_DATA_DIR", ROOT / "data"))
COMPETITION = DATA / "competition"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LOCK = ROOT / "config" / "datasets.lock.json"
SPLITS = ROOT / "config" / "splits.json"
