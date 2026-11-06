"""Download locked files, resuming and verifying each one."""

import hashlib
import json
import time

import requests
from tqdm import tqdm

from . import paths

CHUNK = 1 << 20
RETRIES = 8
MAX_BACKOFF_S = 60
HEADERS = {"User-Agent": "jaicodyn-data/1.0"}


def load_lock():
    return json.loads(paths.LOCK.read_text("utf-8"))["datasets"]


def _matches_hash(path, entry):
    if "sha256" in entry:
        hasher, want = hashlib.sha256(), entry["sha256"]
    else:  # git blob hash
        hasher, want = hashlib.sha1(), entry["git_sha1"]
        hasher.update("blob {}\0".format(path.stat().st_size).encode())
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest() == want


def _fetch(url, part, size, bar):
    have = part.stat().st_size if part.exists() else 0
    headers = dict(HEADERS, Range="bytes={}-".format(have)) if have else HEADERS
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if have and response.status_code == 200:  # range ignored, restart
            have = 0
            bar.reset()
        elif response.status_code not in (200, 206):
            response.raise_for_status()
        with open(part, "ab" if have else "wb") as handle:
            for block in response.iter_content(CHUNK):
                handle.write(block)
                bar.update(len(block))
    if part.stat().st_size != size:
        raise IOError("short read")


def download(entry, source):
    target = paths.RAW / source / entry["path"]
    if target.exists() and target.stat().st_size == entry["size"]:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    initial = part.stat().st_size if part.exists() else 0
    with tqdm(total=entry["size"], initial=initial, unit="B", unit_scale=True, leave=False,
              desc=target.name[-40:]) as bar:
        for attempt in range(1, RETRIES + 1):
            try:
                _fetch(entry["url"], part, entry["size"], bar)
                break
            except (requests.RequestException, IOError):
                if attempt == RETRIES:
                    raise
                time.sleep(min(2 ** attempt, MAX_BACKOFF_S))

    if not _matches_hash(part, entry):
        part.unlink()
        raise IOError("{}: hash mismatch".format(target.name))
    part.replace(target)
