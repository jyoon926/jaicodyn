"""Rebuild config/datasets.lock.json from upstream APIs, downloading no data."""

import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

HF_REPO = "pysport/idsse-data"
HF_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
SKILLCORNER_REVISION = "4340d274572876239c154c90bc507a9b3250a656"
METRICA_REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"
HEADERS = {"User-Agent": "jaicodyn-data/1.0"}


def get_json(url):
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def github_tree(repo, revision):
    url = "https://api.github.com/repos/{}/git/trees/{}?recursive=1".format(repo, revision)
    tree = get_json(url)
    if tree["truncated"]:
        raise RuntimeError("{}: tree listing truncated".format(repo))
    return [item for item in tree["tree"] if item["type"] == "blob"]


def lfs_pointer(repo, revision, path):
    url = "https://raw.githubusercontent.com/{}/{}/{}".format(repo, revision, path)
    text = requests.get(url, headers=HEADERS, timeout=60).text
    oid, size = re.search(r"oid sha256:(\w+)\s+size (\d+)", text).groups()
    return oid, int(size)


def idsse():
    url = "https://huggingface.co/api/datasets/{}/tree/{}?recursive=true".format(
        HF_REPO, HF_REVISION)
    entries = []
    for item in get_json(url):
        path = item["path"]
        if item["type"] != "file" or not path.endswith(".xml"):
            continue
        entry = {"match": path.rsplit("-", 1)[1][:-4], "path": path, "size": item["size"],
                 "url": "https://huggingface.co/datasets/{}/resolve/{}/{}".format(
                     HF_REPO, HF_REVISION, path)}
        if item.get("lfs"):
            entry["sha256"] = item["lfs"]["oid"]
        else:
            entry["git_sha1"] = item["oid"]
        entries.append(entry)
    return entries


def skillcorner():
    repo = "SkillCorner/opendata"
    keep = re.compile(
        r"data/matches/(\d+)/\d+_(match\.json|phases_of_play\.csv|tracking_extrapolated\.jsonl)$")
    entries = []
    for item in github_tree(repo, SKILLCORNER_REVISION):
        hit = keep.match(item["path"])
        if not hit:
            continue
        entry = {"match": hit.group(1), "path": item["path"][len("data/"):]}
        if item["size"] < 512:  # LFS pointer
            oid, size = lfs_pointer(repo, SKILLCORNER_REVISION, item["path"])
            entry.update(size=size, sha256=oid,
                         url="https://media.githubusercontent.com/media/{}/{}/{}".format(
                             repo, SKILLCORNER_REVISION, item["path"]))
        else:
            entry.update(size=item["size"], git_sha1=item["sha"],
                         url="https://raw.githubusercontent.com/{}/{}/{}".format(
                             repo, SKILLCORNER_REVISION, item["path"]))
        entries.append(entry)
    return entries


def metrica():
    repo = "metrica-sports/sample-data"
    entries = []
    for item in github_tree(repo, METRICA_REVISION):
        hit = re.match(r"data/(Sample_Game_(\d+)/.+)$", item["path"])
        if hit:
            entries.append({"match": "game{}".format(hit.group(2)), "path": hit.group(1),
                            "size": item["size"], "git_sha1": item["sha"],
                            "url": "https://raw.githubusercontent.com/{}/{}/{}".format(
                                repo, METRICA_REVISION, item["path"])})
    return entries


def main():
    lock = {"revisions": {"idsse": HF_REVISION, "metrica": METRICA_REVISION,
                          "skillcorner": SKILLCORNER_REVISION},
            "datasets": {"idsse": idsse(), "metrica": metrica(), "skillcorner": skillcorner()}}
    for table in lock["datasets"].values():
        table.sort(key=lambda entry: entry["path"])
    path = ROOT / "config" / "datasets.lock.json"
    path.write_text(json.dumps(lock, indent=1) + "\n", "utf-8")
    for name, table in lock["datasets"].items():
        print("{:<12} {:>3} files {:>6.2f} GB".format(
            name, len(table), sum(entry["size"] for entry in table) / 1e9))


if __name__ == "__main__":
    main()
