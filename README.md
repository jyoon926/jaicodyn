# Jaicodyn

Team Jaicodyn's entry to the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) possession challenge.

## Setup

Run everything from the repository root.

### Competition data

The challenge data bundle is not in this repository. Download it, then point `MTAH_DATA_DIR` at the
folder that holds `tactical/` (the official starter tools read the same variable):

```bash
export MTAH_DATA_DIR=~/Downloads/MoreThanAHackData   # wherever you unpacked the bundle
```

Add that line to `~/.zshrc` (or `~/.bashrc`) so every new terminal has it. If it is unset, the code
looks in `data/competition/` instead, so unpacking the bundle there works too.

The folder should look like this:

```
MoreThanAHackData/
  tactical/
    BAR-ATH_tactical_atd.txt             required: tracking
    BAR-ATH_tactical_atd.xml             required: tracking metadata, team names
    BAR-ATH_tactical_smart_tagging.xml   required: set-piece tags, used by the decoder
    BAR-ATH_tactical_homography.csv      optional: only for the viewer
    BAR-ATH_tactical.mp4                 optional: only for the viewer (a -001 suffix is fine)
  broadcast/                             optional: not used by anything here
```

### Install and build

```bash
git submodule update --init
pip install -r requirements.txt
python -m scripts.prepare_data     # downloads the labelled open games (about 4.6 GB), processes all
python -m pytest
python -m scripts.decode BAR-ATH   # writes jaicodyn.csv
```

Set `MTAH_DATA_DIR` before `prepare_data`, or the competition match is skipped. Once every game
is processed, `data/raw/` (the original downloads) can be deleted to free space; `prepare_data`
fetches the same files again if needed, checked against the hashes in `config/datasets.lock.json`.

To watch a match with its tracking overlaid, run `python -m scripts.viewer`. See [docs/viewer.md](docs/viewer.md).

## Layout

```
challenge/   the official repository, as a submodule
config/      dataset lockfile, splits
data/        all data (gitignored)
docs/        our docs
jaicodyn/    our package: data, features, decode, viz
reports/     generated reports: tracking noise profile, baseline scores
scripts/     command-line entry points
tests/       unit tests
```

## Data

See [data/README.md](data/README.md) and [docs/data.md](docs/data.md).

## Deliverables

`jaicodyn.csv`, `jaicodyn.pdf` or `jaicodyn.md`, and this repository. The CSV must regenerate byte for byte.
