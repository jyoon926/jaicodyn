# Jaicodyn

Team Jaicodyn's entry to the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) possession challenge.

## Setup

Run everything from the repository root.

```bash
git submodule update --init
pip install -r requirements.txt
python -m scripts.prepare_data
python -m pytest
```

To watch a match with its tracking overlaid, run `python -m scripts.viewer`. See [docs/viewer.md](docs/viewer.md).

## Layout

```
challenge/   the official repository, as a submodule
config/      dataset lockfile, splits
data/        all data (gitignored)
docs/        our docs
jaicodyn/    our package: data, features, viz
reports/     generated reports: tracking noise profile, baseline scores
scripts/     command-line entry points
tests/       unit tests
```

## Data

See [data/README.md](data/README.md) and [docs/data.md](docs/data.md).

## Deliverables

`jaicodyn.csv`, `jaicodyn.pdf` or `jaicodyn.md`, and this repository. The CSV must regenerate byte for byte.
