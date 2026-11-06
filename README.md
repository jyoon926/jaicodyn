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

## Layout

```
challenge/   the official repository, as a submodule
config/      dataset lockfile, splits
data/        all data (gitignored)
docs/        our docs
jaicodyn/    our package
scripts/     command-line entry points
tests/       unit tests
```

## Data

See [data/README.md](data/README.md) and [docs/data.md](docs/data.md).

## Deliverables

`jaicodyn.csv`, `jaicodyn.pdf` or `jaicodyn.md`, and this repository. The CSV must regenerate byte for byte.
