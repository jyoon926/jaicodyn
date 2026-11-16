# Jaicodyn

Team Jaicodyn's entry to the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) possession challenge.

## Reproduce the submission

You need the challenge data bundle (not in this repository), Python 3.11 (tested on 3.11.11) and
git. About two minutes end to end; nothing else is downloaded.

```bash
git clone --recurse-submodules https://github.com/jyoon926/jaicodyn.git
cd jaicodyn
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/MoreThanAHackData        # the folder that holds tactical/
python -m scripts.prepare_data --sources competition   # parses the BAR-ATH tracking, about 1 min
python -m scripts.decode BAR-ATH                       # writes jaicodyn.csv
python challenge/evaluation/validate_submission.py jaicodyn.csv
shasum -a 256 jaicodyn.csv
```

The file should hash to

```
d908ab74ce41f9610c12ea21c4cd3aa583d35f81cf8292b98b16916ea5556bf1  jaicodyn.csv
```

The submission uses only the competition match: its tracking and its automatic smart tagging. The
labelled open games below were used to choose the decoder's settings, which are written into the
code ([jaicodyn/decode/__init__.py](jaicodyn/decode/__init__.py)), so they are not needed to
regenerate the file. No model is trained at run time, and nothing is random.

## Setup

Run everything from the repository root.

### Competition data

The challenge data bundle is not in this repository. Download it, then point `MTAH_DATA_DIR` at the
folder that holds `tactical/` (the official starter tools read the same variable):

```bash
export MTAH_DATA_DIR=/path/to/your/hackathon/data   # the folder that holds tactical/
```

**Run this in every terminal before any command below**, or add it to `~/.zshrc` (or `~/.bashrc`)
so every new terminal has it. Check it with `echo $MTAH_DATA_DIR`. If it is unset, the code looks
in `data/competition/` instead, so unpacking the bundle there works too.

> **Why it matters:** the processed match loads without the bundle, but the smart tagging is read
> from it. A run without `MTAH_DATA_DIR` used to write a valid-looking but *different*
> `jaicodyn.csv`. `scripts.decode` now stops with an error instead; if you see "smart tagging not
> found", set the variable and run again.

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

### Full setup: tuning and evaluation

Only needed to rerun the experiments, not to regenerate the submission.

```bash
export MTAH_DATA_DIR=/path/to/your/hackathon/data   # every new terminal, unless in ~/.zshrc
git submodule update --init
pip install -r requirements.txt
python -m scripts.prepare_data     # downloads the labelled open games (about 4.6 GB), processes all
python -m scripts.degrade          # builds idsse_degraded and metrica_degraded from them
python -m pytest
python -m scripts.decode --split train validation   # scores the labelled games
python -m scripts.tune             # the settings search, on train
```

Set `MTAH_DATA_DIR` before `prepare_data`, or the competition match is skipped. Once every game
is processed, `data/raw/` (the original downloads) can be deleted to free space; `prepare_data`
fetches the same files again if needed, checked against the hashes in `config/datasets.lock.json`.

To watch a match with its tracking overlaid, run `python -m scripts.viewer`. See [docs/viewer.md](docs/viewer.md).

## Layout

```
challenge/   the official repository, as a submodule
config/      dataset lockfile, splits, degradation settings
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

`jaicodyn.csv`, `jaicodyn.pdf` or `jaicodyn.md`, and this repository. The CSV must regenerate byte
for byte (see above). Submission files and all match data are gitignored: the data is not
redistributable, and the CSV is rebuilt from it.
