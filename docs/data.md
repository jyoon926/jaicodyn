# Data

Every match, from any source, is one `Match`.

```python
from jaicodyn.data import load, split

match = load("idsse/J03WMX")
match.frames        # frame, period, ball_x, ball_y, label
match.players       # frame, team, track_id, x, y
match.possessions   # start, stop, team, period
match.video, match.homography, match.tags   # competition only, else None
match.fps, match.pitch, match.home, match.away

for match in split("train"):
    ...
```

Coordinates are 0-1 with the origin top left. `NaN` means not observed. Labels are `H`, `A`, `D`,
or `X` where none exists. Tables load on first use.

To merge short flips: `absorb_short_spells(match.frames, min_frames)`.

## Build

```bash
python -m scripts.prepare_data     # downloads about 4.6 GB
python -m scripts.check_data       # sanity-check every match
```

`config/splits.json` holds the splits. `python -m scripts.build_lock` re-pins upstream files.

## Sources

| Source | Matches | FPS | Labels |
|---|---|---|---|
| `competition` | 1, from `data/competition/` | 25 | `X` |
| `idsse` | 7, Bundesliga | 25 | `BallPossession`, `BallStatus` |
| `metrica` | 3, anonymised | 25 | derived from events |
| `skillcorner` | 20, A-League | 10 | phases of play |

To add a source, write `read(match_id) -> Tracking` and register it in `sources/__init__.py`.

Providers define possession differently, so choose any label conversion on other games only.

## Attribution

Cite these in the technical document.

- **IDSSE** (CC BY 4.0, [approved](https://github.com/metrica-sports/more-than-a-hack-2027/issues/16)):
  Bassek, Rein, Weber and Memmert. An integrated dataset of spatiotemporal and event data in elite
  soccer. *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
- **Metrica** (no formal licence, credit requested): https://github.com/metrica-sports/sample-data
- **SkillCorner** (MIT repository, credit requested): https://github.com/SkillCorner/opendata
