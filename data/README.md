# Data

Nothing here is committed except this file.

```
data/
  competition/   unpack the challenge bundle here: tactical/, broadcast/
  raw/           downloaded datasets, as published
  processed/     every match, in one schema
```

If the challenge bundle lives elsewhere, set `MTAH_DATA_DIR` to the folder holding `tactical/` and
`broadcast/`. Our code and the official starter tools both read it; unset, both use
`data/competition/`.
