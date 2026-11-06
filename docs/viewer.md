# Viewer

```bash
python -m scripts.viewer                  # pick a match in the page
python -m scripts.viewer idsse/J03WMX     # open one directly
```

It runs locally and works for every match. Video overlays need the competition files in `data/competition/`; other matches play on a pitch view.

To compare labels, save a `frame,label` CSV as `data/predictions/<source>/<id>/<name>.csv`. It appears as a timeline row. The competition match also shows the public truth.

| Control | Action |
|---|---|
| Space | play or pause |
| Left, Right | 1 frame (Shift: 1 s, Alt: 10 s) |
| `[`, `]` | previous or next possession |
| `-`, `=` | slower or faster |
| `v` | swap video and pitch |
| Scroll on the upper timeline | zoom at the cursor |
| Horizontal or Shift+scroll on the upper timeline | pan |
| Click a player | follow them |
| `?` | show all shortcuts |

The URL keeps your position when paused, so a link opens the same frame.
