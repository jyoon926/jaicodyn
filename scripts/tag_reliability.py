"""Label-free accuracy of each smart-tagging phase code as an indicator of the team in possession.

    python -m scripts.tag_reliability

Agreement between a code's team (flipped for the "defending" mirrors) and the decoder's team on
the competition match's visible frames, and the accuracy p_t it implies if tag and decoder erred
independently: agreement = p_d p_t + (1 - p_d)(1 - p_t), p_d the decoder's visible-frame team
accuracy on the degraded idsse train games. Upper bounds -- the tags come from the same tracking,
so errors are shared. The values feed PHASES in jaicodyn/decode/tags.py.
"""

import json

import numpy as np

from jaicodyn.data import load, paths
from jaicodyn.data.labels import target_labels
from jaicodyn.decode import decode
from jaicodyn.decode.tags import PHASES, set_piece_restart_p
from jaicodyn.features import build
from scripts.build_features import resolve


def decoder_visible_accuracy():
    ids = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse_degraded"]
    ok = n = 0
    for i in ids:
        m = load("idsse_degraded/{}".format(i))
        f = build(m)
        t = target_labels(m, "strict")
        pr = decode(f, m.fps, m.pitch, m.tags)
        sel = f["ball_x"].notna().to_numpy() & np.isin(pr, ["H", "A"]) & np.isin(t, ["H", "A"])
        ok += np.sum(pr[sel] == t[sel])
        n += sel.sum()
    return ok / n


def main():
    p_d = decoder_visible_accuracy()
    m = load(resolve("BAR-ATH"))
    f = build(m)
    pr = decode(f, m.fps, m.pitch, m.tags, restart_p_home=set_piece_restart_p(f, m.tags, m.fps))
    frame, visible = f["frame"].to_numpy(), f["ball_x"].notna().to_numpy()
    print("decoder visible-frame team accuracy (degraded idsse train): {:.3f}\n".format(p_d))
    print("{:<30} {:>7} {:>10} {:>9} {:>9}".format("code", "frames", "in gaps", "agree", "p_t"))
    codes = m.tags["code"].str.upper()
    for code, (flip, _) in PHASES.items():
        rows = m.tags[codes == code]
        team = np.full(len(f), "", dtype=object)
        for s, e, t in zip(rows["start"], rows["stop"], rows["team"]):
            if t in ("H", "A"):
                a, b = np.searchsorted(frame, [s, e])
                team[a:b] = ({"H": "A", "A": "H"}[t] if flip else t)
        tagged = team != ""
        sel = tagged & visible & np.isin(pr, ["H", "A"])
        agree = np.mean(pr[sel] == team[sel]) if sel.any() else np.nan
        p_t = (agree - (1 - p_d)) / (2 * p_d - 1)
        print("{:<30} {:6.1f}% {:9.1f}% {:9.3f} {:9.2f}".format(
            code, 100 * tagged.mean(), 100 * (tagged & ~visible).mean(), agree, p_t))


if __name__ == "__main__":
    main()
