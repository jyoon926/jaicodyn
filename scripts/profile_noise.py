"""measure how noisy the automatic tracking is, before modelling anything.

    python -m scripts.profile_noise > reports/noise_report.txt
    # first run parses the ATD (~1-2 min) and caches it

Everything is measured on graded frames only, per half, so nothing straddles halftime.
The public-window section uses the 5-minute GT to *understand* where gaps fall -- never to set
constants (RULES.md).

The parsed arrays are cached under data/derived/ (gitignored with the rest of data/).
"""

import os

import numpy as np

from jaicodyn.data import paths

os.environ.setdefault("MTAH_DATA_DIR", str(paths.COMPETITION))  # before challenge.games reads it
from challenge.games import (GAME, GAMES, half_video_blocks, require,  # noqa: E402
                             source_frames)
from challenge.starter.baseline import team_of_each_track  # noqa: E402

DERIVED = paths.COMPETITION / "derived"  # where earlier runs left the parsed ATD cache

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_SLOTS = 46
N_COLS = N_SLOTS * 3 + 3
GAP_BUCKETS = [(1, 1), (2, 5), (6, 25), (26, 125), (126, 10 ** 9)]
GT_PUBLIC = os.path.join(paths.ROOT, "challenge", "evaluation", "gt",
                         "BAR-ATH_possession_gt_public.csv")


# ---------------------------------------------------------------- loading

def load_atd(raw_path, n_frames):
    """(players [n,46,3], ball [n,3]) float32, positions in metres, NaN where missing."""
    cache = os.path.join(DERIVED, "atd_tactical.npy")
    if os.path.isfile(cache):
        data = np.load(cache)
    else:
        print("parsing {} (cached afterwards) ...".format(os.path.basename(raw_path)))
        data = np.full((n_frames, N_COLS), np.nan, dtype=np.float32)
        with open(raw_path) as handle:
            for line in handle:
                vals = line.strip().replace(":", ",").replace(";", ",").split(",")
                if len(vals) != N_COLS + 1:
                    continue
                data[int(vals[0]) - 1] = np.array(vals[1:], dtype=np.float32)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, data)
    # players is [x, y, s] for each of 46 slots; ball is [x, y, s] for the ball
    players = data[:, :N_SLOTS * 3].reshape(-1, N_SLOTS, 3).copy()
    ball = data[:, N_SLOTS * 3:].copy()
    for arr in (players, ball):
        arr[..., 0] *= LENGTH
        arr[..., 1] *= WIDTH
    return players, ball


# ---------------------------------------------------------------- helpers

def true_runs(mask):
    """(starts, lengths) of runs of True."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    starts = np.where(d == 1)[0]
    return starts, np.where(d == -1)[0] - starts


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "n/a"
    return "  ".join("p{}={:.2f}".format(q, v) for q, v in zip(qs, np.percentile(x, qs)))


def header(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def nearest(players, team, pos, frames):
    """Distance and team of the nearest visible player to pos[i] at frames[i]."""
    pxy = players[frames, :, :2]                       # [k,46,2]
    d = np.linalg.norm(pxy - pos[:, None, :], axis=2)  # NaN for missing players
    d = np.where(np.isnan(d), np.inf, d)
    idx = np.argmin(d, axis=1)
    dist = d[np.arange(len(frames)), idx]
    tm = np.where(np.isfinite(dist), team[idx], -1)
    return np.where(np.isfinite(dist), dist, np.nan), tm


# ---------------------------------------------------------------- ball

def profile_ball(players, ball, team, blocks):
    header("BALL")
    vis = ~np.isnan(ball[:, 0])

    # speed-unit check: reported s vs finite-difference speed (m/s) on consecutive visible frames
    both = vis[1:] & vis[:-1]
    fd = np.full(len(ball), np.nan)
    fd[1:][both] = np.linalg.norm(ball[1:, :2][both] - ball[:-1, :2][both], axis=1) * FPS
    s = ball[:, 2]
    ok = np.isfinite(fd) & np.isfinite(s) & (fd > 1) & (fd < 40)
    if ok.sum():
        print("speed unit check (ball): median s / finite-diff(m/s) = {:.2f}  "
              "(~1 -> m/s, ~3.6 -> km/h)".format(np.median(s[ok] / fd[ok])))

    for half, a, b in blocks:
        sl = slice(a, b + 1)
        v = vis[sl]
        print("\nhalf {}: frames {}..{}  ball visible {:.1%}".format(half, a, b, v.mean()))
        per_min = [v[i:i + 60 * FPS].mean() for i in range(0, len(v), 60 * FPS)]
        print("  per-minute visibility: {}".format(pct(per_min, (0, 10, 50, 90, 100))))

        starts, lens = true_runs(~v)
        starts = starts + a
        print("  {} gaps, missing frames {}".format(len(lens), int(lens.sum())))
        print("  {:<12} {:>7} {:>12}".format("gap length", "count", "frame share"))
        for lo, hi in GAP_BUCKETS:
            m = (lens >= lo) & (lens <= hi)
            label = "{}-{}".format(lo, hi) if hi < 10 ** 9 else ">{}".format(lo - 1)
            print("  {:<12} {:>7} {:>11.1%}".format(label, int(m.sum()),
                                                    lens[m].sum() / max(lens.sum(), 1)))

        # both ends of the gap, interior gaps only
        inner = (starts > a) & (starts + lens <= b)
        g0, gl = starts[inner], lens[inner]
        last, nxt = g0 - 1, g0 + gl
        p_last, p_next = ball[last, :2], ball[nxt, :2]
        d_last, t_last = nearest(players, team, p_last, last)
        d_next, t_next = nearest(players, team, p_next, nxt)
        jump = np.linalg.norm(p_next - p_last, axis=1)
        implied = jump / ((gl + 1) / FPS)
        print("\n  gap endpoints, by gap length bucket:")
        print("  {:<10} {:>6} {:>10} {:>10} {:>10} {:>10} {:>9}".format(
            "bucket", "n", "dLast med", "dNext med", "jump med", "v_impl med", "sameTeam"))
        for lo, hi in GAP_BUCKETS:
            m = (gl >= lo) & (gl <= hi)
            if not m.any():
                continue
            label = "{}-{}".format(lo, hi) if hi < 10 ** 9 else ">{}".format(lo - 1)
            same = (t_last[m] == t_next[m]) & (t_last[m] >= 0)
            print("  {:<10} {:>6} {:>9.1f}m {:>9.1f}m {:>9.1f}m {:>8.1f}m/s {:>8.0%}".format(
                label, int(m.sum()), np.nanmedian(d_last[m]), np.nanmedian(d_next[m]),
                np.nanmedian(jump[m]), np.nanmedian(implied[m]), same.mean()))
        at_feet = d_last < 1.5
        print("  gaps starting with ball <1.5 m from a player: {:.0%}; of those, "
              "same team nearest at reappearance: {:.0%}".format(
                  at_feet.mean(), ((t_last == t_next) & at_feet).sum() / max(at_feet.sum(), 1)))

        # visible-ball quality
        idx = np.where(v)[0] + a
        fdv = fd[idx]
        print("\n  frame-to-frame ball speed (m/s): {}".format(pct(fdv)))
        print("  teleports (>40 m/s between consecutive frames): {}".format(
            int(np.sum(fdv > 40))))
        bx, by = ball[idx, 0], ball[idx, 1]
        off = (bx < 0) | (bx > LENGTH) | (by < 0) | (by > WIDTH)
        print("  visible ball off the pitch: {:.1%} of visible frames".format(off.mean()))
        st = np.zeros(len(ball), dtype=bool)
        st[idx] = fdv < 0.5
        _, slens = true_runs(st[sl])
        print("  stationary runs (<0.5 m/s) >= 1 s: {}   >= 3 s: {}".format(
            int(np.sum(slens >= FPS)), int(np.sum(slens >= 3 * FPS))))
        dn, _ = nearest(players, team, ball[idx, :2], idx)
        print("  nearest player to visible ball (m): {}".format(pct(dn)))
        print("  nearest > 2 m: {:.0%}   > 5 m: {:.0%}".format(
            np.nanmean(dn > 2), np.nanmean(dn > 5)))


# ---------------------------------------------------------------- players

def profile_players(players, team, blocks):
    header("PLAYERS")
    vis = ~np.isnan(players[:, :, 0])  # [n,46]

    sp = players[:, :, 2]
    both = vis[1:] & vis[:-1]
    fd = np.full(vis.shape, np.nan)
    diff = np.linalg.norm(players[1:, :, :2] - players[:-1, :, :2], axis=2) * FPS
    fd[1:][both] = diff[both]
    ok = np.isfinite(fd) & np.isfinite(sp) & (fd > 1) & (fd < 10)
    if ok.sum():
        print("speed unit check (players): median s / finite-diff(m/s) = {:.2f}".format(
            np.median(sp[ok] / fd[ok])))

    for half, a, b in blocks:
        sl = slice(a, b + 1)
        v = vis[sl]
        n = len(v)
        print("\nhalf {}".format(half))
        for t, name in ((0, "home"), (1, "away")):
            c = v[:, team == t].sum(axis=1)
            dist = {k: np.mean(c == k) for k in range(8, 14)}
            print("  {} visible per frame: mean {:.1f}  <9: {:.1%}  ==11: {:.1%}  >11: {:.1%}"
                  .format(name, c.mean(), np.mean(c < 9), dist[11], np.mean(c > 11)))

        lens_all = []
        for slot in range(N_SLOTS):
            _, lens = true_runs(v[:, slot])
            lens_all.extend(lens.tolist())
        lens_all = np.array(lens_all) / FPS
        print("  tracklets: {}  ({:.1f} per minute)  lifetime s: {}".format(
            len(lens_all), len(lens_all) / (n / FPS / 60), pct(lens_all)))
        print("  share of player-frames in tracklets < 5 s: {:.1%}".format(
            lens_all[lens_all < 5].sum() / max(lens_all.sum(), 1e-9)))

        jumps = np.nansum(fd[sl] > 12)
        print("  ID-switch proxy (slot moves > 12 m/s in one frame): {}  ({:.1f}/min)".format(
            int(jumps), jumps / (n / FPS / 60)))

        dup = 0
        step = 5000
        for i in range(a, b + 1, step):
            j = min(i + step, b + 1)
            for t in (0, 1):
                p = players[i:j][:, team == t, :2]
                d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=3)
                iu = np.triu_indices(p.shape[1], 1)
                dup += int(np.sum(np.any(d[:, iu[0], iu[1]] < 1.0, axis=1)))
        print("  frames with a same-team pair < 1 m apart (duplicate proxy): {:.1%}".format(
            dup / n))


# ---------------------------------------------------------------- public window

def profile_public_window(ball):
    header("PUBLIC WINDOW (understanding only -- do not tune on this)")
    if not os.path.isfile(GT_PUBLIC):
        print("no public GT found")
        return
    labels = np.loadtxt(GT_PUBLIC, delimiter=",", skiprows=1, dtype=str, usecols=1)
    graded = labels != "X"
    vis = ~np.isnan(ball[:, 0])
    for lab in ("H", "A", "D"):
        m = labels == lab
        print("  ball visible on GT {} frames: {:.1%}".format(lab, vis[m].mean()))
    idx = np.where(graded)[0]
    lab = labels[idx]
    change = idx[1:][lab[1:] != lab[:-1]]
    near = np.zeros(len(labels), dtype=bool)
    for f in change:
        near[max(f - FPS, 0):f + FPS + 1] = True
    near &= graded
    print("  ball visible within +/-1 s of a GT label change: {:.1%}  (elsewhere: {:.1%})"
          .format(vis[near].mean(), vis[graded & ~near].mean()))


if __name__ == "__main__":
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    print("slots: home {}  away {}".format(int(np.sum(team == 0)), int(np.sum(team == 1))))
    players, ball = load_atd(raw_path, source_frames("tactical"))
    blocks = half_video_blocks(GAME)
    profile_ball(players, ball, team, blocks)
    profile_players(players, team, blocks)
    profile_public_window(ball)
