"""Build idsse_degraded and metrica_degraded: games made to look like the competition tracking.

    python -m scripts.degrade --measure     # measure the settings into config/degrade.json
    python -m scripts.degrade --measure-noise   # only the ball-position noise (D below)
    python -m scripts.degrade               # build data/processed/{idsse,metrica}_degraded/*

--measure reads the competition match's own tracking and automatic tags, no labels:
    missing, gap_lengths_s  how much of the ball track is missing, in how long gaps
    dead_ratio              how much likelier the ball is missing while dead (A below); the tags
                            give a lower bound to check it against (B)
    line_weights            how much likelier a gap starts near or beyond a line (C)
    ball_noise_*            the slow error on the ball position (D): fitted so the idsse train
                            games, with it added, put the moving ball as far from its nearest
                            player as the competition match does
SkillCorner's own ratio (11x) is recorded but not used: its TV feed cuts to replays during
stoppages, which the competition's wide shot does not. See jaicodyn/data/degrade.py.
"""

import argparse
import json

import numpy as np
import pandas as pd

from jaicodyn.data import available, load, paths
from jaicodyn.data.degrade import CONFIG, degrade, drift, settings
from jaicodyn.data.match import write
from jaicodyn.data.sources import SOURCES
from jaicodyn.features import build
from jaicodyn.features.ball import true_runs
from jaicodyn.features.team import dense_positions
from scripts.build_features import resolve


SPEED_BINS = [0, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 9]  # player speed / match median
LINE_BANDS = [(-1000, 0), (0, 3), (3, 10), (10, 1000)]       # metres from the nearest line


def speed_ratio(feats, fps, window_s=3.0):
    s = feats["speed_med"].to_numpy(float)
    smooth = pd.Series(s).rolling(int(window_s * fps), center=True, min_periods=1).mean()
    return smooth.to_numpy() / np.nanmedian(s)


def line_distance(feats, pitch):
    """Metres from the ball to the nearest touchline or goal line; negative off the pitch."""
    length, width = pitch
    x, y = feats["ball_x"].to_numpy(float), feats["ball_y"].to_numpy(float)
    return np.minimum.reduce([x, length - x, y, width - y])


def dead_given_speed():
    """P(dead | player speed ratio bin) on the clean idsse train games, where every player is
    tracked as in the competition's wide shot."""
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse"]
    num, den = np.zeros(len(SPEED_BINS) - 1), np.zeros(len(SPEED_BINS) - 1)
    for match_id in train:
        m = load("idsse/{}".format(match_id))
        lab = m.frames.sort_values("frame")["label"].to_numpy()
        b = np.clip(np.digitize(speed_ratio(build(m), m.fps), SPEED_BINS) - 1, 0, len(num) - 1)
        for k in range(len(num)):
            sel = (lab != "X") & (b == k)
            den[k] += sel.sum()
            num[k] += (sel & (lab == "D")).sum()
    return num / np.maximum(den, 1)


def ratio_from_speed(comp, feats, missing):
    """A: Bayes on the competition match. P(dead | speed) from idsse gives the dead share of its
    missing and visible frames; P(missing | dead) / P(missing | live) follows."""
    p_dead = dead_given_speed()
    b = np.clip(np.digitize(speed_ratio(feats, comp.fps), SPEED_BINS) - 1, 0, len(p_dead) - 1)
    q = p_dead[b]
    dead, dead_missing, share = q.mean(), q[missing].mean(), missing.mean()
    return (dead_missing * share / dead) / ((1 - dead_missing) * share / (1 - dead))


def ratio_from_tags(comp, feats, missing):
    """B: missing share just before automatic set-piece restarts vs far from any tag. A lower
    bound: untagged stoppages (fouls, most throw-ins) sit in the 'far' frames."""
    fps, frame = comp.fps, feats["frame"].to_numpy()
    tags = comp.tags
    starts = tags.loc[tags["code"].str.upper() == "SET PIECES", "start"].to_numpy()
    near, far = np.zeros(len(feats), bool), np.ones(len(feats), bool)
    for s in starts:
        a, b = np.searchsorted(frame, [s - 10 * fps, s + 2 * fps])
        near[a:b] = True
        a, b = np.searchsorted(frame, [s - 30 * fps, s + 30 * fps])
        far[a:b] = False
    return missing[near].mean() / missing[far].mean()


def line_weights(comp, feats, missing):
    """C: how much likelier a gap starts with the ball in each band from the lines than the
    visible ball is to be there -- the ball leaving the wide shot."""
    d = line_distance(feats, comp.pitch)
    starts, lengths = true_runs(missing)
    last = starts[(lengths >= 5 * comp.fps) & (starts > 0)] - 1
    last = last[~np.isnan(d[last])]
    seen = d[~missing]
    return [round(float(np.mean((d[last] >= lo) & (d[last] < hi))
                        / max(np.mean((seen >= lo) & (seen < hi)), 1e-9)), 2)
            for lo, hi in LINE_BANDS]


def skillcorner_ratio():
    """For reference only: TV replays during stoppages make this far higher than the wide shot's."""
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["skillcorner"]
    dead, live = [], []
    for match_id in train:
        t = load("skillcorner/{}".format(match_id)).frames
        miss, lab = t["ball_x"].isna().to_numpy(), t["label"].to_numpy()
        dead.append(miss[lab == "D"].mean())
        live.append(miss[np.isin(lab, ["H", "A"])].mean())
    return np.mean(dead) / np.mean(live)


def merge_settings(comp):
    """(merges per second of a player being within 1.5 m of an opponent, merge lengths in s)."""
    from jaicodyn.data.degrade import close_rows
    from jaicodyn.features.ghost import merges
    frames = comp.frames.sort_values("frame")
    players = comp.players[comp.players["frame"].isin(frames["frame"])]
    events = merges(players, frames, comp.pitch, comp.fps)
    close_seconds = len(close_rows(players, comp.pitch)) / comp.fps
    lengths = np.sort([(end - start) / comp.fps for _, _, start, end, _ in events])
    return len(events) / close_seconds, lengths, len(events)


NOISE_BANDS = [(1, 3), (3, 6)]   # m/s: ball rolling at a player's feet or passed along the ground
NOISE_QS = (10, 25, 50)
NOISE_SIGMA_M = np.round(np.arange(0.3, 0.95, 0.05), 2)
NOISE_TAU_S = (0.25, 0.5, 1, 2, 4, 8)   # Gaussian smoothing width, s
ROUGH_LAGS = (1, 4, 8, 32)
ROUGH_SLACK = 0.02  # see noise_settings


def ball_and_players(match, live_only):
    """(ball [n,2] m, every player [n,k,2] m, frames to use) for the noise fit."""
    length, width = match.pitch
    f = match.frames.sort_values("frame", ignore_index=True)
    index = pd.Index(f["frame"])
    pos = np.concatenate([dense_positions(match.players, index, t, match.pitch) for t in "HA"],
                         axis=1)
    ball = np.column_stack([f["ball_x"].to_numpy(float) * length,
                            f["ball_y"].to_numpy(float) * width])
    use = np.isin(f["label"].to_numpy(), ["H", "A"]) if live_only else np.ones(len(f), bool)
    return ball, pos, use, f["period"].to_numpy()


def nearest_quantiles(ball, pos, use, fps):
    """NOISE_QS quantiles of the nearest player's distance, per ball-speed band in NOISE_BANDS.

    Ball speed is a 5-frame median of the frame-to-frame speed, taken on the noisy ball: it is
    what the competition match gives too.
    """
    v = np.r_[np.nan, np.hypot(*np.diff(ball, axis=0).T) * fps]
    v = pd.Series(v).rolling(5, center=True, min_periods=3).median().to_numpy()
    d = np.linalg.norm(pos - ball[:, None], axis=2)
    near = np.min(np.where(np.isnan(d), np.inf, d), axis=1)
    out = []
    for lo, hi in NOISE_BANDS:
        sel = use & (v >= lo) & (v < hi) & np.isfinite(near)
        out += list(np.percentile(near[sel], NOISE_QS))
    return np.array(out)


def roughness(ball, use):
    """Mean |second difference| of the ball, metres, at ROUGH_LAGS frames."""
    out = []
    for k in ROUGH_LAGS:
        d = ball[2 * k:] - 2 * ball[k:-k] + ball[:-2 * k]
        ok = use[2 * k:] & use[:-2 * k] & use[k:-k]
        out.append(float(np.nanmean(np.abs(d[ok]))))
    return np.array(out)


def noise_settings(comp):
    """D: (sigma_m, tau_s, competition quantiles, clean idsse, fitted idsse).

    Competition: every visible frame (no labels; the moving-ball bands leave out most dead ball).
    idsse train: live frames, with drift(sigma, tau) added to the ball, the same noise draw for
    every setting. For each tau the sigma with the least squared quantile error is found, and of
    the taus whose noise leaves the ball's roughness within ROUGH_SLACK of clean idsse at every
    lag, the best fit is kept: the competition's ball track is no rougher than idsse's at any lag
    up to 1.3 s, so the noise must not add roughness. sigma comes out 0.55-0.6 m at every tau; the
    roughness rule rules out tau under 1 s, and the fit is flat above it.
    """
    target = nearest_quantiles(*ball_and_players(comp, False)[:3], comp.fps)
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse"]
    games = [ball_and_players(load("idsse/{}".format(i)), True) for i in train]
    clean = np.mean([nearest_quantiles(b, p, u, 25) for b, p, u, _ in games], axis=0)
    smooth = np.mean([roughness(b, u) for b, _, u, _ in games], axis=0)
    best = None   # (loss, sigma, tau, quantiles)
    for tau in NOISE_TAU_S:
        unit = [drift(len(b), 1.0, tau, 25, np.random.default_rng(k), period)
                for k, (b, _, _, period) in enumerate(games)]
        fit = None
        for sigma in NOISE_SIGMA_M:
            got = np.mean([nearest_quantiles(b + sigma * e, p, u, 25)
                           for (b, p, u, _), e in zip(games, unit)], axis=0)
            loss = float(np.sum((got - target) ** 2))
            if fit is None or loss < fit[0]:
                fit = (loss, float(sigma), tau, got)
        rough = np.mean([roughness(b + fit[1] * e, u) for (b, _, u, _), e in zip(games, unit)],
                        axis=0)
        if np.all(rough <= smooth * (1 + ROUGH_SLACK)) and (best is None or fit[0] < best[0]):
            best = fit
    _, sigma, tau, got = best
    unit = [drift(len(b), sigma, tau, 25, np.random.default_rng(k), period)
            for k, (b, _, _, period) in enumerate(games)]
    rough = {"lags_frames": list(ROUGH_LAGS),
             "competition": roughness(*ball_and_players(comp, False)[::2]),
             "idsse_clean": np.mean([roughness(b, u) for b, _, u, _ in games], axis=0),
             "idsse_with_noise": np.mean([roughness(b + e, u)
                                          for (b, _, u, _), e in zip(games, unit)], axis=0)}
    return sigma, tau, target, clean, got, rough


def measure_noise(config):
    """Fill config's ball_noise_* (D) from the competition match; returns config."""
    sigma, tau, target, clean, fitted, rough = noise_settings(load(resolve("BAR-ATH")))
    config["ball_noise_sigma_m"] = sigma
    config["ball_noise_tau_s"] = tau
    r = lambda a: [round(float(v), 3) for v in a]  # noqa: E731
    config.setdefault("checks", {})["ball_noise_nearest_m"] = {
        "bands_mps": NOISE_BANDS, "quantiles": list(NOISE_QS),
        "competition": r(target), "idsse_clean": r(clean), "idsse_with_noise": r(fitted)}
    config["checks"]["ball_noise_roughness_m"] = {
        k: (v if k == "lags_frames" else r(v)) for k, v in rough.items()}
    print("ball noise: sigma {:.2f} m per axis, tau {} s. Nearest player (m), q{} per band {}:\n"
          "  competition {}\n  idsse clean {}\n  idsse noisy {}".format(
              sigma, tau, list(NOISE_QS), NOISE_BANDS, r(target), r(clean), r(fitted)))
    print("  ball roughness, mean |2nd difference| (m) at lags {}: competition {}, idsse clean {}, "
          "noisy {}".format(list(ROUGH_LAGS), r(rough["competition"]), r(rough["idsse_clean"]),
                            r(rough["idsse_with_noise"])))
    return config


def measure():
    comp = load(resolve("BAR-ATH"))
    feats = build(comp)
    missing = feats["ball_x"].isna().to_numpy()
    _, lengths = true_runs(missing)
    gaps = lengths / comp.fps
    long = np.sort(gaps[gaps >= 5])
    a = ratio_from_speed(comp, feats, missing)
    b = ratio_from_tags(comp, feats, missing)
    w = line_weights(comp, feats, missing)
    config = {
        "missing": round(float(missing.mean()), 4),
        "dead_ratio": round(float(a), 2),
        "line_bands_m": LINE_BANDS,
        "line_weights": w,
        "stoppage_spill_s": None,  # no cut: at this ratio cutting stoppage gaps changes nothing
        "gap_lengths_s": [round(float(g), 2) for g in long],
        "checks": {"dead_ratio_from_tags_lower_bound": round(float(b), 2),
                   "dead_ratio_skillcorner_not_used": round(float(skillcorner_ratio()), 2)},
        "sources": "all from competition/BAR-ATH tracking and its automatic smart tagging, no "
                   "labels; dead_ratio by Bayes with P(dead | player speed) from the idsse "
                   "train labels. SkillCorner's ratio is TV replays, so it is not used.",
    }
    config["line_weights_applied"] = calibrate(config)
    measure_noise(config)
    hazard, lengths, n = merge_settings(comp)
    config["merge_hazard_per_s"] = round(float(hazard), 4)
    config["merge_lengths_s"] = [round(float(v), 2) for v in lengths]
    print("cross-team merges: {} on the competition match, {:.3f} per second of a player within "
          "1.5 m of an opponent, median length {:.1f}s".format(n, hazard, np.median(lengths)))
    CONFIG.write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    print("wrote {}: missing {:.1%}, dead_ratio {:.2f} (tags lower bound {:.2f}, SkillCorner "
          "{:.1f} not used), line weights {}, {} gap lengths (median {:.1f}s)".format(
              CONFIG, config["missing"], a, b, config["checks"]["dead_ratio_skillcorner_not_used"],
              w, len(long), np.median(long)))


def start_weights(tracks, bands):
    """Gap-start weight per band (gap starts vs visible frames), over (frames, raw ball) pairs."""
    starts_in, seen_in = np.zeros(len(bands)), np.zeros(len(bands))
    for frames, d in tracks:
        miss = frames["ball_x"].isna().to_numpy()
        s, n = true_runs(miss)
        last = s[(n >= 5 * 25) & (s > 0)] - 1
        for k, (lo, hi) in enumerate(bands):
            starts_in[k] += np.sum((d[last] >= lo) & (d[last] < hi))
            seen_in[k] += np.sum((d[~miss] >= lo) & (d[~miss] < hi))
    return (starts_in / starts_in.sum()) / np.maximum(seen_in / seen_in.sum(), 1e-9)


def calibrate(c, rounds=3):
    """Weights that make degraded idsse train games reproduce the competition's gap-start pattern.

    The measured weights already contain the stoppages-happen-at-lines effect, which placing gaps
    over idsse's own stoppages adds again. Matched on shape -- each band's weight relative to
    mid-pitch -- since the visible ball sits in different places in idsse than in the competition,
    so the absolute weights cannot all match at once.
    """
    train = json.loads(paths.SPLITS.read_text("utf-8"))["splits"]["train"]["idsse"]
    matches = [load("idsse/{}".format(i)) for i in train]
    target = np.asarray(c["line_weights"])
    applied = target.copy()
    for _ in range(rounds):
        tracks = []
        for m in matches:
            frames = degrade(m, dict(c, line_weights=list(applied))).frames
            raw = m.frames.sort_values("frame", ignore_index=True)
            length, width = m.pitch
            x, y = raw["ball_x"].to_numpy() * length, raw["ball_y"].to_numpy() * width
            tracks.append((frames, np.minimum.reduce([x, length - x, y, width - y])))
        achieved = start_weights(tracks, c["line_bands_m"])
        shape, want = achieved / achieved[-1], target / target[-1]  # relative to mid-pitch
        print("  calibration: starts vs mid-pitch {} target {}".format(np.round(shape, 2),
                                                                     np.round(want, 2)))
        applied = applied * want / np.maximum(shape, 1e-9)
    return [round(float(v), 3) for v in applied]


def report(match, c):
    t = match.frames.sort_values("frame")
    miss, lab = t["ball_x"].isna().to_numpy(), t["label"].to_numpy()
    graded = lab != "X"
    dead, live = miss[graded & (lab == "D")].mean(), miss[graded & np.isin(lab, ["H", "A"])].mean()
    _, lengths = true_runs(miss)
    gaps = lengths / match.fps
    print("{:<26} missing {:5.1%} (target {:.1%})  dead {:4.0%}  live {:4.0%}  ratio {:5.1f} "
          "(target {:.1f})  gaps {:>3}, median {:4.1f}s".format(
              match.key, miss[graded].mean(), c["missing"], dead, live, dead / max(live, 1e-9),
              c["dead_ratio"], len(gaps), np.median(gaps) if len(gaps) else 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true", help="measure into config/degrade.json")
    parser.add_argument("--measure-noise", action="store_true",
                        help="measure only the ball-position noise into config/degrade.json")
    args = parser.parse_args()
    if args.measure_noise:
        config = measure_noise(settings())
        CONFIG.write_text(json.dumps(config, indent=1) + "\n", "utf-8")
        print("wrote {}".format(CONFIG))
        return
    if args.measure:
        measure()
        return
    c = settings()
    for match in [m for m in available() if m.source in ("idsse", "metrica")]:
        target = match.source + "_degraded"
        write(target, match.id, SOURCES[target].read(match.id))
        report(load("{}/{}".format(target, match.id)), c)


if __name__ == "__main__":
    main()
