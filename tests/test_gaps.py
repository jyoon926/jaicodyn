import numpy as np
import pandas as pd

from jaicodyn.decode.gaps import (Context, GapParams, gap_runs, home_direction, mask_ball,
                                  periods, simulate)

FPS, PITCH = 25, (100.0, 50.0)


def feats(n=3000, periods_=None):
    t = np.arange(n)
    return pd.DataFrame({
        "frame": t, "period": periods_ if periods_ is not None else np.ones(n, int),
        "ball_x": 50 + 10 * np.sin(t / 50), "ball_y": np.full(n, 25.0),
        "near_H": np.full(n, 0.5), "near_A": np.full(n, 3.0), "speed_med": np.full(n, 2.0),
        "cx_H": np.full(n, 40.0), "cx_A": np.full(n, 60.0)})


def test_simulated_gaps_avoid_real_gaps_and_period_edges():
    f = feats(periods_=np.r_[np.ones(1500, int), np.full(1500, 2)])
    missing = np.zeros(len(f), dtype=bool)
    missing[700:900] = True
    p = GapParams(tries=500)
    sim = simulate(missing, periods(f), FPS, np.random.default_rng(0), p)
    assert sim.any() and not (sim & missing).any()
    for a, b in gap_runs(sim, periods(f), 1):
        assert f["period"].iloc[a] == f["period"].iloc[b - 1]
        assert 5 * FPS <= b - a <= 30 * FPS
        assert not missing[a - 50:b + 50].any()   # 2 s of visible context each side


def test_gap_runs_split_at_periods():
    hidden = np.ones(10, dtype=bool)
    assert gap_runs(hidden, [(0, 4), (4, 10)], 1) == [(0, 4), (4, 10)]
    assert gap_runs(hidden, [(0, 4), (4, 10)], 5) == [(4, 10)]


def test_mask_ball_blanks_only_hidden_rows():
    f = feats(10)
    hidden = np.zeros(10, dtype=bool)
    hidden[3:5] = True
    m = mask_ball(f, hidden)
    assert m["ball_x"].isna().tolist() == hidden.tolist()
    assert not m["speed_med"].isna().any() and not f["ball_x"].isna().any()


def test_home_direction_from_centroids():
    f = feats(10)
    assert (home_direction(f, PITCH, periods(f)) == 1).all()   # home sits left: attacks +x
    f["cx_H"], f["cx_A"] = 60.0, 40.0
    assert (home_direction(f, PITCH, periods(f)) == -1).all()


def test_context_reads_both_ends_of_a_gap():
    f = feats()
    hidden = np.zeros(len(f), dtype=bool)
    hidden[1000:1500] = True
    f.loc[hidden, ["ball_x", "near_H", "near_A"]] = np.nan
    f.loc[1500:, ["near_H", "near_A"]] = [3.0, 0.5]   # away nearest after the gap
    table, idx = Context(f, FPS, PITCH, None).rows(hidden, [(1000, 1500)], 100)
    assert len(table) == 5 and idx[0] == 1000
    assert table["who_before"].iloc[0] > 0.9 and table["who_after"].iloc[0] < -0.9
    assert np.isclose(table["gap_s"].iloc[0], 20)
