import numpy as np
import pandas as pd

from jaicodyn.decode import A, D, H, Params, decode, switch_costs, viterbi

FPS, PITCH = 25, (100.0, 50.0)


def rolling(n, start=50.0):
    """A ball in play: 2 m/s. A still ball beside a player is a dead-ball cue."""
    return start + np.arange(n) * 0.08


def feats(near_h, near_a, ball_x=None, period=None):
    n = len(near_h)
    ball_x = rolling(n) if ball_x is None else np.broadcast_to(ball_x, n)
    return pd.DataFrame({"frame": np.arange(n), "period": period or [1] * n,
                         "ball_x": ball_x, "ball_y": np.full(n, 25.0),
                         "near_H": near_h, "near_A": near_a, "speed_med": np.full(n, 2.0)})


def test_viterbi_pays_switch_cost_only_for_enough_evidence():
    cost = switch_costs()
    score = np.zeros((10, 3))
    score[:, H] = 1
    score[5:6, A] = 2          # one frame of A evidence: not worth two switches
    assert set(viterbi(score, cost)) == {H}
    score[5:, A] = 3           # sustained A evidence: switch once
    assert list(viterbi(score, cost)) == [H] * 5 + [A] * 5


def test_short_touch_does_not_flip_possession():
    near_h = np.r_[np.full(100, 0.5), np.full(10, 3.0), np.full(100, 0.5)]
    near_a = np.r_[np.full(100, 3.0), np.full(10, 0.5), np.full(100, 3.0)]
    assert set(decode(feats(near_h, near_a), FPS, PITCH)) == {"H"}   # 0.4 s touch by A


def test_sustained_control_flips_and_gap_holds():
    near_h = np.r_[np.full(100, 0.5), np.full(100, np.nan), np.full(100, 3.0)]
    near_a = np.r_[np.full(100, 3.0), np.full(100, np.nan), np.full(100, 0.5)]
    x = np.r_[rolling(100), np.full(100, np.nan), rolling(100)]
    out = decode(feats(near_h, near_a, x), FPS, PITCH)
    assert out[0] == "H" and out[-1] == "A"
    assert len(set(out[:100])) == 1 and len(set(out[-50:])) == 1


def test_ball_off_pitch_is_dead():
    n = 150
    rolling = 50 + np.arange(50) * 0.08   # 2 m/s, a ball in play
    x = np.r_[rolling, np.full(50, -5.0), rolling]
    out = decode(feats(np.full(n, 0.5), np.full(n, 3.0), x), FPS, PITCH)
    assert out[75] == "D" and out[0] == "H" and out[-1] == "H"


def test_still_ball_with_walking_players_is_dead():
    n = 500
    f = feats(np.full(n, 0.5), np.full(n, 3.0))
    f["speed_med"] = np.r_[np.full(250, 3.0), np.full(250, 0.5)]   # median over match 1.75
    f["ball_x"] = np.r_[50 + np.arange(250) * 0.08, np.full(250, 70.0)]
    out = decode(f, FPS, PITCH)
    assert out[100] == "H" and out[-50] == "D"


def test_posterior_is_a_distribution_and_agrees():
    from jaicodyn.decode import posterior
    near_h = np.r_[np.full(100, 0.5), np.full(100, 3.0)]
    near_a = np.r_[np.full(100, 3.0), np.full(100, 0.5)]
    f = feats(near_h, near_a, 50 + np.arange(200) * 0.08)
    post = posterior(f, FPS, PITCH)
    assert np.allclose(post.sum(axis=1), 1)
    assert post[20, H] > 0.5 and post[180, A] > 0.5


def test_periods_decoded_separately():
    n = 50
    near_h = np.r_[np.full(n, 0.5), np.full(n, 3.0)]
    near_a = np.r_[np.full(n, 3.0), np.full(n, 0.5)]
    out = decode(feats(near_h, near_a, period=[1] * n + [2] * n), FPS,
                 PITCH, p=Params(turnover_s=100))
    assert out[0] == "H" and out[-1] == "A"
