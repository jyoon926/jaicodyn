import numpy as np
import pandas as pd

from jaicodyn.decode import A, D, H, Params, decode, switch_costs, viterbi

FPS, PITCH = 25, (100.0, 50.0)


def feats(near_h, near_a, ball_x=50.0, period=None):
    n = len(near_h)
    return pd.DataFrame({"period": period or [1] * n, "ball_x": np.broadcast_to(ball_x, n),
                         "ball_y": np.full(n, 25.0), "near_H": near_h, "near_A": near_a})


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
    x = np.r_[np.full(100, 50.0), np.full(100, np.nan), np.full(100, 50.0)]
    out = decode(feats(near_h, near_a, x), FPS, PITCH)
    assert out[0] == "H" and out[-1] == "A"
    assert len(set(out[:100])) == 1 and len(set(out[-50:])) == 1


def test_ball_off_pitch_is_dead():
    n = 150
    x = np.r_[np.full(50, 50.0), np.full(50, -5.0), np.full(50, 50.0)]
    out = decode(feats(np.full(n, 0.5), np.full(n, 3.0), x), FPS, PITCH)
    assert out[75] == "D" and out[0] == "H" and out[-1] == "H"


def test_periods_decoded_separately():
    n = 50
    near_h = np.r_[np.full(n, 0.5), np.full(n, 3.0)]
    near_a = np.r_[np.full(n, 3.0), np.full(n, 0.5)]
    out = decode(feats(near_h, near_a, period=[1] * n + [2] * n), FPS,
                 PITCH, Params(turnover_s=100))
    assert out[0] == "H" and out[-1] == "A"
