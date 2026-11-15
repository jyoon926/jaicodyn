import numpy as np
import pandas as pd

from jaicodyn.decode import A, D, H, Params, decode, decode_anchor_rules, switch_costs, viterbi
from jaicodyn.decode.anchors import anchor_labels

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


def test_anchor_rules_pin_only_some_frames_and_fill_between_them():
    near_h = np.r_[np.full(100, 0.5), np.full(10, 3.0), np.full(100, 0.5)]
    near_a = np.r_[np.full(100, 3.0), np.full(10, 0.5), np.full(100, 3.0)]
    anchors = anchor_labels(feats(near_h, near_a), FPS, p=Params(anchor_closer_m=2.0))
    assert np.any(anchors == "H")
    assert np.any(anchors == None)

    near_h = np.r_[np.full(100, 0.5), np.full(100, 3.0)]
    near_a = np.r_[np.full(100, 3.0), np.full(100, 0.5)]
    out = decode_anchor_rules(feats(near_h, near_a), FPS, PITCH)
    assert out[0] == "H" and out[-1] == "A"
    assert np.any(np.asarray(out) == "H") and np.any(np.asarray(out) == "A")


def test_anchor_rules_require_longer_sustained_control():
    near_h = np.r_[np.full(20, 3.0), np.full(12, 0.5), np.full(20, 3.0)]
    near_a = np.r_[np.full(20, 0.5), np.full(12, 3.0), np.full(20, 0.5)]
    anchors = anchor_labels(feats(near_h, near_a), FPS)
    assert not np.any(anchors[20:32] == "H")
    assert np.any(anchors == None)


def test_anchor_rules_require_a_very_clear_margin():
    near_h = np.r_[np.full(30, 1.4), np.full(12, 0.4), np.full(30, 3.0)]
    near_a = np.r_[np.full(30, 3.0), np.full(12, 2.5), np.full(30, 0.5)]
    anchors = anchor_labels(feats(near_h, near_a), FPS, p=Params(anchor_closer_m=2.0))
    assert not np.any(anchors[30:42] == "H")
    assert np.any(anchors == None)


def test_anchor_rules_skip_dead_ball_frames():
    near_h = np.r_[np.full(60, 0.5), np.full(20, 0.2)]
    near_a = np.r_[np.full(60, 3.0), np.full(20, 3.0)]
    x = np.r_[50 + np.arange(60) * 0.08, np.full(20, 50.0)]
    anchors = anchor_labels(feats(near_h, near_a, x), FPS, pitch=PITCH)
    assert np.all(anchors[60:] == None)


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


def restart_case(near_h_after=1.45, near_a_after=1.55):
    """H in possession, ball out for 3 s, then play resumes with H a little nearer. The default
    is too little evidence (about 0.4 s over 4 s) to pay for a turnover once play is under way."""
    near_h = np.r_[np.full(100, 0.5), np.full(75, 0.5), np.full(100, near_h_after)]
    near_a = np.r_[np.full(100, 3.0), np.full(75, 3.0), np.full(100, near_a_after)]
    x = np.r_[rolling(100), np.full(75, -5.0), rolling(100)]
    return feats(near_h, near_a, x)


def spells(labels):
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i))
            start = i
    return out


def test_restart_evidence_is_zero_at_half_and_outside_windows():
    from jaicodyn.decode import restart_evidence
    assert (restart_evidence(np.array([0.5, np.nan])) == 0).all()
    assert restart_evidence(np.array([0.9]))[0] > 0 > restart_evidence(np.array([0.1]))[0]


def test_restart_windows_follow_each_restart():
    from jaicodyn.decode import restart_windows
    labels = np.array(list("HHDDDAAAAAAHH"))
    period = np.ones(len(labels), int)
    w = restart_windows(labels, period, np.full(len(labels), 0.2), fps=1,
                        p=Params(restart_window_s=4))
    assert np.isnan(w[[2, 3, 4, 9, 10]]).all()      # dead, and after the window
    assert (w[[0, 1]] == 0.2).all()                  # kickoff, cut short by the stoppage
    assert (w[5:9] == 0.2).all()                     # 4 frames after the restart


def test_uninformative_restart_probs_change_nothing():
    from jaicodyn.decode import posterior
    f = restart_case()
    half = np.full(len(f), 0.5)
    for scale in (0.5, 1.0, 2.0):
        p = Params(restart_scale=scale)
        assert (decode(f, FPS, PITCH, p=p) == decode(f, FPS, PITCH, p=p, restart_p_home=half)).all()
        assert np.allclose(posterior(f, FPS, PITCH, p=p),
                           posterior(f, FPS, PITCH, p=p, restart_p_home=half))


def test_confident_restart_prob_gives_the_restart_to_that_team():
    f = restart_case()
    assert decode(f, FPS, PITCH)[-1] == "H"
    out = decode(f, FPS, PITCH, restart_p_home=np.full(len(f), 0.001))   # away restarts
    assert out[120] == "D" and out[-1] == "A"


def test_restart_possession_is_not_handed_straight_back():
    # H clearly nearer after the restart: the per-stoppage cost version restarted to A and gave
    # the ball back to H one frame later. Spread as evidence, A must hold through the window.
    f = restart_case(1.4, 1.6)
    out = decode(f, FPS, PITCH, restart_p_home=np.full(len(f), 0.001))
    after = [sp for sp in spells(out) if sp[1] >= 175]
    assert after[0][0] == "A" and after[0][2] - after[0][1] >= 3 * FPS


def test_set_piece_tags_give_the_restart_team():
    import pandas as pd
    from jaicodyn.decode.tags import SET_PIECE_P, set_piece_restart_p
    f = feats(np.full(2000, 0.5), np.full(2000, 3.0))
    tags = pd.DataFrame({"start": [100, 1000, 1800], "stop": [975, 1875, 2675],
                         "code": ["SET PIECES", "Set Pieces", "SET PIECES"],
                         "team": ["A", "H", ""], "type": ["Corner Kick", "Goal Kick", "Throw-in"]})
    p = set_piece_restart_p(f, tags, FPS)
    assert np.isclose(p[100], 1 - SET_PIECE_P) and np.isclose(p[1000 + 10 * FPS], SET_PIECE_P)
    assert np.isnan(p[1800 + 5])                                   # throw-ins carry no team
    assert np.isnan(p[100 - 2 * FPS]) and np.isnan(p[100 + 16 * FPS])
    assert set_piece_restart_p(f, None, FPS) is None


def test_phase_evidence_only_in_gaps_and_only_between_teams():
    import pandas as pd
    from jaicodyn.decode.tags import phase_team_evidence
    n = 600
    x = np.r_[rolling(200), np.full(200, np.nan), rolling(200)]
    f = feats(np.r_[np.full(200, 0.5), np.full(200, np.nan), np.full(200, 0.5)],
              np.r_[np.full(200, 3.0), np.full(200, np.nan), np.full(200, 3.0)], x)
    tags = pd.DataFrame({"start": [0, 0], "stop": [n, n], "type": ["", ""],
                         "code": ["PLAYERS IN THE BOX", "DEFENDING IN MIDDLE THIRD"],
                         "team": ["H", "H"]})       # the mirror names H as defending: A has it
    e = phase_team_evidence(f, tags)
    assert (e[:200] == 0).all() and (e[400:] == 0).all()   # visible ball: no tag evidence
    assert (e[200:400] < 0).all()                           # the more reliable source wins: A
    from jaicodyn.decode import emissions
    base, with_tags = emissions(f, FPS, PITCH), emissions(f, FPS, PITCH, team_evidence=e)
    assert np.allclose(base[:, 2], with_tags[:, 2])          # dead-ball scores untouched
    assert (with_tags[200:400].max(axis=1) <= base[200:400].max(axis=1) + 1e-12).all()


def test_no_tags_no_phase_evidence():
    from jaicodyn.decode.tags import phase_team_evidence
    assert phase_team_evidence(feats(np.full(10, 0.5), np.full(10, 3.0)), None) is None
