import numpy as np

from jaicodyn.features.touch import touch_columns, touch_frames, velocity_change

FPS = 25


def pass_and_return(n=100, turn=50):
    """Ball rolls +x at 8 m/s, turned back at frame `turn`."""
    x = np.where(np.arange(n) < turn, np.arange(n) * 0.32, (2 * turn - np.arange(n)) * 0.32)
    return np.column_stack([x, np.full(n, 30.0)])


def test_rolling_ball_has_no_velocity_change():
    ball = np.column_stack([np.arange(100) * 0.32, np.full(100, 30.0)])
    dv = velocity_change(ball, FPS, np.ones(100, int))
    assert np.nanmax(dv) < 1e-9


def test_turn_at_home_player_is_one_home_touch():
    ball = pass_and_return()
    near_h, near_a = np.full(100, 0.5), np.full(100, 5.0)
    rows, team = touch_frames(ball, near_h, near_a, FPS, np.ones(100, int))
    assert list(rows) == [50] and list(team) == [1]
    th, ta = touch_columns(ball, near_h, near_a, FPS, np.ones(100, int))
    assert th[45:56].all() and not th[:44].any() and not ta.any()


def test_turn_with_nobody_near_or_in_a_duel_names_no_team():
    ball = pass_and_return()
    period = np.ones(100, int)
    rows, _ = touch_frames(ball, np.full(100, 4.0), np.full(100, 5.0), FPS, period)
    assert len(rows) == 0                                    # a swerve in the air
    rows, team = touch_frames(ball, np.full(100, 0.6), np.full(100, 0.9), FPS, period)
    assert list(team) == [0]                                 # both at the ball


def test_unknown_ball_around_the_turn_gives_no_touch():
    ball = pass_and_return()
    ball[47:54] = np.nan                                     # the turn itself unseen
    rows, _ = touch_frames(ball, np.full(100, 0.5), np.full(100, 5.0), FPS, np.ones(100, int))
    assert len(rows) == 0
