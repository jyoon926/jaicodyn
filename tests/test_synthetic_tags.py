import numpy as np

from jaicodyn.decode import synthetic_tags as st


def test_solve_q_hits_the_target_and_refuses_the_impossible():
    p_dec, p_true = 0.6, 0.95
    for acc, r in ((0.75, 0.0), (0.75, 0.3), (0.7, 0.6)):
        q = st.solve_q(acc, r, p_dec, p_true)
        reached = r * p_dec + (1 - r) * ((1 - q) * p_true + q * (1 - p_true))
        assert 0 <= q <= 1 and np.isclose(reached, acc)
    assert st.solve_q(0.95, 0.6, p_dec, p_true) is None      # truth part would need > 100%
    assert st.solve_q(0.7, 1.0, p_dec, p_true) is None


def test_intervals_fixed_and_to_possession_end():
    truth = np.array(list("HHHHHAAAAA"))
    period = np.ones(10, int)
    trig = [("BUILD UP", 1, "H")]
    assert st.intervals(trig, 3, 1, 10, period)[0][1:3] == (1, 4)                 # fixed 3 s
    assert st.intervals(trig, None, 1, 10, period, truth, 0)[0][1:3] == (1, 5)    # to the end
    assert st.intervals(trig, None, 1, 10, period, truth, 2)[0][1:3] == (1, 7)    # + 2 s overrun
    period[4:] = 2
    assert st.intervals(trig, 8, 1, 10, period)[0][2] == 4                         # stays in its half


def test_make_tags_adds_mirrors_and_copies_the_decoder():
    decoded = np.array(list("DDAAAAAAAA"))
    frame = np.arange(10)
    iv = [("BUILD UP", 0, 6, "H"), ("BALL IN THE BOX", 6, 10, "H")]
    tags = st.make_tags(iv, decoded, frame, r=1.0 - 1e-12, q=0.0, seed=1)
    assert list(tags["code"]) == ["BUILD UP", "DEFENDING IN ATTACKING THIRD", "BALL IN THE BOX"]
    assert list(tags["team"]) == ["A", "H", "A"]   # copied: first live decoder team; mirror flipped
    truthful = st.make_tags(iv, decoded, frame, r=0.0, q=0.0, seed=1)
    assert list(truthful["team"]) == ["H", "A", "H"]
