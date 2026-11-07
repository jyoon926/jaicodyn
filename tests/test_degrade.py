import numpy as np

from jaicodyn.data.degrade import hide_mask, targets
from jaicodyn.features.ball import true_runs

FPS = 25
LENGTHS = np.array([5.0, 8.0, 12.0, 20.0, 40.0])


def match_labels(seed=0, n_stops=60):
    """Alternating live play (H/A) and stoppages, two halves, X before kickoff."""
    rng = np.random.default_rng(seed)
    parts = [np.full(500, "X")]
    for _ in range(n_stops):
        parts.append(np.full(int(rng.integers(20, 60) * FPS), rng.choice(["H", "A"])))
        parts.append(np.full(int(rng.integers(5, 40) * FPS), "D"))
    labels = np.concatenate(parts)
    period = np.where(np.arange(len(labels)) < len(labels) // 2, 1, 2)
    return labels, period


def test_targets_keep_overall_share_and_ratio():
    labels = np.array(["D"] * 30 + ["H"] * 70)
    dead, live = targets(labels, 0.2, 10)
    assert np.isclose(0.3 * dead + 0.7 * live, 0.2)
    assert np.isclose(dead / live, 10)


def test_targets_cap_dead_and_give_the_rest_to_live():
    labels = np.array(["D"] * 30 + ["H"] * 70)
    dead, live = targets(labels, 0.4, 10)   # uncapped dead would be 108%
    assert dead == 0.95 and np.isclose(0.3 * dead + 0.7 * live, 0.4)


def test_mask_hits_targets_and_respects_bounds():
    labels, period = match_labels()
    mask = hide_mask(labels, period, FPS, 0, 0.4, 10, LENGTHS)
    graded = labels != "X"
    assert not mask[~graded].any()
    for s, n in zip(*true_runs(mask)):
        assert period[s] == period[s + n - 1]
        assert n >= 5 * FPS
    dead = mask[labels == "D"].mean()
    live = mask[np.isin(labels, ["H", "A"])].mean()
    assert 0.3 < mask[graded].mean() < 0.5
    assert dead / live > 5


def test_same_seed_same_mask():
    labels, period = match_labels()
    a = hide_mask(labels, period, FPS, 7, 0.4, 10, LENGTHS)
    b = hide_mask(labels, period, FPS, 7, 0.4, 10, LENGTHS)
    assert (a == b).all()
