from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.analytics.robustness.core import (
    bootstrap_paths,
    breakeven,
    expand_grid,
    path_stats,
    pbo_cscv,
    random_grid,
    shuffle_trade_paths,
    walk_forward_windows,
)


def test_expand_and_random_grid():
    grid = expand_grid({"b": [1, 2], "a": ["x", "y", "z"]})
    assert len(grid) == 6 and grid[0] == {"a": "x", "b": 1}
    pts = random_grid(
        {"n": {"min": 2, "max": 5, "type": "int"}, "f": {"values": ["a", "b"]}},
        20,
        np.random.default_rng(0),
    )
    assert all(2 <= p["n"] <= 5 and p["f"] in ("a", "b") for p in pts)


def test_walk_forward_windows_cover_the_tail():
    ws = walk_forward_windows(100, train=40, test=25, anchored=False)
    assert [(w.train_start, w.train_end, w.test_end) for w in ws] == [
        (0, 40, 65),
        (25, 65, 90),
        (50, 90, 100),
    ]
    anchored = walk_forward_windows(100, 40, 25, anchored=True)
    assert all(w.train_start == 0 for w in anchored)


def test_pbo_detects_overfitting_on_noise_and_not_on_real_edge():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, size=(1000, 30))
    assert 0.3 < pbo_cscv(noise, 10)["pbo"] < 0.8
    edge = noise.copy()
    edge[:, 0] += 0.004  # one genuinely better trial
    assert pbo_cscv(edge, 10)["pbo"] < 0.1


def test_monte_carlo_paths_and_stats():
    r = np.random.default_rng(1).normal(0.0005, 0.01, 500)
    paths = bootstrap_paths(r, 200, 300, 5.0, seed=3)
    assert paths.shape == (200, 300)
    np.testing.assert_array_equal(paths, bootstrap_paths(r, 200, 300, 5.0, seed=3))
    stats = path_stats(paths)
    assert stats["fan"].shape == (5, 300)
    assert (stats["max_drawdown"] <= 0).all()
    shuffled = shuffle_trade_paths(np.array([0.1, -0.05, 0.02]), 50, 0)
    np.testing.assert_allclose(np.sort(shuffled, axis=1)[0], [-0.05, 0.02, 0.1])
    # order does not change terminal wealth for a pure product
    np.testing.assert_allclose(path_stats(shuffled)["terminal"], 1.1 * 0.95 * 1.02)


def test_breakeven_interpolation():
    assert breakeven([0, 1, 2, 3], [0.1, 0.05, -0.05, -0.1]) == pytest.approx(1.5)
    assert breakeven([0, 1], [0.1, 0.2]) is None
