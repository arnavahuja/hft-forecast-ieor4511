from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.analytics.robustness.core import (
    breakeven,
    pbo_cscv,
    walk_forward_windows,
)


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


def test_breakeven_interpolation():
    assert breakeven([0, 1, 2, 3], [0.1, 0.05, -0.05, -0.1]) == pytest.approx(1.5)
    assert breakeven([0, 1], [0.1, 0.2]) is None
