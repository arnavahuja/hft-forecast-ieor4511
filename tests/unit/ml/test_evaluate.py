"""Metrics, deflation and the comparison table."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.ml.base import DIRECTION, FWD_LOGRET, LABEL_END, MAGNITUDE_BPS, SESSION_ID
from hft_forecast.ml.deflate import bonferroni, deflated, overfitting, t_stat_p_value
from hft_forecast.ml.evaluate import evaluate
from hft_forecast.ml.report import (
    BEST_BY,
    Entry,
    comparison_table,
    entry_from_metrics,
    write_report,
)

N_SESSIONS, PER_SESSION = 8, 120


def _predictions(skill: float, seed: int = 0) -> pl.DataFrame:
    """Predictions whose accuracy rises with ``skill`` (0 = coin flip, 1 = perfect)."""
    rng = np.random.default_rng(seed)
    n = N_SESSIONS * PER_SESSION
    y_true = rng.normal(scale=1e-4, size=n)
    truth = np.sign(y_true).astype(int)
    right = rng.random(n) < skill
    guess = np.where(right, truth, -truth)
    p_up = np.where(guess > 0, 0.7, 0.3)
    stamps = np.datetime64("2024-07-01T14:00", "us") + np.arange(n) * np.timedelta64(5, "s")
    return pl.DataFrame(
        {
            C.TIMESTAMP: stamps,
            C.INSTRUMENT: ["AAA"] * n,
            SESSION_ID: np.repeat(np.arange(N_SESSIONS), PER_SESSION),
            "fold": np.repeat([0, 1], n // 2).astype(np.int32),
            "model": ["m@1.0.0"] * n,
            "p_down": 1.0 - p_up,
            "p_flat": np.zeros(n),
            "p_up": p_up,
            "y_hat_logret": guess * np.abs(y_true) * skill,
            FWD_LOGRET: y_true,
            DIRECTION: truth,
            MAGNITUDE_BPS: np.abs(y_true) * 1e4,
            LABEL_END: stamps,
        }
    )


def test_a_skilled_model_scores_above_a_coin_flip():
    good = evaluate(_predictions(0.9))["models"]["m@1.0.0"]["pooled"]
    coin = evaluate(_predictions(0.5))["models"]["m@1.0.0"]["pooled"]
    assert good["accuracy"] > 0.8 > coin["accuracy"]
    assert good["auc_up"] > coin["auc_up"]
    assert good["ic"] > coin["ic"]
    assert good["log_loss"] < coin["log_loss"]
    assert good["brier"] < coin["brier"]


def test_metrics_are_reported_per_fold_and_pooled():
    report = evaluate(_predictions(0.7))
    model = report["models"]["m@1.0.0"]
    assert [f["fold"] for f in model["folds"]] == [0, 1]
    assert sum(f["n_rows"] for f in model["folds"]) == model["pooled"]["n_rows"]
    assert report["n_sessions"] == N_SESSIONS


def test_the_confusion_matrix_and_reliability_table_add_up():
    pooled = evaluate(_predictions(0.7))["models"]["m@1.0.0"]["pooled"]
    assert sum(sum(row) for row in pooled["confusion"]) == pooled["n_rows"]
    assert sum(b["n_rows"] for b in pooled["reliability"]) == pooled["n_rows"]
    assert all(0.0 <= b["observed"] <= 1.0 for b in pooled["reliability"])


def test_the_bootstrap_interval_brackets_the_estimate():
    pooled = evaluate(_predictions(0.75))["models"]["m@1.0.0"]["pooled"]
    low, high = pooled["ci"]["accuracy"]
    assert low < pooled["accuracy"] < high
    assert high - low > 0  # sessions resampled, not rows


def test_top_decile_metrics_use_fewer_rows():
    pooled = evaluate(_predictions(0.8))["models"]["m@1.0.0"]["pooled"]
    assert 0 < pooled["top_decile"]["n_rows"] < pooled["n_rows"]


def test_deflation_needs_trials():
    returns = np.random.default_rng(1).normal(1e-4, 1e-3, 500)
    alone = deflated(returns, [], 1)
    assert alone["dsr"] is None and alone["psr"] is not None
    many = deflated(returns, [0.05, 0.02, 0.1, -0.01], 40)
    assert many["dsr"] is not None and many["dsr"] <= (many["psr"] or 1.0)


def test_pbo_needs_four_trials():
    rng = np.random.default_rng(2)
    assert overfitting(rng.normal(size=(200, 3)))["pbo"] is None
    assert overfitting(rng.normal(size=(400, 8)))["pbo"] is not None


def test_bonferroni_and_p_values():
    assert bonferroni(0.01, 10) == pytest.approx(0.1)
    assert bonferroni(0.5, 10) == 1.0
    assert t_stat_p_value(1.96, 100) == pytest.approx(0.05, abs=0.002)


def test_the_comparison_table_bolds_the_best_model_per_horizon(tmp_path):
    metrics = evaluate(_predictions(0.8))
    good = entry_from_metrics("h30s", "m@1.0.0", metrics, {BEST_BY: 1.4})
    weak = Entry(horizon="h30s", model="zero@1.0.0", values={BEST_BY: 0.1})
    other = Entry(horizon="h5m", model="m@1.0.0", values={BEST_BY: 0.9})
    table = comparison_table([good, weak, other])
    assert table[0] == ["metric", "h30s / m@1.0.0", "h30s / zero@1.0.0", "h5m / m@1.0.0"]
    row = next(r for r in table if r[0].startswith("net Sharpe @ quoted_spread"))
    assert row[1].startswith("**") and not row[2].startswith("**")
    assert row[3].startswith("**")  # best of its own horizon
    accuracy = next(r for r in table if r[0].startswith("accuracy"))
    assert "[" in accuracy[1]  # the bootstrap interval travels with the number

    csv_path, md_path = write_report([good, weak, other], tmp_path / "report")
    assert csv_path.exists() and md_path.read_text().startswith("| metric |")


def test_missing_numbers_render_as_a_dash():
    table = comparison_table([Entry(horizon="h30s", model="m")])
    assert any(row[1] == "—" for row in table[1:])
