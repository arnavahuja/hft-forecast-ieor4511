"""The feature study: does it find a signal that is there, and stay quiet when it is not."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import DIRECTION, FWD_LOGRET, LABEL_VALID, SESSION_ID, HorizonSpec
from hft_forecast.ml.eda import (
    BUCKET,
    FEATURE,
    HORIZON_SECONDS,
    EdaParams,
    block_ic,
    bucket_profile,
    coverage,
    feature_correlation,
    ic_ranking,
    missing_shares,
    return_decay,
    run_eda,
)
from hft_forecast.ml.eda_charts import eda_charts
from hft_forecast.services.eda import json_safe

HORIZON = HorizonSpec(name="t", seconds=60, grid="5s")
BARS = 200
STEPS = HORIZON.steps


def panel(sessions: int = 4, instruments=("AAA", "BBB"), seed: int = 0) -> pl.DataFrame:
    """A dataset-shaped frame where ``f_signal`` leads the label and ``f_noise`` does not."""
    rng = np.random.default_rng(seed)
    frames = []
    for name in instruments:
        for session in range(sessions):
            steps = rng.normal(0, 1e-4, BARS)
            mid = 100.0 * np.exp(np.cumsum(steps))
            forward = np.log(np.roll(mid, -STEPS) / mid)
            forward[-STEPS:] = np.nan
            frames.append(
                pl.DataFrame(
                    {
                        C.TIMESTAMP: np.arange(BARS) * 5,
                        C.INSTRUMENT: [name] * BARS,
                        SESSION_ID: [session] * BARS,
                        C.MID: mid,
                        C.SPREAD_BPS: rng.uniform(1, 3, BARS),
                        # the signal is the label plus noise of the same size: an IC around
                        # 0.7, far from the 0.01 of real microstructure but unmistakable
                        "f_signal": forward + rng.normal(0, 1e-4, BARS),
                        "f_noise": rng.normal(0, 1, BARS),
                        "f_flat": np.ones(BARS),
                        FWD_LOGRET: forward,
                        DIRECTION: np.sign(np.nan_to_num(forward)).astype("int8"),
                        LABEL_VALID: ~np.isnan(forward),
                    }
                )
            )
    return pl.concat(frames)


FEATURES = ["f_signal", "f_noise", "f_flat"]


def valid(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col(LABEL_VALID))


# ------------------------------------------------------------------ information coefficient


def test_ic_finds_the_planted_signal_and_not_the_noise():
    ranking = ic_ranking(block_ic(valid(panel()), FEATURES))
    rows = {r[FEATURE]: r for r in ranking.to_dicts()}
    assert rows["f_signal"]["mean_ic"] > 0.5
    assert rows["f_signal"]["share_positive"] == 1.0
    assert abs(rows["f_noise"]["mean_ic"]) < 0.1
    assert ranking.get_column(FEATURE).to_list()[0] == "f_signal", "strongest comes first"


def test_a_constant_feature_is_absent_rather_than_blank():
    """Its correlation is undefined, and polars reports that as NaN, not null."""
    assert "f_flat" not in block_ic(valid(panel()), FEATURES).get_column(FEATURE).to_list()


def test_one_row_per_instrument_session_block():
    ic = block_ic(valid(panel(sessions=3, instruments=("AAA", "BBB"))), ["f_signal"])
    assert ic.height == 6
    assert ic.get_column("n").min() >= 1


def test_small_blocks_are_skipped_not_imputed():
    assert block_ic(valid(panel()), ["f_signal"], min_block_rows=10_000).is_empty()


def test_t_stat_is_taken_over_blocks():
    ranking = ic_ranking(block_ic(valid(panel(sessions=8)), ["f_signal"]))
    row = ranking.to_dicts()[0]
    expected = row["mean_ic"] * math.sqrt(row["blocks"]) / row["std_ic"]
    assert row["t_stat"] == pytest.approx(expected, rel=1e-6)
    assert row["blocks"] == 16


# ------------------------------------------------------------------ the other sections


def test_buckets_are_monotone_for_a_monotone_feature():
    profile = bucket_profile(valid(panel()), ["f_signal"], buckets=5)
    means = profile.sort(BUCKET).get_column("median_block_mean_bps").to_list()
    assert means == sorted(means), "a feature that leads the return tilts the buckets in order"
    assert profile.height == 5
    assert profile.get_column("blocks").to_list() == [8] * 5


def test_decay_covers_the_configured_horizon():
    decay = return_decay(panel(), ["f_signal"], [3, 12, 24], bar_seconds=5)
    seconds = decay.get_column(HORIZON_SECONDS).to_list()
    assert seconds == [15, 60, 120]
    peak = decay.sort("mean_ic", descending=True).get_column(HORIZON_SECONDS).to_list()[0]
    assert peak == 60, "the signal was planted at the 60s horizon, so the IC peaks there"


def test_correlation_leaves_out_constant_features():
    corr = feature_correlation(valid(panel()), FEATURES)
    names = set(corr.get_column(FEATURE).to_list())
    assert names == {"f_signal", "f_noise"}
    diagonal = corr.filter(pl.col(FEATURE) == pl.col("other")).get_column("correlation").to_list()
    assert all(d == pytest.approx(1.0) for d in diagonal)


def test_missing_shares_count_nulls_and_infinities():
    frame = panel(sessions=1, instruments=("AAA",))
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < BARS // 4)
        .then(None)
        .otherwise(pl.col("f_noise"))
        .alias("f_noise")
    )
    shares = missing_shares(frame, ["f_noise", "f_signal"])
    by_feature = {r[FEATURE]: r["missing_share"] for r in shares.to_dicts()}
    assert by_feature["f_noise"] == pytest.approx(0.25)
    assert by_feature["f_signal"] == pytest.approx(STEPS / BARS), "the tail has no label"


def test_coverage_never_claims_more_ready_than_labelled():
    rows = coverage(panel(), FEATURES).to_dicts()
    assert len(rows) == 2
    for row in rows:
        assert row["model_ready_share"] <= row["label_valid_share"]
        assert row["sessions"] == 4
        assert row["bars"] == 4 * BARS
        assert row["median_spread_bps"] > 0


# ------------------------------------------------------------------ the whole study


def test_run_eda_studies_the_training_sessions_only():
    report = run_eda(panel(sessions=10), HORIZON, EdaParams(train_sessions=4))
    assert report.summary["sessions"] == 4
    assert report.summary["instruments"] == 2
    assert report.summary["top_feature"] == "f_signal"
    assert report.summary["features"] == 3
    assert report.summary["features_ranked"] == 2, "the constant feature is not ranked"
    assert report.horizon_seconds == 60
    assert report.bar_seconds == 5


def test_run_eda_without_features_says_so():
    frame = panel().drop(FEATURES)
    with pytest.raises(ConfigError, match=r"no f_\* columns"):
        run_eda(frame, HORIZON, EdaParams())


def test_run_eda_without_a_valid_label_says_so():
    frame = panel().with_columns(pl.lit(value=False).alias(LABEL_VALID))
    with pytest.raises(ConfigError, match="no rows with a valid label"):
        run_eda(frame, HORIZON, EdaParams())


def test_every_table_is_populated_and_charted():
    report = run_eda(panel(), HORIZON, EdaParams())
    tables = report.tables()
    assert set(tables) == {"coverage", "ranking", "buckets", "decay", "correlation", "missing"}
    assert all(rows for rows in tables.values())
    charts = eda_charts(report)
    assert [c.id for c in charts] == [
        "eda_ic_ranking",
        "eda_bucket_profile",
        "eda_ic_decay",
        "eda_coverage",
        "eda_correlation",
        "eda_missing",
    ]
    assert all(c.series for c in charts)


def test_the_payload_carries_no_non_json_floats():
    """NaN and Infinity are not JSON, and one of them rejects the whole document."""
    report = run_eda(panel(), HORIZON, EdaParams())
    payload = {"summary": report.summary, "tables": report.tables()}
    assert json_safe({"a": float("nan"), "b": [float("inf")], "c": 1.5}) == {
        "a": None,
        "b": [None],
        "c": 1.5,
    }
    import json

    json.dumps(json_safe(payload), allow_nan=False)
