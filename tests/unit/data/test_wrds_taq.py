"""TAQ NBBO and trade bars against a fake WRDS: labelling, filters, staleness, caps."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import SourceEnvironment
from hft_forecast.core.types import BAR_LABEL_KEY, DataRequest, Frequency
from hft_forecast.data.sources.wrds_taq import (
    NBBO_COLUMNS,
    WrdsTaqRawSource,
    WrdsTaqSource,
    nbbo_columns,
)
from hft_forecast.data.validation import QUOTE_FILTER_COUNTS_KEY, Severity, validate
from tests.fixtures.fake_taq import GAP_END, GAP_START, GAP_SYMBOL, FakeTaq

NORMAL = date(2024, 7, 2)
EARLY_CLOSE = date(2024, 11, 29)  # the day after Thanksgiving: 13:00 local
CAL = TradingCalendar()


@pytest.fixture
def fake() -> FakeTaq:
    return FakeTaq()


def _source(cls, fake, monkeypatch, tmp_path: Path, **params):
    src = cls(
        cls.Params(**params), env=SourceEnvironment(tmp_path, tmp_path, wrds_username="tester")
    )
    monkeypatch.setattr(src, "_query", fake.query)
    return src


@pytest.fixture
def source(fake, monkeypatch, tmp_path: Path) -> WrdsTaqSource:
    return _source(WrdsTaqSource, fake, monkeypatch, tmp_path)


def _req(dataset="taq_nbbo_bars", day=NORMAL, **kw) -> DataRequest:
    base = {
        "source": "wrds_taq",
        "dataset": dataset,
        "instruments": ("AAA", "BBB"),
        "start": day,
        "end": day,
        "frequency": Frequency.S5,
    }
    return DataRequest(**{**base, **kw})


def test_unverified_year_refuses_to_guess_column_names():
    assert 2024 in NBBO_COLUMNS
    with pytest.raises(DataError) as info:
        nbbo_columns(1999)
    assert "describe_table" in info.value.details["how"]


def test_bars_are_labelled_at_the_bar_end_within_the_session(source):
    data = source.fetch(_req())
    assert data.metadata[BAR_LABEL_KEY] == "end"
    assert data.metadata[C.PRICE_BASIS_KEY] == "mid"
    open_utc, close_utc = CAL.session_bounds(NORMAL)
    step = np.timedelta64(5, "s")
    stamps = data.timestamps.astype("datetime64[us]")
    # bars before the session's first quote stay out; the last bar ends at the close
    assert stamps.min() >= open_utc + step
    assert stamps.max() == close_utc
    offsets = (stamps - (open_utc + step)).astype("timedelta64[s]").astype(int)
    assert np.all(offsets % 5 == 0)


def test_early_close_session_stops_at_13_00_local(source):
    data = source.fetch(_req(day=EARLY_CLOSE))
    _, close_utc = CAL.session_bounds(EARLY_CLOSE)
    assert data.timestamps.astype("datetime64[us]").max() == close_utc
    assert close_utc == np.datetime64("2024-11-29T18:00", "us")


def test_mid_is_consistent_and_close_is_the_mid(source):
    frame = source.fetch(_req()).frame
    mid = (frame[C.BID] + frame[C.ASK]).to_numpy() / 2
    assert np.allclose(frame[C.MID].to_numpy(), mid, atol=1e-12)
    assert np.allclose(frame[C.CLOSE].to_numpy(), mid, atol=1e-12)


def test_a_quote_gap_is_forward_filled_and_flagged_stale(source):
    data = source.fetch(_req())
    gap = data.select_instruments((GAP_SYMBOL,)).frame
    utc = gap[C.TIMESTAMP].to_numpy().astype("datetime64[s]").astype(np.int64)
    seconds = utc % 86_400 - 4 * 3600  # UTC seconds of day to New York local, July
    inside = (seconds > GAP_START) & (seconds <= GAP_END)
    stale = gap[C.IS_STALE].to_numpy()
    assert stale[inside].min() == 1.0
    assert stale[inside].sum() > 20
    filled = gap[C.MID].to_numpy()[inside]
    assert np.all(filled == filled[0])  # the last quote before the gap, carried forward
    assert gap[C.N_QUOTES].to_numpy()[inside].max() == 0


def test_rejected_quotes_are_counted_per_filter(source):
    counts = source.fetch(_req()).metadata[QUOTE_FILTER_COUNTS_KEY]
    assert counts["non_positive"] > 0
    assert counts["locked_or_crossed"] > 0
    assert counts["quote_condition"] > 0
    assert counts["wide_spread"] == 0


def test_filters_and_conditions_reach_the_sql(source, fake):
    source.fetch(_req())
    bars = next(q for q in fake.queries if "with q as" in q)
    assert "sym_suffix is null" in bars
    assert "qu_cond in %(conds)s" in bars
    assert "best_ask > best_bid" in bars
    assert "mid_twap" in bars


def test_quality_report_has_no_errors(source):
    data = source.fetch(_req())
    report = validate(data, CAL)
    assert [i for i in report.issues if i.severity is Severity.ERROR] == []


def test_trade_bars_carry_vwap_and_counts(source):
    data = source.fetch(_req("taq_trade_bars"))
    assert {C.VWAP, C.N_TRADES} <= set(data.fields)
    _, close_utc = CAL.session_bounds(NORMAL)
    assert data.timestamps.astype("datetime64[us]").max() == close_utc


def test_unknown_dataset_is_rejected(source):
    with pytest.raises(DataError):
        source.fetch(_req("taq_quotes"))


def test_raw_quotes_are_capped_and_uncacheable(fake, monkeypatch, tmp_path: Path):
    src = _source(WrdsTaqRawSource, fake, monkeypatch, tmp_path, max_raw_sessions=1)
    assert src.cacheable is False
    data = src.fetch(_req("taq_nbbo_raw", frequency=Frequency.S1))
    assert len(data) > 1000
    with pytest.raises(DataError, match="capped"):
        src.fetch(_req("taq_nbbo_raw", day=NORMAL, end=date(2024, 7, 10), frequency=Frequency.S1))
