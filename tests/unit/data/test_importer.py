"""Importing level-1 quote files and level-N book files through the same mapping."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataValidationError
from hft_forecast.core.types import BAR_LABEL_KEY, DataRequest, Frequency
from hft_forecast.data.importer import (
    ImportMapping,
    apply_mapping,
    guess_book_columns,
    guess_mapping,
    quote_metadata,
)

STAMPS = [datetime(2024, 7, 1, 14, 0), datetime(2024, 7, 1, 14, 1)]


def book_frame(levels: int = 2) -> pl.DataFrame:
    """A level-N book file as a venue might hand it over: prices and sizes per level."""
    cols: dict[str, object] = {"ts": STAMPS, "symbol": ["AAA", "AAA"]}
    for level in range(1, levels + 1):
        cols[f"bid_px_{level}"] = [100.0 - level * 0.01, 100.5 - level * 0.01]
        cols[f"bid_sz_{level}"] = [10.0 * level, 12.0 * level]
        cols[f"ask_px_{level}"] = [100.02 + level * 0.01, 100.52 + level * 0.01]
        cols[f"ask_sz_{level}"] = [20.0 * level, 22.0 * level]
    cols[C.N_QUOTES] = [7.0, 9.0]
    cols[C.IS_STALE] = [0.0, 0.0]
    return pl.DataFrame(cols)


def quote_frame() -> pl.DataFrame:
    """A level-1 quote file with no price column at all."""
    return pl.DataFrame(
        {
            "ts": STAMPS,
            "symbol": ["AAA", "AAA"],
            "best_bid": [100.0, 100.5],
            "best_ask": [100.02, 100.52],
            "best_bidsiz": [10.0, 12.0],
            "best_asksiz": [20.0, 22.0],
        }
    )


def test_guess_finds_the_book_and_keeps_it_out_of_the_extra_columns():
    mapping = guess_mapping(book_frame(levels=3), "aaa")
    assert mapping.book_levels == 3
    assert mapping.book_columns["bid_px_1"] == "bid_px_1"
    assert len(mapping.book_columns) == 12
    assert mapping.extra == []
    assert mapping.columns[C.N_QUOTES] == C.N_QUOTES


def test_a_partial_level_ends_the_book():
    columns = ["bid_px_1", "bid_sz_1", "ask_px_1", "ask_sz_1", "bid_px_2", "bid_sz_2"]
    assert len(guess_book_columns(columns)) == 4


def test_alternative_level_names_are_recognised():
    columns = ["bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"]
    assert guess_book_columns(columns) == {
        "bid_px_1": "bid_price_1",
        "bid_sz_1": "bid_size_1",
        "ask_px_1": "ask_price_1",
        "ask_sz_1": "ask_size_1",
    }


def test_a_book_import_writes_every_level_and_derives_level_one():
    frame = book_frame(levels=2)
    mapping = guess_mapping(frame, "aaa").model_copy(update={"frequency": Frequency.MIN1})
    out = apply_mapping(frame, mapping)
    assert set(C.book_columns(2)) <= set(out.columns)
    assert out[C.BID].to_list() == frame["bid_px_1"].to_list()
    assert out[C.ASK_SIZE].to_list() == frame["ask_sz_1"].to_list()
    assert out[C.MID][0] == pytest.approx(100.01)
    assert out[C.SPREAD][0] == pytest.approx(0.04)
    assert out[C.SPREAD_BPS][0] == pytest.approx(0.04 / 100.01 * 1e4)
    # the size-weighted mid leans towards the side with less size on it
    assert out[C.MICROPRICE][0] == pytest.approx((99.99 * 20.0 + 100.03 * 10.0) / 30.0)
    assert out[C.CLOSE].to_list() == out[C.MID].to_list()


def test_a_level_one_file_needs_no_price_column():
    frame = quote_frame()
    mapping = guess_mapping(frame, "aaa").model_copy(update={"frequency": Frequency.MIN1})
    assert mapping.book_levels == 0
    out = apply_mapping(frame, mapping)
    assert out[C.CLOSE].to_list() == out[C.MID].to_list()
    assert quote_metadata(mapping, out.columns) == {BAR_LABEL_KEY: "end", C.PRICE_BASIS_KEY: "mid"}


def test_a_mapped_price_column_stays_the_price_basis():
    frame = quote_frame().with_columns(pl.Series("last", [100.0, 100.4]))
    mapping = guess_mapping(frame, "aaa")
    assert mapping.columns[C.CLOSE] == "last"
    out = apply_mapping(frame, mapping)
    assert out[C.CLOSE].to_list() == [100.0, 100.4]
    assert quote_metadata(mapping, out.columns)[C.PRICE_BASIS_KEY] == "trade"


def test_level_one_must_agree_with_the_book():
    frame = book_frame(levels=1).with_columns(pl.Series(C.BID, [99.0, 99.0]))
    mapping = guess_mapping(frame, "aaa")
    assert mapping.columns[C.BID] == C.BID
    with pytest.raises(DataValidationError, match="disagree"):
        apply_mapping(frame, mapping)


def test_declaring_more_levels_than_are_mapped_names_the_missing_columns():
    frame = book_frame(levels=1)
    mapping = guess_mapping(frame, "aaa").model_copy(update={"book_levels": 2})
    with pytest.raises(DataValidationError, match="bid_px_2"):
        apply_mapping(frame, mapping)


def test_start_labelled_bars_are_shifted_to_the_bar_end():
    frame = quote_frame()
    mapping = guess_mapping(frame, "aaa").model_copy(
        update={"frequency": Frequency.MIN1, "bar_label": "start"}
    )
    out = apply_mapping(frame, mapping)
    assert out[C.TIMESTAMP].dt.replace_time_zone(None).to_list() == [
        datetime(2024, 7, 1, 14, 1),
        datetime(2024, 7, 1, 14, 2),
    ]


def test_a_file_without_a_price_or_quotes_is_refused():
    frame = pl.DataFrame({"ts": STAMPS, "symbol": ["AAA", "AAA"], "signal": [1.0, 2.0]})
    mapping = ImportMapping(timestamp="ts", symbol="symbol", extra=["signal"])
    with pytest.raises(DataValidationError, match="close"):
        apply_mapping(frame, mapping)


def test_an_imported_book_round_trips_through_the_service(data_service, tmp_path):
    path = tmp_path / "aaa_book.parquet"
    book_frame(levels=2).write_parquet(path)
    mapping = guess_mapping(book_frame(levels=2), "aaa_book").model_copy(
        update={"frequency": Frequency.MIN1}
    )
    record = data_service.import_file(path, mapping)
    assert record.rows == 2
    assert "bid_px_2" in record.fields
    data = data_service.source("local_file").fetch(
        DataRequest(
            source="local_file",
            dataset=record.id,
            instruments=("AAA",),
            start=STAMPS[0].date(),
            end=STAMPS[-1].date(),
            frequency=Frequency.MIN1,
        )
    )
    assert data.metadata[BAR_LABEL_KEY] == "end"
    assert data.metadata[C.PRICE_BASIS_KEY] == "mid"
    assert not [i for i in data_service.catalog.quality(record.id).issues if i.severity == "error"]
