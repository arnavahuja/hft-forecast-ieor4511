from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError
from hft_forecast.core.types import DataRequest, Frequency
from hft_forecast.data.importer import guess_mapping, preview
from hft_forecast.data.sources.synthetic import SyntheticSource


def _req(start=date(2021, 1, 1), end=date(2021, 6, 30)) -> DataRequest:
    return DataRequest(
        source="synthetic", dataset="gbm", instruments=("AAA", "BBB"), start=start, end=end
    )


@pytest.fixture
def counted(monkeypatch):
    calls: list[tuple[date, date]] = []
    original = SyntheticSource.fetch

    def fetch(self, request):
        calls.append((request.start, request.end))
        return original(self, request)

    monkeypatch.setattr(SyntheticSource, "fetch", fetch)
    return calls


def test_fetch_caches_and_catalogs(data_service, counted):
    first = data_service.fetch(_req())
    second = data_service.fetch(_req())
    assert len(counted) == 1
    assert first.frame.equals(second.frame)
    records = data_service.catalog.records()
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "synthetic" and rec.instruments == ["AAA", "BBB"]
    assert rec.rows == len(first)
    assert data_service.catalog.quality(rec.id).rows == len(first)
    assert len(data_service.catalog.preview(rec.id, limit=5)) == 5


def test_partial_overlap_fetches_only_missing(data_service, counted):
    data_service.fetch(_req(end=date(2021, 3, 31)))
    data = data_service.fetch(_req(start=date(2021, 2, 1), end=date(2021, 6, 30)))
    assert counted == [(date(2021, 1, 1), date(2021, 3, 31)), (date(2021, 4, 1), date(2021, 6, 30))]
    ts = data.timestamps.astype("datetime64[D]")
    assert ts[0] >= pl.Series([date(2021, 2, 1)]).to_numpy()[0]
    assert str(ts[-1]) == "2021-06-30"
    # a sub-range is now served from cache
    data_service.fetch(_req(start=date(2021, 1, 4), end=date(2021, 5, 1)))
    assert len(counted) == 2


def test_refresh_creates_new_version_and_old_is_loadable(data_service):
    data_service.fetch(_req())
    rec = data_service.catalog.records()[0]
    data_service.refresh(rec.id)
    versions = data_service.cache.versions(rec.id)
    assert [v.version for v in versions] == [1, 2]
    old = data_service.fetch(_req(), cache_version=1)
    assert old.metadata["cache_version"].startswith("v1@")


CSV = """Date,Ticker,Close,Volume,my_signal
2021-01-04,XYZ,10.0,100,0.1
2021-01-05,XYZ,10.5,110,0.2
2021-01-06,XYZ,-1.0,120,0.3
2021-01-05,XYZ,10.5,110,0.2
2021-01-04,QQQ,20.0,100,0.5
2021-01-05,QQQ,21.0,100,0.6
"""


def test_import_wizard_roundtrip(data_service, tmp_path):
    path = tmp_path / "my_data.csv"
    path.write_text(CSV)
    pv = preview(path)
    mapping = guess_mapping(pl.read_csv(path, try_parse_dates=True), "my_data")
    assert pv["mapping"]["timestamp"] == "Date"
    assert mapping.symbol == "Ticker"
    assert mapping.columns == {C.CLOSE: "Close", C.VOLUME: "Volume"}
    assert mapping.extra == ["my_signal"]
    record = data_service.import_file(path, mapping)
    assert record.instruments == ["QQQ", "XYZ"]
    assert "my_signal" in record.fields
    quality = data_service.catalog.quality(record.id)
    checks = {i.check for i in quality.issues}
    assert {"duplicates", "non_positive_price", "monotonic"} <= checks
    data = data_service.fetch(
        DataRequest(
            source="local_file", dataset=record.id, start=date(2021, 1, 1), end=date(2021, 12, 31)
        )
    )
    assert data.has_field("my_signal")
    assert data.frequency is Frequency.D1
    assert len(data) == 5  # duplicate removed
    # one-click re-import with the saved mapping
    again = data_service.import_file(path, data_service.saved_mapping(record.id), record.id)
    assert again.id == record.id


def test_partitions_split_by_instrument_and_month(data_service):
    parts = data_service.partitions(_req(date(2021, 1, 15), date(2021, 3, 2)))
    assert [(p.instruments, p.start, p.end) for p in parts] == [
        (("AAA",), date(2021, 1, 15), date(2021, 1, 31)),
        (("AAA",), date(2021, 2, 1), date(2021, 2, 28)),
        (("AAA",), date(2021, 3, 1), date(2021, 3, 2)),
        (("BBB",), date(2021, 1, 15), date(2021, 1, 31)),
        (("BBB",), date(2021, 2, 1), date(2021, 2, 28)),
        (("BBB",), date(2021, 3, 1), date(2021, 3, 2)),
    ]
    assert len(data_service.partitions(_req(), by="year")) == 2


def test_fetch_partitioned_covers_the_same_rows_as_one_big_fetch(data_service):
    request = _req(date(2021, 1, 4), date(2021, 3, 31))
    keys = list(C.KEY_COLUMNS)
    # the synthetic source draws its own path per request, so only the keys are comparable
    whole = data_service.fetch(request).frame.select(keys).sort(keys)
    parts = data_service.fetch_partitioned(request).frame.select(keys).sort(keys)
    assert parts.equals(whole)
    assert data_service.cached_file(data_service.partitions(request)[0]).exists()


def test_fetch_partitioned_rejects_an_unknown_unit(data_service):
    with pytest.raises(DataError):
        data_service.partitions(_req(), by="week")
