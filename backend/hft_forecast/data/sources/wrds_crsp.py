"""WRDS data source: CRSP daily and monthly stock files and point-in-time universes.

Correctness:
    * PERMNO is the stable identifier (``instrument_id``); tickers are metadata.
    * Delisted securities are included and delisting returns are merged into ``ret``
      (``(1 + ret)(1 + dlret) - 1`` on the delisting date), so results are survivorship-free.
    * Index universes are point-in-time (``crsp.dsp500list`` start/end dates) and returned as a
      ``universe_member`` field.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar, Final

import numpy as np
import pandas as pd
import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError
from hft_forecast.core.params import SourceParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import (
    Adjustment,
    AssetClass,
    DataRequest,
    DatasetInfo,
    Frequency,
    Instrument,
    MarketData,
)
from hft_forecast.data.normalize import normalize_frame
from hft_forecast.data.sources.wrds_connection import WrdsConnection

UNIVERSE_FIELD: Final = "universe_member"
SHARES_SCALE: Final = 1_000.0
"""CRSP ``shrout`` is in thousands of shares."""
SEARCH_LIMIT: Final = 25
COMMON_SHARE_CODES: Final = (10, 11)
MAJOR_EXCHANGES: Final = (1, 2, 3)

UNIVERSE_SP500: Final = "sp500"
UNIVERSE_COMMON: Final = "crsp_common"
KNOWN_UNIVERSES: Final = (UNIVERSE_SP500, UNIVERSE_COMMON)
PERFORMANCE_DELISTING: Final = (500, *range(520, 585))
"""Delisting codes treated as performance-related (Shumway 1997)."""
NASDAQ: Final = 3


class WrdsParams(SourceParams):
    """WRDS source parameters (no secrets)."""

    common_stock_only: bool = Field(True, description="Restrict universes to share codes 10/11")
    share_codes: list[int] = Field(
        default_factory=lambda: list(COMMON_SHARE_CODES),
        description="Share codes (shrcd) of the 'crsp_common' universe",
    )
    exchanges: list[int] = Field(
        default_factory=lambda: list(MAJOR_EXCHANGES),
        description="Exchange codes (exchcd) of the 'crsp_common' universe (1 NYSE, 2 AMEX, "
        "3 NASDAQ)",
    )
    fill_missing_delisting: bool = Field(
        True,
        description="Use a fixed return when a performance-related delisting "
        "(codes 500, 520-584) has no dlret",
    )
    delisting_fill_nyse_amex: float = Field(
        -0.30, ge=-1, le=0, description="Fill for NYSE/AMEX delistings"
    )
    delisting_fill_nasdaq: float = Field(
        -0.55, ge=-1, le=0, description="Fill for NASDAQ delistings"
    )


def _num(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series[float]:
    """Numeric column (or a constant column when missing)."""
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _pid(value: object) -> str:
    """PERMNO as a string id (WRDS may return floats)."""
    return str(int(float(str(value))))


def _sql_list(values: list[str] | tuple[str, ...]) -> str:
    return ",".join(str(int(v)) for v in values)


def fill_delisting_returns(delist: pd.DataFrame, params: WrdsParams) -> pd.DataFrame:
    """Fill missing ``dlret`` of performance-related delistings by exchange; drop the rest."""
    if delist.empty:
        return delist
    d = delist.copy()
    dlret = _num(d, "dlret")
    if params.fill_missing_delisting and "dlstcd" in d:
        code = _num(d, "dlstcd")
        exch = _num(d, "exchcd")
        fill = np.where(
            exch == NASDAQ, params.delisting_fill_nasdaq, params.delisting_fill_nyse_amex
        )
        dlret = dlret.where(dlret.notna() | ~code.isin(PERFORMANCE_DELISTING), fill)
    d["dlret"] = dlret
    return d[d["dlret"].notna()]


def merge_delisting(
    daily: pd.DataFrame, delist: pd.DataFrame, monthly: bool = False
) -> pd.DataFrame:
    """Merge delisting returns into ``ret`` on the delisting date (or append that date).

    Monthly files are dated on the month's last trading day, so a delisting is matched to
    the row of its month (or appended at that month's last date in the file).
    """
    if delist.empty:
        return daily
    d = delist.rename(columns={"dlstdt": "date"})[["permno", "date", "dlret"]]
    d = d[d["permno"].isin(daily["permno"].unique())]
    if monthly and len(daily):
        month = pd.to_datetime(daily["date"]).dt.to_period("M")
        last = pd.Series(pd.to_datetime(daily["date"]).values, index=month).groupby(level=0).max()
        d_month = pd.to_datetime(d["date"]).dt.to_period("M")
        d = d.assign(date=d_month.map(last).values).dropna(subset=["date"])
    merged = daily.merge(d, on=["permno", "date"], how="outer")
    ret = pd.to_numeric(merged["ret"], errors="coerce")
    dl = pd.to_numeric(merged["dlret"], errors="coerce")
    merged["ret"] = np.where(dl.notna(), (1 + ret.fillna(0.0)) * (1 + dl) - 1, ret)
    return merged.drop(columns=["dlret"])


def crsp_to_canonical(frame: pd.DataFrame, adjustment: Adjustment) -> pl.DataFrame:
    """Map CRSP daily/monthly stock-file rows to the canonical schema."""
    df = frame.copy()
    df["close"] = _num(df, "prc").abs()
    cfac = _num(df, "cfacpr", 1.0).replace(0, np.nan)
    df["shrout"] = _num(df, "shrout")
    df["market_cap"] = df["close"] * df["shrout"] * SHARES_SCALE
    out = pd.DataFrame(
        {
            C.TIMESTAMP: pd.to_datetime(df["date"]).dt.tz_localize("UTC"),
            C.INSTRUMENT: df["permno"].astype("Int64").astype(str),
            C.CLOSE: df["close"],
            C.RETURN: _num(df, "ret"),
            C.VOLUME: _num(df, "vol"),
            "market_cap": df["market_cap"],
            "shares_outstanding": df["shrout"] * SHARES_SCALE,
            "cfacpr": cfac,
        }
    )
    for src, dst in (("openprc", C.OPEN), ("askhi", C.HIGH), ("bidlo", C.LOW)):
        if src in df:
            out[dst] = _num(df, src).abs()
    if adjustment is not Adjustment.RAW:
        for col in (C.OPEN, C.HIGH, C.LOW, C.CLOSE):
            if col in out:
                out[col] = out[col] / out["cfacpr"].fillna(1.0)
        out[C.VOLUME] = out[C.VOLUME] * out["cfacpr"].fillna(1.0)
    if UNIVERSE_FIELD in df:
        out[UNIVERSE_FIELD] = df[UNIVERSE_FIELD].astype(float)
    return normalize_frame(pl.from_pandas(out.drop(columns=["cfacpr"])))


def membership_flags(daily: pd.DataFrame, spans: pd.DataFrame) -> pd.Series[float]:
    """1.0 on dates where the permno is inside one of its membership spans."""
    merged = daily[["permno", "date"]].reset_index().merge(spans, on="permno", how="left")
    inside = (merged["date"] >= merged["start"]) & (
        merged["ending"].isna() | (merged["date"] <= merged["ending"])
    )
    flags = inside.groupby(merged["index"]).any()
    return flags.reindex(daily.index, fill_value=False).astype(float)


@register(
    "data_source",
    name="wrds",
    version="1.0.0",
    tags=["crsp", "academic"],
    capabilities={"asset:equity", "freq:daily"},
)
class WrdsSource(WrdsConnection):
    """CRSP daily and monthly stock files with point-in-time S&P 500 membership."""

    Params = WrdsParams
    params: WrdsParams
    source_version: ClassVar[str] = "wrds-1"

    # ---- description

    def describe(self) -> list[DatasetInfo]:
        """Datasets offered."""
        survivorship = "Includes delisted securities and delisting returns."
        return [
            DatasetInfo(
                source="wrds",
                dataset="crsp_daily",
                description="CRSP daily stock file",
                frequencies=(Frequency.D1,),
                asset_classes=(AssetClass.EQUITY,),
                fields=(
                    C.OPEN,
                    C.HIGH,
                    C.LOW,
                    C.CLOSE,
                    C.VOLUME,
                    C.RETURN,
                    "market_cap",
                    "shares_outstanding",
                    UNIVERSE_FIELD,
                ),
                survivorship_bias_free=True,
                quality="research",
                notes=survivorship,
            ),
            DatasetInfo(
                source="wrds",
                dataset="crsp_monthly",
                description="CRSP monthly stock file",
                frequencies=(Frequency.MO1,),
                fields=(C.CLOSE, C.VOLUME, C.RETURN, "market_cap", UNIVERSE_FIELD),
                survivorship_bias_free=True,
                quality="research",
                notes=survivorship + " Universes: " + ", ".join(KNOWN_UNIVERSES),
            ),
        ]

    def search_instruments(self, query: str) -> list[Instrument]:
        """Search CRSP names by ticker or company name."""
        q = query.strip().upper()
        if not q:
            return []
        frame = self._query(
            "select distinct on (permno) permno, ticker, comnam, exchcd, nameendt "
            "from crsp.dsenames where upper(ticker) = %(q)s or upper(comnam) like %(like)s "
            "order by permno, nameendt desc limit %(n)s",
            params={"q": q, "like": f"%{q}%", "n": SEARCH_LIMIT},
        )
        return [
            Instrument(
                id=_pid(r["permno"]),
                symbol=str(r["ticker"] or r["permno"]),
                name=str(r["comnam"]) if r["comnam"] else None,
                exchange=_pid(r["exchcd"]) if pd.notna(r["exchcd"]) else None,
            )
            for r in frame.to_dict("records")
        ]

    # ---- fetching

    def fetch(self, request: DataRequest) -> MarketData:
        """Dispatch on the dataset."""
        if request.dataset in ("crsp_daily", "crsp_monthly"):
            return self._crsp(request)
        raise DataError(f"Unknown WRDS dataset '{request.dataset}'")

    def _resolve_permnos(
        self, instruments: tuple[str, ...], start: date, end: date
    ) -> tuple[list[str], dict[str, Instrument]]:
        numeric = [i for i in instruments if i.isdigit()]
        tickers = [i.upper() for i in instruments if not i.isdigit()]
        meta: dict[str, Instrument] = {}
        if tickers:
            names = self._query(
                "select permno, ticker, comnam from crsp.dsenames where ticker in %(t)s "
                "and namedt <= %(end)s and nameendt >= %(start)s",
                params={"t": tuple(tickers), "start": start, "end": end},
            )
            for r in names.to_dict("records"):
                pid = _pid(r["permno"])
                numeric.append(pid)
                meta[pid] = Instrument(id=pid, symbol=str(r["ticker"]), name=str(r["comnam"]))
        return sorted(set(numeric)), meta

    def _universe_filter(self) -> str:
        """SQL condition on ``crsp.msenames`` rows for the 'crsp_common' universe."""
        p = self.params
        shr = _sql_list([str(c) for c in p.share_codes])
        exch = _sql_list([str(c) for c in p.exchanges])
        return f"shrcd in ({shr}) and exchcd in ({exch})"

    def _universe_spans(self, universe: str, start: date, end: date) -> pd.DataFrame:
        if universe == UNIVERSE_SP500:
            return self._query(
                "select permno, start, ending from crsp.dsp500list "
                "where start <= %(end)s and (ending >= %(start)s or ending is null)",
                params={"start": start, "end": end},
                date_cols=["start", "ending"],
            )
        if universe == UNIVERSE_COMMON:
            return self._query(
                "select permno, namedt as start, nameendt as ending from crsp.msenames "
                f"where {self._universe_filter()} "
                "and namedt <= %(end)s and (nameendt >= %(start)s or nameendt is null)",
                params={"start": start, "end": end},
                date_cols=["start", "ending"],
            )
        raise DataError(
            f"Unknown WRDS universe '{universe}'", details={"known": list(KNOWN_UNIVERSES)}
        )

    def _crsp(self, request: DataRequest) -> MarketData:
        daily = request.dataset == "crsp_daily"
        table, delist_table = (
            ("crsp.dsf", "crsp.dsedelist") if daily else ("crsp.msf", "crsp.msedelist")
        )
        spans = None
        permnos, meta = self._resolve_permnos(request.instruments, request.start, request.end)
        if request.universe:
            spans = self._universe_spans(request.universe, request.start, request.end)
            permnos = sorted(set(permnos) | {_pid(p) for p in spans["permno"]})
        if not permnos:
            raise DataError("No PERMNOs to fetch: give tickers, PERMNOs or a universe")
        cols = "permno, date, prc, ret, vol, shrout, cfacpr"
        if daily:
            cols += ", openprc, askhi, bidlo"
        frame = self._query(
            f"select {cols} from {table} where permno in ({_sql_list(permnos)}) "
            "and date between %(start)s and %(end)s",
            params={"start": request.start, "end": request.end},
            date_cols=["date"],
        )
        delist = self._query(
            "select distinct on (d.permno) d.permno, d.dlstdt, d.dlret, d.dlstcd, n.exchcd "
            f"from {delist_table} d left join crsp.msenames n on d.permno = n.permno "
            "and n.namedt <= d.dlstdt "
            f"where d.permno in ({_sql_list(permnos)}) "
            "and d.dlstdt between %(start)s and %(end)s and d.dlstcd <> 100 "
            "order by d.permno, n.namedt desc",
            params={"start": request.start, "end": request.end},
            date_cols=["dlstdt"],
        )
        frame = merge_delisting(
            frame, fill_delisting_returns(delist, self.params), monthly=not daily
        )
        if spans is not None:
            spans = spans.assign(permno=spans["permno"].astype(int))
            frame = frame.assign(permno=frame["permno"].astype(int))
            frame[UNIVERSE_FIELD] = membership_flags(frame, spans)
        canon = crsp_to_canonical(frame, request.adjustment)
        for pid in canon.get_column(C.INSTRUMENT).unique().to_list():
            meta.setdefault(pid, Instrument(id=pid, symbol=pid))
        return MarketData(
            canon,
            Frequency.D1 if daily else Frequency.MO1,
            meta,
            metadata={
                "source": "wrds",
                "adjustment": request.adjustment.value,
                "survivorship_bias_free": True,
                "quality": "research",
                "universe": request.universe or "",
            },
        )
