"""WRDS data source: CRSP, Compustat, Fama-French factors and point-in-time universes.

Credentials: the WRDS user name comes from ``WRDS_USERNAME`` in ``.env``; the password is read
by the ``wrds`` library from ``~/.pgpass``. Nothing secret is ever stored by this package.

Correctness:
    * PERMNO is the stable identifier (``instrument_id``); tickers are metadata.
    * Delisted securities are included and delisting returns are merged into ``ret``
      (``(1 + ret)(1 + dlret) - 1`` on the delisting date), so results are survivorship-free.
    * Compustat fundamentals are stamped with a conservative public-availability date
      (``datadate + availability_lag_days``), never the fiscal period end.
    * Index universes are point-in-time (``crsp.dsp500list`` start/end dates) and returned as a
      ``universe_member`` field.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd
import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import DataError, DataSourceUnavailableError
from hft_forecast.core.interfaces import DataSource
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
    OptionRight,
)
from hft_forecast.data.normalize import normalize_frame

UNIVERSE_FIELD: Final = "universe_member"
FACTORS_ID: Final = "FF"
SHARES_SCALE: Final = 1_000.0
"""CRSP ``shrout`` is in thousands of shares."""
SEARCH_LIMIT: Final = 25
COMMON_SHARE_CODES: Final = (10, 11)
MAJOR_EXCHANGES: Final = (1, 2, 3)

FACTOR_COLUMNS: Final = {
    "mktrf": "mkt_rf",
    "smb": "smb",
    "hml": "hml",
    "rmw": "rmw",
    "cma": "cma",
    "rf": "rf",
    "umd": "mom",
}
FUNDAMENTAL_COLUMNS: Final = {
    "ceq": "common_equity",
    "at": "total_assets",
    "lt": "total_liabilities",
    "sale": "sales",
    "revt": "revenue",
    "cogs": "cogs",
    "ni": "net_income",
    "oibdp": "ebitda",
    "capx": "capex",
    "dvc": "dividends_common",
    "seq": "stockholders_equity",
    "pstk": "preferred_stock",
    "pstkrv": "preferred_redemption",
    "pstkl": "preferred_liquidating",
    "txditc": "deferred_taxes",
}
UNIVERSE_SP500: Final = "sp500"
UNIVERSE_COMMON: Final = "crsp_common"
KNOWN_UNIVERSES: Final = (UNIVERSE_SP500, UNIVERSE_COMMON)
PERFORMANCE_DELISTING: Final = (500, *range(520, 585))
"""Delisting codes treated as performance-related (Shumway 1997)."""
NASDAQ: Final = 3
INDEX_SERIES: Final = {
    "CRSP_VW": "vwretd",
    "CRSP_VW_EXDIV": "vwretx",
    "CRSP_EW": "ewretd",
    "CRSP_EW_EXDIV": "ewretx",
    "SP500": "sprtrn",
}
TREASURY_SERIES: Final = (
    "b1ret",
    "b2ret",
    "b5ret",
    "b7ret",
    "b10ret",
    "b20ret",
    "b30ret",
    "t30ret",
    "t90ret",
)
INDEX_BASE: Final = 100.0


class WrdsParams(SourceParams):
    """WRDS source parameters (no secrets)."""

    availability_lag_days: int = Field(
        180, ge=0, le=730, description="Days after fiscal year end before fundamentals are used"
    )
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


def _pgpass_path() -> Path:
    return Path(os.environ.get("PGPASSFILE", Path.home() / ".pgpass"))


def _num(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series[float]:
    """Numeric column (or a constant column when missing)."""
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _pid(value: object) -> str:
    """PERMNO as a string id (WRDS may return floats)."""
    return str(int(float(str(value))))


def _month_start(day: date) -> date:
    """First day of the month (monthly tables may be dated at month start)."""
    return day.replace(day=1)


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


def fundamentals_to_canonical(frame: pd.DataFrame, lag_days: int) -> pl.DataFrame:
    """Compustat rows (already linked to permno) stamped at their availability date."""
    df = frame.copy()
    available = pd.to_datetime(df["datadate"]) + pd.Timedelta(days=lag_days)
    out = pd.DataFrame(
        {
            C.TIMESTAMP: available.dt.tz_localize("UTC"),
            C.INSTRUMENT: df["permno"].astype("Int64").astype(str),
            "fiscal_period_end": pd.to_datetime(df["datadate"]).dt.strftime("%Y-%m-%d"),
        }
    )
    for src, dst in FUNDAMENTAL_COLUMNS.items():
        if src in df:
            out[dst] = _num(df, src)
    out["fiscal_year"] = pd.to_datetime(df["datadate"]).dt.year.astype(float)
    out["book_equity"] = book_equity(df)
    return normalize_frame(pl.from_pandas(out))


def series_to_canonical(frame: pd.DataFrame, series: dict[str, str]) -> pl.DataFrame:
    """Return series (``{instrument: column}``) to instruments priced as a cumulative index."""
    parts = []
    dates = pd.to_datetime(frame["date"]).dt.tz_localize("UTC")
    for inst, col in series.items():
        if col not in frame:
            continue
        ret = _num(frame, col)
        level = INDEX_BASE * (1.0 + ret.fillna(0.0)).cumprod()
        parts.append(
            pd.DataFrame({C.TIMESTAMP: dates, C.INSTRUMENT: inst, C.CLOSE: level, C.RETURN: ret})
        )
    if not parts:
        raise DataError("None of the requested series exist", details={"known": list(series)})
    return normalize_frame(pl.from_pandas(pd.concat(parts, ignore_index=True)))


def book_equity(df: pd.DataFrame) -> pd.Series[float]:
    """Fama-French book equity: SE + deferred taxes - preferred stock (millions).

    SE is ``seq``, else ``ceq + pstk``, else ``at - lt``; preferred stock is ``pstkrv``, else
    ``pstkl``, else ``pstk`` (missing counts as 0).
    """
    se = _num(df, "seq")
    se = se.fillna(_num(df, "ceq") + _num(df, "pstk").fillna(0.0))
    se = se.fillna(_num(df, "at") - _num(df, "lt"))
    pref = _num(df, "pstkrv").fillna(_num(df, "pstkl")).fillna(_num(df, "pstk")).fillna(0.0)
    return se + _num(df, "txditc").fillna(0.0) - pref


def factors_to_canonical(frame: pd.DataFrame) -> pl.DataFrame:
    """Fama-French factor rows to a single-instrument (``FF``) canonical frame."""
    df = frame.copy()
    out = pd.DataFrame(
        {
            C.TIMESTAMP: pd.to_datetime(df["date"]).dt.tz_localize("UTC"),
            C.INSTRUMENT: FACTORS_ID,
        }
    )
    for src, dst in FACTOR_COLUMNS.items():
        if src in df:
            out[dst] = _num(df, src)
    return normalize_frame(pl.from_pandas(out))


TAQ_TZ: Final = "America/New_York"
TAQ_OPEN: Final = "09:30:00"
TAQ_CLOSE: Final = "16:00:00"


def taq_sql(day: date, minutes: int) -> str:
    """Bar aggregation query over one day's TAQ millisecond trade table."""
    table = f"taqm_{day.year}.ctm_{day:%Y%m%d}"
    seconds = 60 * minutes
    return (
        "select sym_root, floor(extract(epoch from time_m) / "
        f"{seconds}) as bucket, "
        "(array_agg(price order by time_m))[1] as open, max(price) as high, "
        "min(price) as low, (array_agg(price order by time_m desc))[1] as close, "
        f"sum(size) as volume from {table} "
        "where sym_root in %(syms)s and sym_suffix is null and tr_corr = '00' "
        f"and time_m >= '{TAQ_OPEN}' and time_m < '{TAQ_CLOSE}' "
        "group by sym_root, bucket order by sym_root, bucket"
    )


def taq_to_canonical(frame: pd.DataFrame, day: date, minutes: int) -> pl.DataFrame:
    """TAQ bar rows (bucket = seconds-since-midnight / bar length) to canonical form."""
    local = pd.Timestamp(day) + pd.to_timedelta(
        frame["bucket"].astype(float) * 60 * minutes, unit="s"
    )
    out = pd.DataFrame(
        {
            C.TIMESTAMP: local.dt.tz_localize(TAQ_TZ).dt.tz_convert("UTC"),
            C.INSTRUMENT: frame["sym_root"].astype(str),
            C.OPEN: _num(frame, "open"),
            C.HIGH: _num(frame, "high"),
            C.LOW: _num(frame, "low"),
            C.CLOSE: _num(frame, "close"),
            C.VOLUME: _num(frame, "volume"),
        }
    )
    return normalize_frame(pl.from_pandas(out))


OPTION_MULTIPLIER: Final = 100.0
OPTIONM_STRIKE_SCALE: Final = 1000.0


def optionm_to_canonical(frame: pd.DataFrame) -> tuple[pl.DataFrame, dict[str, Instrument]]:
    """OptionMetrics rows (with a ``ticker`` column) to a canonical chain plus metadata."""
    df = frame.copy()
    bid, ask = _num(df, "best_bid"), _num(df, "best_offer")
    strike = _num(df, "strike_price") / OPTIONM_STRIKE_SCALE
    right = np.where(df["cp_flag"].astype(str).str.upper() == "C", "call", "put")
    expiry = pd.to_datetime(df["exdate"]).dt.strftime("%Y-%m-%d")
    ids = df["optionid"].map(_pid)
    out = pd.DataFrame(
        {
            C.TIMESTAMP: pd.to_datetime(df["date"]).dt.tz_localize("UTC"),
            C.INSTRUMENT: ids,
            "underlying": df["ticker"].astype(str),
            "right": right,
            "expiry": expiry,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "iv": _num(df, "impl_volatility"),
            "delta": _num(df, "delta"),
            "gamma": _num(df, "gamma"),
            "vega": _num(df, "vega"),
            "theta": _num(df, "theta"),
            "open_interest": _num(df, "open_interest"),
            C.VOLUME: _num(df, "volume"),
        }
    )
    meta = {}
    for oid, und, r, k, e in zip(ids, out["underlying"], right, strike, expiry, strict=True):
        if oid not in meta:
            meta[oid] = Instrument(
                id=oid,
                symbol=oid,
                asset_class=AssetClass.OPTION,
                multiplier=OPTION_MULTIPLIER,
                underlying=und,
                strike=float(k),
                expiry=date.fromisoformat(e),
                right=OptionRight(r),
            )
    return normalize_frame(pl.from_pandas(out)), meta


@register(
    "data_source",
    name="wrds",
    version="1.0.0",
    tags=["crsp", "compustat", "fama-french", "academic"],
    capabilities={"asset:equity", "freq:daily", "needs:fundamentals"},
)
class WrdsSource(DataSource):
    """CRSP stock files, Compustat fundamentals, Fama-French factors and S&P 500 membership."""

    Params = WrdsParams
    params: WrdsParams
    source_version: ClassVar[str] = "wrds-1"
    _connection: Any = None

    # ---- connection

    def available(self) -> tuple[bool, str]:
        """WRDS needs the ``wrds`` package, a user name and ``~/.pgpass``."""
        try:
            import wrds  # noqa: F401
        except ImportError:
            return False, "Install the optional extra: pip install -e '.[wrds]'"
        if not self.env.wrds_username:
            return False, "Set WRDS_USERNAME in .env"
        if not _pgpass_path().exists():
            return False, (
                '~/.pgpass not found: run `python -c "import wrds; '
                'wrds.Connection()"` once in a terminal to create it'
            )
        return True, ""

    def _connect(self) -> Any:
        ok, reason = self.available()
        if not ok:
            raise DataSourceUnavailableError(reason)
        if self._connection is None:
            import wrds

            self._connection = wrds.Connection(wrds_username=self.env.wrds_username)
        return self._connection

    def _query(
        self, sql: str, params: dict[str, Any] | None = None, date_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Run SQL on WRDS (isolated so tests can substitute recorded responses)."""
        frame: pd.DataFrame = self._connect().raw_sql(sql, params=params, date_cols=date_cols)
        return frame

    def test_connection(self) -> dict[str, Any]:
        """Connection check used by the Settings page."""
        try:
            self._query("select 1 as ok")
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "message": f"Connected as {self.env.wrds_username}"}

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
            DatasetInfo(
                source="wrds",
                dataset="compustat_annual",
                description="Compustat annual fundamentals linked to PERMNO, stamped "
                "at availability date",
                frequencies=(Frequency.D1,),
                fields=(*FUNDAMENTAL_COLUMNS.values(), "book_equity", "fiscal_year"),
                survivorship_bias_free=True,
                quality="research",
            ),
            DatasetInfo(
                source="wrds",
                dataset="taq_bars",
                description="Intraday OHLCV bars aggregated from TAQ trades "
                "(regular hours, tr_corr = 00)",
                frequencies=(Frequency.MIN1, Frequency.MIN5, Frequency.MIN15, Frequency.H1),
                fields=C.OHLCV,
                quality="research",
            ),
            DatasetInfo(
                source="wrds",
                dataset="optionm_chain",
                description="OptionMetrics daily option chains (mid, IV, Greeks, OI) "
                "for ticker underlyings",
                frequencies=(Frequency.D1,),
                asset_classes=(AssetClass.OPTION,),
                fields=(
                    "mid",
                    "bid",
                    "ask",
                    "iv",
                    "delta",
                    "gamma",
                    "vega",
                    "theta",
                    "open_interest",
                    C.VOLUME,
                    "strike",
                ),
                quality="research",
            ),
            DatasetInfo(
                source="wrds",
                dataset="ff_factors",
                description="Fama-French 5 factors plus momentum and the T-bill rate "
                "(daily or monthly)",
                frequencies=(Frequency.D1, Frequency.MO1),
                asset_classes=(AssetClass.OTHER,),
                fields=tuple(FACTOR_COLUMNS.values()),
                quality="research",
            ),
            DatasetInfo(
                source="wrds",
                dataset="crsp_index",
                description="CRSP market indices as tradable series: " + ", ".join(INDEX_SERIES),
                frequencies=(Frequency.D1, Frequency.MO1),
                asset_classes=(AssetClass.INDEX,),
                fields=(C.CLOSE, C.RETURN),
                quality="research",
                notes="Instruments: " + ", ".join(INDEX_SERIES),
            ),
            DatasetInfo(
                source="wrds",
                dataset="crsp_treasury",
                description="CRSP constant-maturity Treasury returns (crsp.mcti)",
                frequencies=(Frequency.MO1,),
                asset_classes=(AssetClass.INDEX,),
                fields=(C.CLOSE, C.RETURN),
                quality="research",
                notes="Instruments: " + ", ".join(c.upper() for c in TREASURY_SERIES),
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
        handlers = {
            "crsp_daily": self._crsp,
            "crsp_monthly": self._crsp,
            "compustat_annual": self._compustat,
            "ff_factors": self._factors,
            "taq_bars": self._taq,
            "optionm_chain": self._optionm,
            "crsp_index": self._index,
            "crsp_treasury": self._treasury,
        }
        handler = handlers.get(request.dataset)
        if handler is None:
            raise DataError(f"Unknown WRDS dataset '{request.dataset}'")
        return handler(request)

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

    def _compustat(self, request: DataRequest) -> MarketData:
        permnos, _ = self._resolve_permnos(request.instruments, request.start, request.end)
        lag = self.params.availability_lag_days
        start = request.start - timedelta(days=lag + 366)
        fields = ", ".join(f"f.{c}" for c in FUNDAMENTAL_COLUMNS)
        where_permno = f"and l.lpermno in ({_sql_list(permnos)})" if permnos else ""
        frame = self._query(
            f"select l.lpermno as permno, f.datadate, {fields} "
            "from comp.funda f join crsp.ccmxpf_linktable l on f.gvkey = l.gvkey "
            "where f.indfmt = 'INDL' and f.datafmt = 'STD' and f.popsrc = 'D' "
            "and f.consol = 'C' and l.linktype in ('LU', 'LC') and l.linkprim in ('P', 'C') "
            "and f.datadate >= l.linkdt and (f.datadate <= l.linkenddt or l.linkenddt is null) "
            f"and f.datadate between %(start)s and %(end)s {where_permno}",
            params={"start": start, "end": request.end},
            date_cols=["datadate"],
        )
        canon = fundamentals_to_canonical(frame, lag)
        return MarketData(
            canon,
            Frequency.D1,
            metadata={
                "source": "wrds",
                "dataset": "compustat_annual",
                "availability_lag_days": lag,
            },
        )

    def _factors(self, request: DataRequest) -> MarketData:
        monthly = request.frequency is Frequency.MO1
        five, mom = (
            ("ff.fivefactors_monthly", "ff.factors_monthly")
            if monthly
            else ("ff.fivefactors_daily", "ff.factors_daily")
        )
        frame = self._query(
            "select a.date, a.mktrf, a.smb, a.hml, a.rmw, a.cma, a.rf, b.umd "
            f"from {five} a left join {mom} b on a.date = b.date "
            "where a.date between %(start)s and %(end)s",
            params={
                "start": _month_start(request.start) if monthly else request.start,
                "end": request.end,
            },
            date_cols=["date"],
        )
        return MarketData(
            factors_to_canonical(frame),
            request.frequency if monthly else Frequency.D1,
            {
                FACTORS_ID: Instrument(
                    id=FACTORS_ID, symbol=FACTORS_ID, asset_class=AssetClass.OTHER
                )
            },
            metadata={"source": "wrds", "dataset": "ff_factors"},
        )

    @staticmethod
    def _series_ids(request: DataRequest, known: list[str]) -> list[str]:
        wanted = [i.upper() for i in request.instruments] or known
        unknown = sorted(set(wanted) - set(known))
        if unknown:
            raise DataError(f"Unknown series {unknown}", details={"known": known})
        return wanted

    def _index(self, request: DataRequest) -> MarketData:
        ids = self._series_ids(request, list(INDEX_SERIES))
        table = "crsp.msi" if request.frequency is Frequency.MO1 else "crsp.dsi"
        cols = ", ".join(sorted({INDEX_SERIES[i] for i in ids}))
        frame = self._query(
            f"select date, {cols} from {table} "
            "where date between %(start)s and %(end)s order by date",
            params={"start": request.start, "end": request.end},
            date_cols=["date"],
        )
        canon = series_to_canonical(frame, {i: INDEX_SERIES[i] for i in ids})
        meta = {i: Instrument(id=i, symbol=i, asset_class=AssetClass.INDEX) for i in ids}
        freq = Frequency.MO1 if request.frequency is Frequency.MO1 else Frequency.D1
        return MarketData(
            canon,
            freq,
            meta,
            metadata={"source": "wrds", "dataset": "crsp_index", "adjustment": "total_return"},
        )

    def _treasury(self, request: DataRequest) -> MarketData:
        known = [c.upper() for c in TREASURY_SERIES]
        ids = self._series_ids(request, known)
        cols = ", ".join(i.lower() for i in ids)
        frame = self._query(
            f"select caldt as date, {cols} from crsp.mcti "
            "where caldt between %(start)s and %(end)s order by caldt",
            params={"start": _month_start(request.start), "end": request.end},
            date_cols=["date"],
        )
        canon = series_to_canonical(frame, {i: i.lower() for i in ids})
        meta = {i: Instrument(id=i, symbol=i, asset_class=AssetClass.INDEX) for i in ids}
        return MarketData(
            canon,
            Frequency.MO1,
            meta,
            metadata={"source": "wrds", "dataset": "crsp_treasury", "adjustment": "total_return"},
        )

    def _taq(self, request: DataRequest) -> MarketData:
        minutes = request.frequency.minutes
        if minutes is None:
            raise DataError("TAQ bars need an intraday frequency")
        syms = tuple(i.upper() for i in request.instruments)
        if not syms:
            raise DataError("TAQ bars need ticker symbols")
        frames = []
        for day in TradingCalendar().sessions(request.start, request.end):
            rows = self._query(taq_sql(day, minutes), params={"syms": syms})
            if len(rows):
                frames.append(taq_to_canonical(rows, day, minutes))
        if not frames:
            raise DataError("TAQ returned no trades for the request")
        return MarketData(
            pl.concat(frames),
            request.frequency,
            {s: Instrument(id=s, symbol=s) for s in syms},
            metadata={"source": "wrds", "dataset": "taq_bars", "quality": "research"},
        )

    def _optionm(self, request: DataRequest) -> MarketData:
        tickers = tuple(i.upper() for i in request.instruments if not i.isdigit())
        if not tickers:
            raise DataError("OptionMetrics chains need ticker underlyings")
        secids = self._query(
            "select distinct secid, ticker from optionm.secnmd where ticker in %(t)s",
            params={"t": tickers},
        )
        if secids.empty:
            raise DataError("No OptionMetrics security ids for these tickers")
        frames = []
        for year in range(request.start.year, request.end.year + 1):
            rows = self._query(
                "select o.date, o.exdate, o.cp_flag, o.strike_price, o.best_bid, o.best_offer, "
                "o.impl_volatility, o.delta, o.gamma, o.vega, o.theta, o.open_interest, "
                f"o.volume, o.optionid, o.secid from optionm.opprcd{year} o "
                "where o.secid in %(s)s and o.date between %(start)s and %(end)s",
                params={
                    "s": tuple(int(x) for x in secids["secid"]),
                    "start": request.start,
                    "end": request.end,
                },
                date_cols=["date", "exdate"],
            )
            if len(rows):
                frames.append(rows.merge(secids, on="secid", how="left"))
        if not frames:
            raise DataError("OptionMetrics returned no rows")
        chain, meta = optionm_to_canonical(pd.concat(frames, ignore_index=True))
        return MarketData(
            chain,
            Frequency.D1,
            meta,
            metadata={"source": "wrds", "dataset": "optionm_chain", "quality": "research"},
        )
