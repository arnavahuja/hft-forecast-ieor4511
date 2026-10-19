"""A fake WRDS backend answering the SQL the WRDS source issues (no network, no account)."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from hft_forecast.core.calendar import TradingCalendar

PERMNOS = (10001, 10002, 10003, 10004, 10005, 10006)
TICKERS = {10001: "AAA", 10002: "BBB", 10003: "CCC", 10004: "DDD", 10005: "EEE", 10006: "FFF"}
DELISTED = 10004
DELIST_DATE = pd.Timestamp("2019-06-28")
JOINS_LATE = 10003
JOIN_DATE = pd.Timestamp("2019-01-02")
NON_COMMON_SHRCD = 12
"""Share code of permno 10002 in ``crsp.msenames`` (not a common stock)."""


def _monthly(dsf: pd.DataFrame) -> pd.DataFrame:
    """Monthly stock file from the daily one (last trading day of each month)."""
    df = dsf.assign(m=dsf.date.dt.to_period("M"))
    g = df.groupby(["permno", "m"])
    out = g.agg(
        date=("date", "max"),
        prc=("prc", "last"),
        vol=("vol", "sum"),
        shrout=("shrout", "last"),
        cfacpr=("cfacpr", "last"),
    ).reset_index()
    out["ret"] = g["ret"].apply(lambda x: float(np.prod(1 + x) - 1)).to_numpy()
    out["prc"] = out["prc"].abs()
    return out.drop(columns="m")


class FakeWrds:
    """Deterministic CRSP daily, monthly, delisting, membership and name tables."""

    def __init__(self, start: date = date(2017, 1, 1), end: date = date(2020, 12, 31)) -> None:
        sessions = pd.to_datetime(TradingCalendar().sessions(start, end))
        rng = np.random.default_rng(3)
        rows = []
        mkt = rng.normal(0.0003, 0.01, len(sessions))
        for k, p in enumerate(PERMNOS):
            rets = 0.8 * mkt + rng.normal(0.0002 * k, 0.012, len(sessions))
            price = 20 * (k + 1) * np.cumprod(1 + rets)
            for d, r, px in zip(sessions, rets, price, strict=True):
                if p == DELISTED and d > DELIST_DATE:
                    break
                # CRSP reports bid/ask midpoints as negative prices
                prc = -px if (p == 10002 and d.day == 15) else px
                rows.append(
                    (
                        p,
                        d,
                        prc,
                        r,
                        1e6,
                        5e4,
                        2.0 if p == 10001 else 1.0,
                        px * 0.99,
                        px * 1.01,
                        px * 0.98,
                    )
                )
        self.dsf = pd.DataFrame(
            rows,
            columns=[
                "permno",
                "date",
                "prc",
                "ret",
                "vol",
                "shrout",
                "cfacpr",
                "openprc",
                "askhi",
                "bidlo",
            ],
        )
        self.dsedelist = pd.DataFrame(
            {
                "permno": [DELISTED],
                "dlstdt": [DELIST_DATE],
                "dlret": [-0.30],
                "dlstcd": [500],
                "exchcd": [1],
            }
        )
        self.msf = _monthly(self.dsf)
        self.dsp500list = pd.DataFrame(
            {
                "permno": list(PERMNOS),
                "start": [pd.Timestamp("2000-01-01")] * 6,
                "ending": [None] * 6,
            }
        )
        self.dsp500list.loc[self.dsp500list.permno == JOINS_LATE, "start"] = JOIN_DATE
        self.dsp500list.loc[self.dsp500list.permno == DELISTED, "ending"] = DELIST_DATE
        self.names = pd.DataFrame(
            {
                "permno": list(PERMNOS),
                "ticker": [TICKERS[p] for p in PERMNOS],
                "comnam": [f"{TICKERS[p]} CORP" for p in PERMNOS],
                "exchcd": [1] * 6,
                "nameendt": [pd.Timestamp("2024-12-31")] * 6,
            }
        )
        self.msenames = pd.DataFrame(
            {
                "permno": list(PERMNOS),
                "start": [pd.Timestamp("1990-01-01")] * 6,
                "ending": [pd.Timestamp("2024-12-31")] * 6,
                "shrcd": [11, NON_COMMON_SHRCD, 10, 11, 10, 11],
                "exchcd": [1, 1, 3, 2, 3, 1],
            }
        )
        self.queries: list[str] = []

    def query(
        self, sql: str, params: dict[str, Any] | None = None, date_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Answer a query by table name."""
        self.queries.append(sql)
        params = params or {}
        start = pd.Timestamp(params.get("start", "1900-01-01"))
        end = pd.Timestamp(params.get("end", "2100-01-01"))
        if "crsp.msf " in sql:
            ids = [int(x) for x in sql.split("permno in (")[1].split(")", maxsplit=1)[0].split(",")]
            f = self.msf
            return f[f.permno.isin(ids) & (f.date >= start) & (f.date <= end)].copy()
        if "crsp.msedelist" in sql:
            return self.dsedelist.copy()
        if "crsp.msenames" in sql and "shrcd in" in sql:
            codes = [
                int(x) for x in sql.split("shrcd in (")[1].split(")", maxsplit=1)[0].split(",")
            ]
            exch = [
                int(x) for x in sql.split("exchcd in (")[1].split(")", maxsplit=1)[0].split(",")
            ]
            f = self.msenames
            return f[f.shrcd.isin(codes) & f.exchcd.isin(exch)][["permno", "start", "ending"]]
        if "crsp.dsf" in sql:
            ids = [int(x) for x in sql.split("permno in (")[1].split(")", maxsplit=1)[0].split(",")]
            f = self.dsf
            return f[f.permno.isin(ids) & (f.date >= start) & (f.date <= end)].copy()
        if "crsp.dsedelist" in sql:
            return self.dsedelist.copy()
        if "crsp.dsp500list" in sql:
            return self.dsp500list.copy()
        if "crsp.dsenames" in sql:
            tickers = params.get("t") or (params.get("q"),)
            return self.names[self.names.ticker.isin(tickers)].copy()
        if "select 1" in sql:
            return pd.DataFrame({"ok": [1]})
        raise AssertionError(f"unexpected SQL: {sql}")
