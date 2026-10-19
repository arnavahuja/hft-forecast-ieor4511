"""A fake TAQ backend: deterministic quote and trade ticks, aggregated the way WRDS would.

The source issues one aggregate SQL per session, so the fake has to do the aggregation too.
It applies the same quote filters the SQL applies and plants rows that each filter must
reject, plus a quote gap per session so the client-side forward fill is exercised.
"""

from __future__ import annotations

import re
from datetime import date, time
from typing import Any

import numpy as np
import pandas as pd

SYMBOLS = ("AAA", "BBB")
GAP_SYMBOL = "BBB"
GAP_START, GAP_END = 11 * 3600, 11 * 3600 + 120
"""Local seconds since midnight of the planted quote gap (two minutes, no quotes)."""
QUOTES_PER_SESSION = 2_000
BAD_PER_SESSION = 3
"""Rows planted per session per symbol for each of the three rejecting filters."""


def _ticks(day: date, symbol: str, open_s: int, close_s: int) -> pd.DataFrame:
    """One session of quotes for one symbol, including rows the filters must reject."""
    rng = np.random.default_rng(abs(hash((day.toordinal(), symbol))) % 2**32)
    t = np.sort(rng.uniform(open_s, close_s, QUOTES_PER_SESSION))
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, QUOTES_PER_SESSION)))
    spread = mid * 2e-4
    frame = pd.DataFrame(
        {
            "sym_root": symbol,
            "t": t,
            "bid": mid - spread / 2,
            "ask": mid + spread / 2,
            "bidsiz": 100.0,
            "asksiz": 200.0,
            "qu_cond": "R",
        }
    )
    if symbol == GAP_SYMBOL:
        frame = frame[(frame.t < GAP_START) | (frame.t >= GAP_END)]
    bad = pd.DataFrame(
        {
            "sym_root": symbol,
            "t": np.repeat(np.linspace(open_s + 1, close_s - 1, BAD_PER_SESSION), 3),
            "bid": 100.0,
            "ask": 100.02,
            "bidsiz": 100.0,
            "asksiz": 100.0,
            "qu_cond": "R",
        }
    )
    bad.iloc[0::3, bad.columns.get_loc("bidsiz")] = 0.0  # non_positive
    bad.iloc[1::3, bad.columns.get_loc("ask")] = 99.0  # locked_or_crossed
    bad.iloc[2::3, bad.columns.get_loc("qu_cond")] = "Y"  # quote_condition
    return pd.concat([frame, bad], ignore_index=True).sort_values("t", ignore_index=True)


def _filtered(
    ticks: pd.DataFrame, conds: tuple[str, ...], max_abs: float, max_frac: float
) -> pd.Series[bool]:
    mid = (ticks.bid + ticks.ask) / 2
    spread = ticks.ask - ticks.bid
    return (
        (ticks.bid > 0)
        & (ticks.ask > 0)
        & (ticks.bidsiz > 0)
        & (ticks.asksiz > 0)
        & (ticks.ask > ticks.bid)
        & (spread <= max_abs)
        & (spread / mid <= max_frac)
        & ticks.qu_cond.isin(conds)
    )


class FakeTaq:
    """Answers the SQL ``WrdsTaqSource`` issues, without a WRDS account."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(
        self, sql: str, params: dict[str, Any] | None = None, date_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Dispatch on the shape of the query."""
        self.queries.append(sql)
        params = params or {}
        stamp = re.search(r"taqm_\d{4}\.\w+?_(\d{4})(\d{2})(\d{2})", sql)
        assert stamp is not None, sql
        day = date(*(int(g) for g in stamp.groups()))
        open_s, close_s = _seconds(params["open"]), _seconds(params["close"])
        symbols = tuple(params["syms"])
        if "ctm_" in sql:
            return self._trades(day, symbols, open_s, close_s, _bar_seconds(sql))
        ticks = pd.concat(
            [_ticks(day, s, open_s, close_s) for s in symbols if s in SYMBOLS], ignore_index=True
        )
        if ticks.empty:
            return pd.DataFrame()
        ok = _filtered(ticks, tuple(params["conds"]), params["max_abs"], params["max_frac"])
        if "count(*) filter" in sql:
            return self._counts(ticks, ok)
        if "with q as" in sql:
            return self._bars(ticks[ok], _bar_seconds(sql))
        return self._raw(ticks[ok])

    @staticmethod
    def _counts(ticks: pd.DataFrame, ok: pd.Series[bool]) -> pd.DataFrame:
        bad = ticks[~ok]
        return pd.DataFrame(
            [
                {
                    "non_positive": int(((bad.bidsiz <= 0) | (bad.asksiz <= 0)).sum()),
                    "locked_or_crossed": int((bad.ask <= bad.bid).sum()),
                    "quote_condition": int((bad.qu_cond != "R").sum()),
                    "wide_spread": 0,
                }
            ]
        )

    @staticmethod
    def _bars(ticks: pd.DataFrame, seconds: int) -> pd.DataFrame:
        df = ticks.assign(bucket=np.floor(ticks.t / seconds), mid=(ticks.bid + ticks.ask) / 2)
        df = df.sort_values(["sym_root", "bucket", "t"])
        grouped = df.groupby(["sym_root", "bucket"], as_index=False)
        out = grouped.agg(
            bid=("bid", "last"),
            ask=("ask", "last"),
            bidsiz=("bidsiz", "last"),
            asksiz=("asksiz", "last"),
            open=("mid", "first"),
            high=("mid", "max"),
            low=("mid", "min"),
            n_quotes=("mid", "size"),
            mid_twap=("mid", "mean"),
        )
        return out.sort_values(["sym_root", "bucket"], ignore_index=True)

    @staticmethod
    def _raw(ticks: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sym_root": ticks.sym_root.to_numpy(),
                "time_m": [_as_time(v) for v in ticks.t],
                "bid": ticks.bid.to_numpy(),
                "ask": ticks.ask.to_numpy(),
                "bidsiz": ticks.bidsiz.to_numpy(),
                "asksiz": ticks.asksiz.to_numpy(),
            }
        )

    @staticmethod
    def _trades(
        day: date, symbols: tuple[str, ...], open_s: int, close_s: int, seconds: int
    ) -> pd.DataFrame:
        rows = []
        for symbol in symbols:
            if symbol not in SYMBOLS:
                continue
            ticks = _ticks(day, symbol, open_s, close_s)
            price = (ticks.bid + ticks.ask) / 2
            df = pd.DataFrame(
                {"sym_root": symbol, "bucket": np.floor(ticks.t / seconds), "price": price}
            )
            grouped = df.groupby(["sym_root", "bucket"], as_index=False)
            agg = grouped.agg(
                open=("price", "first"),
                high=("price", "max"),
                low=("price", "min"),
                close=("price", "last"),
                n_trades=("price", "size"),
                vwap=("price", "mean"),
            )
            rows.append(agg.assign(volume=agg.n_trades * 100.0))
        return pd.concat(rows, ignore_index=True)


def _seconds(value: time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def _as_time(seconds: float) -> time:
    total = int(seconds)
    return time(total // 3600, total % 3600 // 60, total % 60)


def _bar_seconds(sql: str) -> int:
    return int(re.search(r"floor\(.*?/ (\d+)\)", sql).group(1))  # type: ignore[union-attr]
