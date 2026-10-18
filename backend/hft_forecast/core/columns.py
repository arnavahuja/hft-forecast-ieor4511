"""Canonical column names used across the whole code base.

Every table crossing a layer boundary uses these names. Defining them once avoids
string drift between the data layer, engines, analytics and the API.
"""

from __future__ import annotations

from typing import Final

TIMESTAMP: Final = "timestamp"
INSTRUMENT: Final = "instrument_id"
SYMBOL: Final = "symbol"

OPEN: Final = "open"
HIGH: Final = "high"
LOW: Final = "low"
CLOSE: Final = "close"
VOLUME: Final = "volume"
ADJ_CLOSE: Final = "adj_close"
DIVIDEND: Final = "dividend"
SPLIT: Final = "split_ratio"
RETURN: Final = "ret"
DELISTING_RETURN: Final = "dlret"

OHLCV: Final = (OPEN, HIGH, LOW, CLOSE, VOLUME)
PRICE_FIELDS: Final = (OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE)
KEY_COLUMNS: Final = (TIMESTAMP, INSTRUMENT)

# ---- quote (NBBO level 1) fields, all at the bar end unless noted

BID: Final = "bid"
ASK: Final = "ask"
BID_SIZE: Final = "bid_size"
ASK_SIZE: Final = "ask_size"
MID: Final = "mid"
MICROPRICE: Final = "microprice"
SPREAD: Final = "spread"
SPREAD_BPS: Final = "spread_bps"
MID_TWAP: Final = "mid_twap"
N_QUOTES: Final = "n_quotes"
N_TRADES: Final = "n_trades"
VWAP: Final = "vwap"
IS_STALE: Final = "is_stale"

QUOTE_FIELDS: Final = (
    BID,
    ASK,
    BID_SIZE,
    ASK_SIZE,
    MID,
    MICROPRICE,
    SPREAD,
    SPREAD_BPS,
    MID_TWAP,
    N_QUOTES,
    IS_STALE,
)

PRICE_BASIS_KEY: Final = "price_basis"
"""MarketData metadata key: ``"mid"`` when close/open are quote mids, else ``"trade"``."""


def book_columns(n_levels: int) -> tuple[str, ...]:
    """Wide order-book column names for ``n_levels`` levels, best level first.

    Level 1 *is* :data:`BID` / :data:`ASK`; an importer that maps both must check they agree.

    Raises:
        ValueError: If ``n_levels`` is negative.
    """
    if n_levels < 0:
        raise ValueError("n_levels must not be negative")
    return tuple(
        f"{side}_{kind}_{level}"
        for level in range(1, n_levels + 1)
        for side in ("bid", "ask")
        for kind in ("px", "sz")
    )
