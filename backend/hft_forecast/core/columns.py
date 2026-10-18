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
