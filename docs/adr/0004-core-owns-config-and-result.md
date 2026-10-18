# ADR 0004: Core owns BacktestConfig, BacktestResult, chart/metric specs; polars convention

Status: accepted

## Context
Section 4 lists `BacktestConfig`/`BacktestResult` under `engine/base.py`, but the layering
rules say plugins (metrics, charts, overlays) depend only on `core`, and those plugins need
both types.

## Decision
- `BacktestConfig` lives in `core/run_config.py`, `BacktestResult` in `core/results.py`,
  chart specs and metric descriptors in `core/specs.py`. `engine/base.py` re-exports them.
- Dataframe convention: polars for all tables (market data, trades, stored results).
  Numerical work (engines, metrics) uses NumPy arrays taken from wide panels.
  pandas is used only inside adapters for libraries that require it.
- Timestamps: polars `Datetime("us","UTC")`; NumPy `datetime64[us]` implicitly UTC.

## Consequences
One import path for plugin authors (`hft_forecast.core`), and import-linter can enforce that
plugins never import `engine`, `services` or `api`.
