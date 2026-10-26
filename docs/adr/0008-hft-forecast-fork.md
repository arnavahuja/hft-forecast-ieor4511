# ADR 0008: Fork to hft-forecast-ieor4511

Status: accepted

Supersedes the parts of ADR 0007 that concern fundamentals and factors.

## Context
`backbone` is a general research platform. This capstone needs one thing: short-horizon
(30 s / 5 min / 30 min) price forecasting on TAQ quote data, evaluated statistically and then
economically through the existing engine, by seven people working in three groups in parallel.

## Decision
- **Rename.** Package `backbone` -> `hft_forecast`, CLI `hftf`, env prefix `HFTF_`, entry-point
  group `hft_forecast.plugins`. The data directory default (`data`) is unchanged.
- **Prune to the economic test.** Options, futures, fundamentals and factor attribution are
  deleted: `core/options_math.py`, `data/futures.py`, `data/fundamentals.py`, the options and
  factor metrics/charts, all strategies but `buy_and_hold`, the risk-based and mean-variance
  constructors, and the drawdown/exit/hedging/option/regime overlays. `BacktestConfig` loses
  `factors`, and `DataSpec` loses `options_chain`, `extra_datasets` and `splices`.
  `AUX_FIELDS` becomes empty (it only ever carried option Greeks).
- **Keep** the data cache/catalog/validation spine, both engines (the event engine was
  removed later by ADR 0010), the lookahead check, the
  analytics and research services, the job/run stores, the API and the front end. The front
  end stays view-only; no new pages.
- **Keep CRSP daily and the point-in-time `sp500` universe.** They are how the team picks the
  names it then pulls TAQ quotes for. CRSP's delisting-adjusted `ret` still drives the return
  basis.
- **`freq:subminute`** replaces the deleted `asset:future` / `asset:option` /
  `needs:fundamentals` / `needs:options_chain` capabilities. A plugin without it is rejected
  at config time on `S1`..`S30` data.
- **TAQ lives in its own source.** `data/sources/wrds_taq.py` holds `taq_nbbo_bars` (the
  default for every horizon group) and `taq_trade_bars`; `wrds_crsp.py` keeps CRSP and the
  universes, and both share `wrds_connection.WrdsConnection`. Bars are right-closed and
  labelled at the bar end, quote filters run in SQL before aggregation, and missing buckets
  are forward-filled within the session and flagged by `is_stale`.
- **Per-year NBBO column names.** `NBBO_COLUMNS` maps a year to the column names verified on
  WRDS with `describe_table`; an unverified year raises instead of guessing, because the
  `taqm_*` schema and the `qu_cond` vocabulary both vary by year.
- **`taq_nbbo_raw` is a second plugin** (`wrds_taq_raw`), not a third dataset: `cacheable` is
  a property of the source, and millions of quote rows per symbol-session must not enter the
  Parquet cache. It is capped at 5 symbols x 20 sessions.
- **New `ml` package** (added from milestone M2) sits between `analytics` and `services` in
  the layer order: `ml.economic` needs the engine and `ml.evaluate` needs `analytics.stats`,
  while `services` orchestrates `ml`. Models and feature sets get arrays in and arrays out and
  may not import data loading, CV or evaluation — an import-linter contract enforces this, and
  it is what lets three groups add models without touching shared code.

## Deviations from the design document
Four places where the design could not be implemented as written; each is small and the
reason is recorded in the code:

- **`taq_nbbo_raw` is its own source.** ``cacheable`` is a property of a source, not of a
  dataset, so the raw quotes live in a sibling plugin rather than a third dataset.
- **Effective sample size sums the uniqueness weights** instead of the Kish formula
  ``(sum w)^2 / sum w^2`` the design names. Kish measures how unequal the weights are, so
  perfectly overlapping labels would score ``n`` rather than ``n / steps``.
- **The third cost variant stresses the spread** rather than stacking square-root impact on
  the quoted spread: a run has one slippage model. The breakeven cost multiple answers the
  same question directly.
- **`Forecaster` and `FeatureSet` are abstract base classes**, not typing protocols, so the
  registries can instantiate them the way every other plugin kind is instantiated. The
  signatures are the ones the design specifies, and ``set_feature_names`` keeps ``fit``
  purely numerical while letting a model name a column.

## Consequences
- Golden snapshots collapse to `buy_and_hold`; the remaining engine, overlay and research
  tests parametrize over cost and execution variants instead of over deleted strategies.
- `BackboneError` keeps its name and its `backbone_error` code: renaming an exception
  hierarchy that the API serializes buys nothing and the front end does not match on it.
- Research sweeps in this repo target overlay parameters, since the only shipped strategy has
  none.
- ML runs share the `runs` table with backtests under `kind='ml'`: they store metrics and a
  manifest, never a fabricated `BacktestResult`, and they count as trials.
- The engine gained exactly one interface change, `CostContext.extra_panels` with
  `CostModel.requires` (plus `BarSnapshot.spread`), which is what lets a cost model read the
  quoted spread from the data instead of assuming one.
