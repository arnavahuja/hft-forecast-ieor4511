# hft-forecast-ieor4511 — Design document for the initial template

**Audience:** an AI coding agent that will modify a fork of `github.com/arnavahuja/backbone`
(commit as of 2026-10-18) into `github.com/arnavahuja/hft-forecast-ieor4511`.

**Scope of this document:** design only, no code. It says what to delete, what to keep
untouched, what to modify and what to add, and what the new interfaces are. It is deliberately
a *template*: the seven-person team will write the actual ML models afterwards. Where a
section says "stub", the agent writes the class/function with its full signature, docstring,
validated parameters and a `NotImplementedError` body (or a trivially correct default), plus a
test that the stub registers and type-checks. Nothing more.

**Governing rule for the agent: do not overdo it.** If a piece of behaviour is not listed under
"Add" or "Modify" below, do not add it. No extra models, no extra features, no extra charts,
no new UI pages, no new infrastructure (no MLflow, no W&B, no Docker, no databases beyond
the existing SQLite `RunStore` and Parquet cache). Every addition must pass the existing
`make check` (ruff, mypy --strict, import-linter, pytest).

---

## 0. Decisions already taken (from the project owner)

| Topic | Decision |
|---|---|
| Frontend + API | **Keep**, view-only. No new pages. Used for data quality reports and economic-test runs. |
| Package name | Rename `backbone` → `hft_forecast`. CLI entry point `hftf`. Env prefix `HFTF_`. |
| TAQ access | WRDS Daily TAQ confirmed: trades (`ctm`), quotes (`cqm`) and NBBO (`nbbom`). |
| Initial data scope | 20–50 S&P 500 names, 3–6 months. |
| 30 s grid | Mid-price sampled on a fixed **5 s** clock. 5 min and 30 min groups use **1 min** bars. |
| Level 2 (future) | Format unknown. Generic wide book schema + importer stub; mapping TBD. |
| Framework | PyTorch on CPU now; keep the model interface framework-agnostic so sklearn / other frameworks plug in later. |
| Economic test | Both: per-name long/short (primary) and cross-sectional portfolio (secondary). |
| Experiment tracking | Files under `data/runs/` + existing SQLite `RunStore`. No new infra. |
| Team split | 3 (30 s) / 2 (5 min) / 2 (30 min). |
| Dates | Final report + presentation mid-December 2026. No interim deadlines. ~7.5 weeks from 2026-10-20. |
| Deletions approved | Options, futures, fundamentals + factor attribution, all strategies / overlays / constructors except what the economic test needs. |

---

## 1. What to reuse, modify, add and delete — file by file

Paths are relative to the repo root. `backend/backbone/` becomes `backend/hft_forecast/` after
the rename in §1.0; the tables below use the **new** package path.

### 1.0 Rename (do this first, as one commit)

- `backend/backbone/` → `backend/hft_forecast/`; every `from backbone...` / `import backbone` →
  `hft_forecast`.
- `pyproject.toml`: `name = "hft-forecast-ieor4511"`, `[project.scripts] hftf = "hft_forecast.cli:main"`,
  `[tool.importlinter] root_package = "hft_forecast"`, `[tool.mypy] packages = ["hft_forecast"]`,
  `[tool.coverage.run] source = ["hft_forecast"]`, entry-point group `hft_forecast.plugins`.
- `core/config.py`: `env_prefix="HFTF_"`; keep `WRDS_USERNAME` alias. `.env.example`
  accordingly. `data_dir` default stays `data`.
- `core/registry.py::_origin_of` checks the `hft_forecast.` prefix.
- `core/discovery.py::ENTRY_POINT_GROUP = "hft_forecast.plugins"`.
- `Makefile`: targets unchanged; `openapi` target invokes `hft_forecast.cli`. `scripts/run.sh`
  likewise. `frontend/` only references the API over HTTP, so nothing changes there except
  the regenerated `openapi.json` / `schema.d.ts` (`make openapi`).
- `README.md`: rewrite (see §1.9).
- `docs/`: keep ADRs 0001–0007 as history; add ADR 0008 (this document's decisions, short form).

### 1.1 `core/` — reuse with targeted modifications

| File | Action | Notes |
|---|---|---|
| `core/__init__.py`, `errors.py`, `logging.py`, `params.py`, `registry.py`, `discovery.py`, `specs.py`, `numeric.py`, `portfolio_math.py` | **Reuse unchanged** (apart from the rename) | |
| `core/columns.py` | **Modify** | Add quote / book / label column constants (§3.2). |
| `core/types.py` | **Modify** | `Frequency`: add `S1`, `S5`, `S10`, `S30`; replace the `minutes` property with `seconds: int \| None` and keep `minutes` as a derived property returning `None` when not a whole minute. Add `BAR_LABEL_KEY = "bar_label"` metadata constant with values `"start"` / `"end"` (§3.4). `MarketData` itself is unchanged. |
| `core/calendar.py` | **Modify** | `periods_per_year` must work from `frequency.seconds` (`sessions * ceil(session_minutes*60 / seconds)`). Add `session_bounds(day) -> (open_utc, close_utc)` using `exchange_calendars` opens/closes (handles early closes) and `session_ids(timestamps) -> IntArray` (session index per timestamp, computed from the calendar in local exchange time, never from `datetime64[D]` in UTC — a 16:00 New York close is 20:00 or 21:00 UTC, so UTC-date grouping is wrong only around DST but session grouping must be exact). |
| `core/run_config.py` | **Modify** | Remove `options_chain`, `splices`, `factors` fields and the `DataSpec.extra_datasets` field (fundamentals are deleted). Keep `experiment`, `seed`, `split`. `ExecutionSpec` unchanged. |
| `core/results.py` | **Modify** | Remove `AUX_FIELDS` option Greeks (set it to `()`), keep everything else. |
| `core/interfaces.py` | **Modify** | Remove `NEEDS_OPTIONS_CHAIN`, `NEEDS_FUNDAMENTALS`, `ASSET_OPTION`, `ASSET_FUTURE` constants. Add `FREQ_SUBMINUTE = "freq:subminute"`. Everything else (DataSource, Strategy, Overlay, CostModel, SlippageModel, FillModel, Metric, ChartBuilder, UniverseProvider, contexts) unchanged. |
| `core/options_math.py` | **Delete** | |

### 1.2 `data/` — reuse the cache/catalog/validation spine; add quote ingestion

| File | Action | Notes |
|---|---|---|
| `data/cache.py`, `data/catalog.py`, `data/normalize.py`, `data/adjustments.py`, `data/importer.py` | **Reuse unchanged** | `importer.py` gets one addition: §3.8 book-level mapping. |
| `data/service.py` | **Modify** | Remove nothing; add `DataService.fetch_partitioned(request, by="month")` helper that splits a long request into per-instrument, per-month `DataRequest`s and concatenates lazily (§3.9). |
| `data/validation.py` | **Modify** | Add intraday checks (§3.6). Generalize `check_calendar_gaps` to intraday (missing bars per session). |
| `data/sources/wrds_crsp.py` | **Modify, split** | Move TAQ code into a new `data/sources/wrds_taq.py`. Keep `crsp_daily` (for daily reference data / benchmarks) and `search_instruments`. Delete `compustat_annual`, `ff_factors`, `optionm_chain`, `crsp_treasury`, `crsp_index` handlers and their helpers. Keep the `sp500` universe query (`crsp.dsp500list`) — it is how the team picks names. |
| `data/sources/wrds_taq.py` | **Add** | `taq_trade_bars` (the existing `taq_bars`, renamed), `taq_nbbo_bars` (new, §3.3), `taq_nbbo_raw` (new, flagged heavy, §3.9). |
| `data/sources/local_file.py`, `data/sources/synthetic.py` | **Reuse unchanged** | `synthetic.py` gets sub-minute frequencies and `bid`/`ask`/`mid` fields so tests run offline (§3.10). |
| `data/sources/synthetic_markets.py` | **Modify** | Delete futures/options generators; keep the equity generator used by tests. |
| `data/sources/yahoo.py` | **Reuse unchanged** | Still useful for daily benchmarks. |
| `data/universes/index_membership.py`, `static.py` | **Reuse unchanged** | |
| `data/universes/liquidity.py`, `size.py` | **Reuse unchanged** | Harmless, small. |
| `data/futures.py`, `data/fundamentals.py` | **Delete** | |

### 1.3 `engine/` — reuse almost entirely

| File | Action | Notes |
|---|---|---|
| `engine/base.py`, `pipeline.py`, `simulation.py`, `returns.py`, `schedule.py`, `trades.py`, `lookahead.py`, `vectorized.py`, `event/*`, `fills/*`, `costs/commissions.py`, `costs/financing.py` | **Reuse unchanged** | The vectorized engine is the economic-test engine. The lookahead check is reused for feature pipelines (§4.5). |
| `engine/resample.py` | **Modify** | `_UNITS` gains `MIN1: ("m", 1)` and second-based entries so a 5 s grid can be resampled to 1 min / 5 min. |
| `engine/vectorized.py` | **Modify (one line)** | Remove `ROLL_FIELD` handling (futures). |
| `engine/costs/slippage.py` | **Modify** | Add `quoted_spread` model (§7.2): pays half of the *observed* `spread` field at the trade bar instead of a fixed bps. |

### 1.4 `strategies/`, `portfolio/` — prune to what the economic test needs

| File | Action |
|---|---|
| `strategies/buy_and_hold.py` | **Keep** (reference / sanity run). |
| `strategies/prediction_signal.py` | **Add** (§7.1). Reads an out-of-sample predictions Parquet and emits targets. |
| All other `strategies/*.py` | **Delete.** |
| `portfolio/construction/equal_weight.py`, `signal_proportional.py`, `quantile.py` | **Keep** (cross-sectional test). |
| `portfolio/construction/mean_variance.py`, `risk_based.py` | **Delete.** |
| `portfolio/overlays/sizing.py`, `limits.py` | **Keep** (gross/net caps, vol target). |
| `portfolio/overlays/drawdown.py`, `exits.py`, `hedging.py`, `options.py`, `regime.py` | **Delete.** |
| `templates/*.tmpl` (strategy, overlay, cost, data_source, metric, chart, constructor, test) and `/templates/example_*.py` | **Keep**; delete nothing. Add `model.py.tmpl` and `feature_set.py.tmpl` (§6.1). |

### 1.5 `analytics/` — reuse; delete options/factor pieces

| File | Action |
|---|---|
| `analytics/stats.py`, `downsample.py`, `chartkit.py`, `service.py`, `robustness/core.py` | **Reuse unchanged.** `stats.deflated_sharpe`, `probabilistic_sharpe`, `bootstrap_ci`, `stationary_bootstrap_indices` and `robustness.pbo_cscv`, `walk_forward_windows`, `breakeven` are called from `ml/`. |
| `analytics/metrics/{returns,risk,risk_adjusted,drawdown,statistical,trades,costs,portfolio,benchmark}.py` | **Reuse unchanged.** |
| `analytics/metrics/options.py`, `factor.py` | **Delete.** |
| `analytics/charts/{performance,drawdown,distribution,rolling,trades,costs,positions,overlays,comparison}.py` | **Reuse unchanged.** |
| `analytics/charts/options.py`, `factor.py` | **Delete.** |
| `analytics/options_stats.py` | **Delete.** |

Frontend impact: the chart registry (`frontend/src/charts/registry.ts`) is keyed by `ChartType`,
not by backend chart plugin names, so deleting backend charts requires **no** frontend change.

### 1.6 `services/`, `api/`, `cli.py`, `scaffold.py`

| File | Action | Notes |
|---|---|---|
| `services/container.py`, `jobs.py`, `job_handlers.py`, `plugins.py`, `run_store.py`, `run_service.py`, `results.py`, `result_io.py`, `report.py`, `report_charts.py`, `tearsheet.py` | **Reuse unchanged** | `RunStore.trials(experiment)` is the trial counter for deflated metrics (§7.4). |
| `services/runner.py` | **Modify** | Delete `_splice`, `splice_returns`, `_join_extra`, options-chain loading, factor fetching. Keep benchmark grafting. |
| `services/research.py` | **Modify** | Delete `regime_analysis` factor dependency if any; keep sweep, walk-forward, Monte Carlo, sensitivity, unlock. (Walk-forward here is bar-based and unpurged; it is **not** what the ML CV uses — see §5.) |
| `services/compatibility.py` | **Modify** | Drop options/fundamentals/futures capability checks; add `freq:subminute` capability check (a strategy without it cannot run on sub-minute data). |
| `api/*` | **Reuse unchanged** except `api/schemas.py` (drop deleted config fields) and regenerated OpenAPI. No new routers. |
| `cli.py` | **Modify** | Add the `hftf ml ...` command group (§6.4) and `hftf data pull-taq` (§3.9). |
| `scaffold.py` | **Modify** | Add `model` and `feature_set` scaffold kinds. |

### 1.7 `ml/` — **Add** (new package, §2 for placement, §4–§7 for contents)

```
hft_forecast/ml/
  __init__.py
  base.py          # Prediction schema, Forecaster + FeatureSet protocols, HorizonSpec
  config.py        # ExperimentConfig (pydantic) + hashing
  datasets.py      # bars -> per-instrument long frames; session ids; lazy loading
  labels.py        # make_labels(): forward log-return, direction classes, session guard
  features/
    __init__.py
    registry.py    # FeatureSet registry (same pattern as core.registry)
    basic.py       # STUB set: returns, spread, imbalance at level 1 — the only shipped set
    book.py        # STUB: level-N features; raises if book columns absent
  cv.py            # purged + embargoed walk-forward by session; uniqueness weights; n_eff
  models/
    __init__.py
    registry.py
    baselines.py   # zero, last-sign, logistic/ridge (sklearn)  — implemented, tiny
    torch_seq.py   # STUB: shared encoder + direction head + magnitude head
  train.py         # fold loop: fit, predict, write predictions parquet, manifest
  evaluate.py      # statistical metrics from predictions; bootstrap CIs
  economic.py      # predictions -> BacktestConfig(s) -> engine runs -> net metrics
  deflate.py       # trial counting, DSR, PBO from RunStore + predictions
  report.py        # one comparison table across horizons (CSV + Markdown)
  determinism.py   # seed_everything, environment manifest
  horizons/
    h30s/  config.yaml  README.md   # owned by the 30 s group
    h5m/   config.yaml  README.md   # owned by the 5 min group
    h30m/  config.yaml  README.md   # owned by the 30 min group
```

### 1.8 `tests/`

| Path | Action |
|---|---|
| `tests/conftest.py`, `helpers.py`, `fixtures/fake_wrds.py` | **Modify**: add fake NBBO table fixtures; drop options/futures/fundamentals fixtures. |
| `tests/golden/snapshots/*` | **Delete all but `buy_and_hold.json`**; regenerate it after the rename (it should be byte-identical apart from plugin version strings). |
| `tests/unit/core/test_options_math.py`, `tests/unit/data/test_futures.py`, `test_fundamentals.py`, `tests/integration/test_futures_intraday.py`, `test_options_flow.py`, `test_factor_benchmarks.py`, `tests/unit/test_factor_building_blocks.py` | **Delete.** |
| `tests/unit/data/test_wrds.py` | **Modify**: keep CRSP daily + sp500 cases; add `taq_nbbo_bars` case against the fake. |
| `tests/unit/ml/` | **Add**: `test_frequency_seconds.py`, `test_labels.py` (no session crossing, lookahead check), `test_cv.py` (purge/embargo invariants), `test_predictions_schema.py`, `test_models_register.py`, `test_report_table.py`. Small, deterministic, synthetic data only. |
| `tests/integration/test_ml_pipeline.py` | **Add**: synthetic 5 s data → labels → baseline → predictions → economic run → table. Must run in < 30 s. |
| `frontend/e2e/*` | **Reuse unchanged.** |

### 1.9 Top-level files

- `README.md`: new quick start (`make dev`, `cp .env.example .env`, `hftf data pull-taq`,
  `hftf ml run ml/horizons/h5m/config.yaml`, `make run` for the UI), the layer diagram with
  `ml`, the team split and the "frozen interface" list (§8.1).
- `configs/examples/`: delete the factor benchmarks `b1..b8`; keep `sma_synthetic.yaml` renamed
  to `buy_and_hold_synthetic.yaml`; add `econ_test_per_name.yaml` and `econ_test_cross_sectional.yaml`
  (§7.1).
- `pyproject.toml` dependencies: add `scikit-learn>=1.5`; add optional extra
  `ml = ["torch>=2.3"]`. Remove `yfinance` only if the Yahoo source is deleted (it is kept, so
  keep). Remove `empyrical-reloaded` from dev if unused after deletions (check).
- `.pre-commit-config.yaml`, `frontend/*`: unchanged apart from regenerated schema.

---

## 2. Where `ml` sits in the layer order and the import-linter changes

### 2.1 Layer order

```
hft_forecast.api
hft_forecast.services
hft_forecast.ml          <- new
hft_forecast.analytics
hft_forecast.engine
hft_forecast.data
hft_forecast.core
```

Rationale: `ml.economic` needs the engine and `ml.evaluate`/`ml.deflate` need
`analytics.stats` and `analytics.robustness`, so `ml` must be above both. `services` (jobs,
CLI orchestration) may call `ml`; `ml` must never import `services` or `api`.

### 2.2 `pyproject.toml` `[tool.importlinter]` changes

1. **Layered architecture**: insert `"hft_forecast.ml"` between `services` and `analytics`.
2. **Core depends on nothing**: add `"hft_forecast.ml"` to `forbidden_modules`.
3. **Plugins depend only on core**: add `"hft_forecast.ml.models"` and
   `"hft_forecast.ml.features"` to `source_modules` (they may not import `services`/`api`).
4. **New contract — "ML plugins depend only on core and ml.base"** (type `forbidden`):
   `source_modules = ["hft_forecast.ml.models", "hft_forecast.ml.features"]`,
   `forbidden_modules = ["hft_forecast.engine", "hft_forecast.analytics", "hft_forecast.data",
   "hft_forecast.ml.train", "hft_forecast.ml.evaluate", "hft_forecast.ml.economic",
   "hft_forecast.ml.cv", "hft_forecast.ml.labels"]`. A model gets arrays in, arrays out; it
   never touches data loading, CV or evaluation. This is what lets three groups add models
   without touching shared code.
5. **New contract — "Horizon packages are independent"** (type `independence`):
   `modules = ["hft_forecast.ml.horizons.h30s", "hft_forecast.ml.horizons.h5m",
   "hft_forecast.ml.horizons.h30m"]`. (These folders hold configs and README; if a group adds
   Python there, it must not import a sibling group's package.)
6. Keep "Strategies are independent of other plugins" and add `"hft_forecast.ml"` to its
   `forbidden_modules` (the `prediction_signal` strategy reads a Parquet file, not the ML
   package).

`torch` is imported lazily inside `ml/models/torch_seq.py` (same pattern as `wrds`/`yfinance`
in the data sources — `PLC0415` is already ignored in ruff). Add `torch.*` and `sklearn.*` to
the mypy `ignore_missing_imports` override.

---

## 3. Data design

### 3.1 Sub-minute frequency support

- `Frequency` gains `S1 = "1s"`, `S5 = "5s"`, `S10 = "10s"`, `S30 = "30s"`. Internal table
  `_INTRADAY_SECONDS` replaces `_INTRADAY_MINUTES`. `is_intraday` is true for all of them.
- `TradingCalendar.periods_per_year(frequency)` derives bars per session from seconds. For a
  5 s grid on XNYS: 390 min × 12 = 4 680 bars/session ≈ 1.18 M bars/year. The number is
  recorded in `MarketData.metadata["periods_per_year"]` by the runner as today.
- `engine/resample.py::period_codes` supports second units (`datetime64[s]` codes) and
  `MIN1`. The event engine's `Context.resampled("close", "1min")` therefore works on 5 s data
  with the existing "completed periods only" guarantee.
- `services/compatibility.py`: strategies and overlays need `freq:subminute` to run on
  `S*` frequencies; `buy_and_hold`, `prediction_signal`, the kept constructors and the kept
  overlays declare it. Anything else is rejected at config time with a clear message.
- The frontend `Frequency` dropdown comes from the OpenAPI enum; regenerate.

**Flag (performance):** `MarketData.panel()` pivots long → wide. 50 names × 126 sessions ×
4 680 bars = 29.5 M rows, 590 k unique timestamps; a single `(590k, 50)` float64 panel is
236 MB — fine. Do **not** call `panel()` on a request that spans more names/months than
that; the ML path (§3.9) works per instrument on long frames and never pivots.

### 3.2 New canonical columns (`core/columns.py`)

```
BID, ASK, BID_SIZE, ASK_SIZE      # level-1 NBBO at bar end
MID                                # (BID + ASK) / 2 at bar end
MICROPRICE                         # size-weighted mid at bar end
SPREAD                             # ASK - BID at bar end (currency)
SPREAD_BPS                         # SPREAD / MID * 1e4
MID_TWAP                           # time-weighted mean mid over the bar
N_QUOTES                           # NBBO updates inside the bar
N_TRADES, VWAP                     # from the trade table when joined
IS_STALE                           # 1.0 if no NBBO update inside the bar (mid carried forward)
QUOTE_FIELDS = (BID, ASK, BID_SIZE, ASK_SIZE, MID, MICROPRICE, SPREAD, SPREAD_BPS, MID_TWAP, N_QUOTES, IS_STALE)
```

Book (Level 2, §3.8): `book_columns(n_levels) -> tuple[str, ...]` returning
`bid_px_1..n, bid_sz_1..n, ask_px_1..n, ask_sz_1..n`. Level 1 of the book **is** `BID`/`ASK`
(the importer must assert they agree).

Engine compatibility: for quote-based datasets the source also writes `CLOSE = MID` and
`OPEN = MID` of the previous bar end (i.e. first mid of the bar) so the unchanged engine,
return computation and all existing metrics/charts work on quote data without special cases.
`HIGH`/`LOW` are the max/min mid inside the bar; `VOLUME` is trade volume if the trade table
is joined, else null. The `metadata["price_basis"] = "mid"` key records this.

### 3.3 TAQ NBBO ingestion (`data/sources/wrds_taq.py`)

Dataset `taq_nbbo_bars` (the default for every horizon group):

- Table: `taqm_{YYYY}.nbbom_{YYYYMMDD}` (one per session). **Flag:** the agent must verify
  the actual column names on WRDS (`wrds.Connection().describe_table("taqm_2024", "nbbom_20240102")`)
  before writing SQL. The expected columns are `sym_root, sym_suffix, time_m, best_bid,
  best_bidsiz, best_ask, best_asksiz, qu_cond` plus (newer years) `time_m_nano`. Column
  presence varies by year; the SQL builder takes the verified column list from a small
  per-year config dict rather than hard-coding it.
- Aggregation happens **server-side** (one SQL per session, same loop shape as the existing
  trade query): bucket = `floor(epoch(time_m) / seconds)`; per `(sym_root, bucket)` take the
  last quote by `time_m` (`DISTINCT ON` or `array_agg(... order by time_m desc)[1]`) for
  BID/ASK/sizes, `count(*)` for `N_QUOTES`, `max/min` of `(best_bid+best_ask)/2` for HIGH/LOW,
  and `MID_TWAP` as the duration-weighted mean (window function with `lead(time_m)`; if this
  proves too slow on WRDS, drop MID_TWAP to a plain mean and flag it).
- Quote filters are applied **in SQL before aggregation** (§3.5) so rejected quotes never
  reach the client.
- After the pull, a client-side step forward-fills missing buckets **within a session only**
  and sets `IS_STALE = 1.0` on those rows. Buckets before the first valid quote of the session
  stay null.
- Timestamps: `time_m` is the SIP timestamp in `America/New_York`; localize with that zone
  and convert to UTC, as the existing `taq_to_canonical` does. Bar-end labelling per §3.4.
- Session bounds come from `TradingCalendar.session_bounds(day)`, not the hard-coded
  `09:30`/`16:00`, so early-close sessions (day after Thanksgiving, Christmas Eve) are cut at
  13:00 instead of returning an empty afternoon.

Dataset `taq_trade_bars`: the existing query, renamed, bar-end labelled, with `VWAP` and
`N_TRADES` added. Used for `VOLUME`/`VWAP` features via a left join on `(timestamp, instrument)`
in `ml/datasets.py` when the experiment config lists both datasets.

Dataset `taq_nbbo_raw`: filtered quotes, no aggregation, one Parquet file per `(symbol, session)`
under `data/raw/taq_nbbo/`. `cacheable = False`. **Flag:** a liquid S&P name produces 1–10 M
NBBO updates per session; raw pulls are for a handful of names and days only (event-time
experiments by the 30 s group). The CLI refuses a raw request for more than 5 symbols × 20
sessions without `--force`.

### 3.4 Timestamp and bar-end conventions

- **All bars are right-closed and labelled at the bar end**: the row with `timestamp = t`
  summarizes `(t − Δ, t]`. The level-1 fields are the last valid NBBO at or before `t`.
  "Information available at `t`" means this row and earlier rows. This is the convention the
  engine already assumes for `close` (decision at the close of bar `t`, trade at `t + lag`).
- The inherited TAQ trade-bar code labels buckets at the bucket **start**; change it to the
  end (`+ seconds`). Record `metadata["bar_label"] = "end"` on every intraday dataset and make
  `ml/datasets.py` refuse data without it.
- First bar of a session ends at `open + Δ` (09:30:05 for 5 s); last bar ends at the session
  close (16:00:00). Bars touching the opening cross (`open`, `open + warmup_seconds`) and the
  closing cross (`close − cooldown_seconds`, `close`) are kept in the data but **masked for
  labels and training** by default (`warmup_seconds = 60`, `cooldown_seconds = 60`, both in
  `ExperimentConfig`).
- Storage is UTC (`Datetime("us", "UTC")`) as today. Session identity is computed from the
  exchange calendar in local time (`core/calendar.py::session_ids`), stored as an integer
  column `session_id` by `ml/datasets.py`, and is the only thing labels/CV use to decide
  "same session".
- **Flag (ambiguity):** TAQ `time_m` (SIP) vs `part_time` (participant exchange time). Use
  `time_m` everywhere; mixing them misaligns quotes and trades by microseconds to
  milliseconds. Document this and do not expose `part_time`.
- **Flag:** DST. `exchange_calendars` handles it; the unit test must include a session in
  March and one in November.

### 3.5 Quote filters (applied in SQL, parameters in `WrdsTaqParams`)

Defaults follow the standard Holden–Jacobsen style cleaning; every threshold is a parameter:

1. `sym_suffix is null` (common shares only; drops preferred/warrants).
2. `best_bid > 0 and best_ask > 0 and best_bidsiz > 0 and best_asksiz > 0`.
3. `best_ask > best_bid` (drop locked and crossed; count them for the quality report).
4. `(best_ask − best_bid) / mid <= max_spread_frac` (default 0.05) **and**
   `best_ask − best_bid <= max_spread_dollars` (default 5.0).
5. `qu_cond` in the normal set (default `{'R'}`; **flag:** verify the code list for the
   chosen years on WRDS — some years use different condition vocabularies).
6. Regular session only: `time_m` within `session_bounds(day)`.
7. Optional (off by default): drop quotes whose mid moves more than `max_jump_bps` from the
   previous valid mid and reverts within `k` quotes (bounce filter). Off because it is a
   modelling choice the groups should test, not a data-cleaning default.

The number of quotes removed by each filter is returned alongside the bars and stored in the
dataset's `QualityReport` (§3.6) so the report can show "N % of quotes dropped by filter X".

### 3.6 Intraday quality checks (`data/validation.py`)

Existing checks stay (duplicates, monotonic, non-positive prices, extreme jumps, calendar gaps
for daily). New checks run when `frequency.is_intraday`:

| Check | Severity | What |
|---|---|---|
| `session_coverage` | warning if < 98 % | bars present / bars expected per session (from calendar bounds and Δ), per instrument |
| `missing_sessions` | warning | calendar sessions with zero rows |
| `stale_run` | info / warning if > 5 % of bars | share of `IS_STALE == 1` and longest stale run (bars) per session; a run longer than `halt_seconds` (default 300) is reported separately as a probable halt |
| `locked_crossed` | info | quotes dropped by filter 3 (from the source's filter counts) |
| `spread_outliers` | warning | bars with `SPREAD_BPS` above the 99.9th percentile of the instrument |
| `bar_label` | **error** | metadata `bar_label != "end"` |
| `session_bounds` | **error** | any bar with timestamp outside `[open + Δ, close]` of its session |
| `mid_consistency` | **error** | `MID != (BID + ASK)/2` beyond 1e-9, or `CLOSE != MID` when `price_basis == "mid"` |
| `early_close` | info | sessions whose close is not 16:00 local, listed |

Reports are stored by the existing `Catalog` and shown on the existing Data page unchanged.

### 3.7 Mid-price definition

`MID = (BID + ASK)/2` of the last valid NBBO at bar end. `MICROPRICE =
(BID·ASK_SIZE + ASK·BID_SIZE)/(BID_SIZE + ASK_SIZE)`. Labels use `MID` only (§4); microprice
is a feature. Both are in currency; returns are **log** returns of `MID`.

### 3.8 Level 2 placeholder

- `importer.ImportMapping` gains `book_levels: int = 0` and `book_columns: dict[str, str]`
  (canonical `bid_px_1` → source column). `apply_mapping` writes the wide book columns and
  derives `BID`/`ASK`/sizes/`MID`/`SPREAD` from level 1 when they are not mapped explicitly.
- `local_file` source already exposes extra columns as fields, so a book dataset is just an
  import with `book_levels > 0`; `DatasetInfo.fields` lists them.
- `ml/features/book.py` is a stub that raises `DataError("book columns not present")` when
  `book_levels == 0` and otherwise exposes level-N imbalance and depth features (signatures
  only).
- **Flag (unknown):** the employer's format. LOBSTER-style message/orderbook pairs need a
  pre-processing script (message replay) that is out of scope until the format is known.
  Record this as an open item in `ml/horizons/*/README.md`.

### 3.9 Memory strategy for quote data

Sizing (50 names, 126 sessions):

| Grid | Rows | Parquet (zstd, ~12 float cols) | In-memory polars float64 |
|---|---|---|---|
| 5 s | 29.5 M | ~1.5–3 GB | ~3 GB |
| 1 min | 2.5 M | ~150 MB | ~250 MB |
| raw NBBO | 10⁸–10⁹ | tens of GB | infeasible |

Rules:

1. **Aggregate on the server.** Only bars cross the wire (§3.3). Raw NBBO is opt-in and capped.
2. **One cache entry per (instrument, month).** `DataService.fetch_partitioned` splits the
   request so each Parquet file is ~100 k (1 min) to ~500 k (5 s) rows; the existing
   versioned cache and catalog are unchanged. A `data/manifests/<name>.yaml` written by
   `hftf data pull-taq` lists the entries that make up a named dataset (e.g. `sp50_2024h1_5s`).
3. **ML loads lazily per instrument.** `ml/datasets.py::load_bars(manifest, instruments,
   sessions) -> pl.LazyFrame` uses `pl.scan_parquet` over the manifest's files, filters, and
   collects **one instrument at a time** for feature computation. Features and labels are
   written back to `data/features/<experiment_hash>/<instrument>.parquet` so training never
   recomputes them. Training reads with `pl.scan_parquet` and converts to `float32` NumPy
   only for the fold's rows.
4. **Never pivot the full 5 s panel for ML.** `MarketData.panel()` is used only by the
   economic test, which runs one instrument (per-name) or a 1 min grid (cross-sectional).
5. **The economic test for the 30 s group runs per instrument** (one `BacktestConfig` per
   name, §7.1); 50 runs × 590 k bars is fine for the vectorized engine.
6. Cap: `ExperimentConfig.max_rows_in_memory` (default 20 M) — exceeding it in a single
   `collect()` raises with a message to narrow `instruments`/`sessions`.
7. `polars` thread count and `torch.set_num_threads` are set from `HFTF_MAX_WORKERS` so two
   people on one machine do not oversubscribe.

**Flag (data access):** WRDS cloud SQL per session for 50 names: the NBBO table is
hundreds of millions of rows per day across all symbols; a well-indexed `sym_root in (...)`
query with server-side aggregation takes seconds to a couple of minutes per session. 126
sessions × 2 datasets ≈ 2–6 hours of pulling. Plan it as a background job on one machine,
not interactive. If WRDS throttles, fall back to the PostgreSQL `wrds-pgdata` direct
connection or WRDS Cloud (SAS/Python on their servers) — out of scope here, flag only.

### 3.10 Synthetic quote data for tests

`data/sources/synthetic.py` gets a `quotes` dataset: a random-walk mid, spread drawn from a
log-normal, sizes Poisson, sub-minute frequencies, bar-end labelled, with `IS_STALE` injected
at a configurable rate and one injected "halt" per instrument. Every `ml` unit test and the
integration test use it. Keep it under 150 lines.

---

## 4. Label and feature design

### 4.1 Horizon specification (`ml/base.py::HorizonSpec`)

```
HorizonSpec(name="h30s", seconds=30, grid=Frequency.S5)   -> steps = 6
HorizonSpec(name="h5m",  seconds=300, grid=Frequency.MIN1) -> steps = 5
HorizonSpec(name="h30m", seconds=1800, grid=Frequency.MIN1) -> steps = 30
```
`steps = seconds // grid.seconds`; the validator rejects non-integer ratios. The three specs
are constants in `ml/base.py`; a group may add a different `grid` for its own horizon (e.g.
30 s on a 1 s grid) by editing only its `horizons/<h>/config.yaml`.

### 4.2 Labels (`ml/labels.py::make_labels(frame, spec, params) -> pl.DataFrame`)

Per instrument, on the bar frame sorted by `timestamp`:

- `fwd_logret = log(MID[t + steps]) − log(MID[t])` computed with a **shift within
  `session_id`** (`.shift(-steps).over("session_id")`), so any label whose target bar is in
  a later session is null. This is the session-crossing guard; there is no other mechanism.
- Additional null rules: `MID[t]` or `MID[t + steps]` null; `IS_STALE[t + steps] == 1` when
  `params.drop_stale_targets` (default true — a stale target bar means the "future mid" is
  just the current one); `t` inside the warmup/cooldown windows (§3.4); the interval
  `(t, t + steps]` contains a gap longer than `halt_seconds`.
- `direction`: three classes `{−1, 0, +1}` with a dead zone `|fwd_logret| <= tau`.
  `tau` is expressed in **bps** (`LabelParams.dead_zone_bps`, default per horizon in the
  group config; suggested starting points 1 bps / 3 bps / 10 bps). A `tau = 0` config gives
  the binary problem; the sign of exactly zero is then class 0 and such rows are dropped.
  **Flag:** the right `tau` is roughly the half-spread for 30 s; this is a research choice,
  not a template default — the template only makes it a parameter and reports the class
  balance.
- `magnitude = |fwd_logret|` (bps) and the signed `fwd_logret` are both stored; the model
  interface (§6.1) predicts the signed return and the direction probabilities, and the
  evaluation derives magnitude metrics from both.
- Output columns: `timestamp, instrument_id, session_id, fwd_logret, direction, magnitude_bps,
  label_valid` (bool) — never drop rows here, because features need the full grid; CV drops
  invalid rows later.
- `label_end_timestamp = timestamp + steps·Δ` is stored explicitly; the CV purge uses it.

### 4.3 Features (`ml/features/`)

- `FeatureSet` protocol: `name`, `lookback_bars: int`, `requires: tuple[str, ...]` (column
  names), `compute(frame: pl.DataFrame) -> pl.DataFrame` returning the same row count with
  added `f_*` columns. Registered through `ml/features/registry.py` (same `register`
  decorator pattern as `core.registry`, separate registry instance).
- The template ships exactly **one** set, `basic` (stub with signatures and a handful of
  trivially causal expressions so the pipeline runs end-to-end): log mid returns at 1, 5, 10,
  `steps` bars; `SPREAD_BPS`; level-1 imbalance `(BID_SIZE − ASK_SIZE)/(BID_SIZE + ASK_SIZE)`;
  `MICROPRICE − MID` in bps; `N_QUOTES` log; rolling realized vol over 20 and 100 bars;
  time-of-day as seconds since open; `IS_STALE`. Nothing else.
- Causality rules enforced by construction: every expression uses `.shift(k >= 0)` or
  `.rolling_*(window)` **over `session_id`** (rolling windows do not cross sessions, which
  also makes the first `lookback_bars` of each session null rather than contaminated by the
  previous close). No `.shift(negative)` anywhere in `features/`; a ruff-style grep test
  (`tests/unit/ml/test_no_negative_shift.py`) asserts it.
- Normalisation is **not** a feature-set concern: the `Forecaster` (or a `StandardScaler`
  inside the baseline pipelines) is fitted on the training fold only. The template's
  `train.py` passes raw features; scaling lives inside the model.
- Coarser-grid features for a fine model (e.g. 1 min realized vol in a 5 s model) must use
  only completed coarser bars: implement with `engine/resample.period_codes` and a
  `group_by_dynamic(..., closed="right", label="right")` followed by `.shift(1)` over
  `session_id`. The template includes one example (`f_vol_1m_completed`) in `basic`.

### 4.4 Alignment of features and labels

`ml/datasets.py::build_dataset(config) -> Path` writes, per instrument,
`[keys] + [session_id] + [f_*] + [fwd_logret, direction, magnitude_bps, label_valid,
label_end_timestamp]` to `data/features/<hash>/`. The hash covers the manifest, `HorizonSpec`,
`LabelParams`, the feature-set names and their params — reproducible by construction.

### 4.5 Lookahead test for feature pipelines

Reuse the idea of `engine/lookahead.py`: `ml/features/registry.py::check_feature_lookahead(
feature_set, frame, n_cuts=5)` computes features on the full frame and on frames truncated
at several cut timestamps and asserts equality up to each cut. Every registered feature set
gets this test automatically via a parametrized pytest (`tests/unit/ml/test_features_causal.py`)
on the synthetic quotes dataset. A group adding a feature set gets the test for free.

---

## 5. Purged / embargoed walk-forward CV and effective sample size

### 5.1 `ml/cv.py::CVSpec`

```
CVSpec(
  scheme = "walk_forward",          # the only scheme in the template
  train_sessions = 40,
  test_sessions  = 10,
  anchored = False,
  embargo_bars = steps,              # default = label horizon in bars
  purge = True,
  min_train_rows = 100_000,
)
```

Windows are built **in session units** with `analytics.robustness.walk_forward_windows`
applied to the sorted unique `session_id`s (not to bars), so a test window always starts at
a session open. **Flag (sample size):** 126 sessions give at most ~8 test windows of 10
sessions; 60 sessions give ~2. With 3 months of data the 30 min group has ~2 folds and
should expect wide confidence intervals; the report must show per-fold results, not just the
pooled number.

### 5.2 Purging and embargo (`ml/cv.py::split(dataset, spec) -> Iterator[Fold]`)

For each window with test rows `T = [t_start, t_end]` (timestamps):

- **Purge**: remove from the training set every row whose label interval
  `[timestamp, label_end_timestamp]` overlaps `[t_start, t_end]`. Because labels never cross
  sessions and test windows start at a session open, the purge only bites at the window's
  trailing edge; it is still implemented generically (interval overlap on timestamps) so a
  group can switch to finer windows later.
- **Embargo**: additionally drop training rows with `timestamp` in
  `(t_end, t_end + embargo_bars·Δ]` (relevant for anchored / non-contiguous schemes; a no-op
  for contiguous forward windows, kept for generality and tested).
- The function returns `Fold(index, train_mask, test_mask, train_sessions, test_sessions)`
  as boolean masks over the dataset rows; it never copies data.
- Invariants tested: no training row's label interval intersects any test row's interval;
  test windows are disjoint and ordered; every test row belongs to exactly one fold.

### 5.3 Overlapping labels: uniqueness weights and effective sample size

- `ml/cv.py::label_uniqueness(timestamps, label_end_timestamps) -> FloatArray`: for each
  row, the average over its label interval of `1 / (number of concurrent labels)`
  (López de Prado, *Advances in Financial ML*, ch. 4). Computed per instrument with a
  difference-array sweep in O(n); concurrency is `steps` almost everywhere, so weights are
  ≈ `1/steps` except at session edges.
- `n_eff = (Σw)² / Σw²` per fold and pooled; for a pure overlap of `steps` this is ≈
  `n / steps`. The report (§7.5) shows `n_rows` and `n_eff` side by side.
- Sample weights are passed to `Forecaster.fit(..., sample_weight=w)`; baselines use them,
  the torch stub exposes the argument. **Flag:** autocorrelation of mid returns beyond the
  label overlap (microstructure noise, volatility clustering) makes the true `n_eff` lower
  than `n/steps`; the template reports the overlap-based number and labels it as an upper
  bound. Block-bootstrap CIs (§7.3) are the honest alternative and are the primary
  uncertainty statement.
- A **non-overlapping subsample** option (`CVSpec.stride = steps`) keeps every `steps`-th
  row per session for a sanity run with independent labels; the report shows both when it
  is enabled.

---

## 6. Model set, training and determinism

### 6.1 Interfaces (`ml/base.py`)

```
class Prediction:                     # one row per (timestamp, instrument, horizon)
    p_down, p_flat, p_up: float       # sums to 1; p_flat = 0 for binary setups
    y_hat_logret: float               # signed expected forward log return
    fold: int
    model: str                        # registry name + version
    # written to data/runs/ml/<experiment>/<run_id>/predictions.parquet together with
    # y_true columns copied from the dataset so evaluation needs no join

class Forecaster(Protocol):
    name: ClassVar[str]; version: ClassVar[str]
    Params: ClassVar[type[PluginParams]]
    def fit(self, X: FloatArray, y_dir: IntArray, y_ret: FloatArray,
            sample_weight: FloatArray | None, seed: int) -> None
    def predict(self, X: FloatArray) -> tuple[FloatArray, FloatArray]   # (probas (n,3), y_hat (n,))
    def state_dict(self) -> dict[str, Any]       # for saving; baselines pickle, torch saves tensors
```
Framework-agnostic on purpose: NumPy in, NumPy out. A sklearn, LightGBM, JAX or torch model
all fit behind it. Models register with `@register_model(name=..., version=...)` in
`ml/models/registry.py` (separate registry instance, same mechanics as `core.registry`).

Scaffolds: `templates/model.py.tmpl` and `templates/feature_set.py.tmpl`; `hftf new model
my_model` and `hftf new feature_set my_set` drop them into `user_plugins/` (discovered like
every other user plugin). Groups should add models under `hft_forecast/ml/models/` via PR,
but `user_plugins/` lets them iterate without touching shared code.

### 6.2 Model set shipped by the template

| Name | Status | Purpose |
|---|---|---|
| `zero` | implemented (10 lines) | predicts class priors from training fold and `y_hat = 0`; the floor every metric is compared to |
| `last_sign` | implemented | direction = sign of the last `steps`-bar return (momentum), `y_hat` = that return; the naive persistence baseline |
| `logit_ridge` | implemented (sklearn `Pipeline(StandardScaler, LogisticRegression)` + `Ridge`) | the linear baseline; the first model every group must beat; supports `sample_weight` |
| `gbm` | **stub** (signature only, optional dependency `lightgbm`) | the usual strong tabular baseline; left for the groups |
| `torch_seq` | **stub** | shared encoder over a `(n, lookback, n_features)` window + two heads: 3-class direction (cross-entropy) and signed return (Huber); loss = `CE + lambda * Huber`. Only `__init__`, `fit`, `predict` signatures, the `nn.Module` skeleton with the two heads and the window-batching helper are written. No architecture choice (GRU/TCN/transformer) is made; `encoder: str` is a parameter the groups fill in. |

Sequence models need windows: `ml/datasets.py::windowed(X, lookback, session_id)` builds the
`(n, lookback, f)` view with a stride-trick **per session** so windows never span a session
boundary; rows with fewer than `lookback` bars since the session open are masked.

### 6.3 Training loop (`ml/train.py::run_experiment(config) -> RunManifest`)

1. `seed_everything(config.seed)` (§6.4), write `environment.json`.
2. `build_dataset` (cached by hash).
3. For each fold from `cv.split`: fit on train rows (weights from §5.3), predict test rows,
   append predictions. Baselines refit per fold; torch models refit per fold from scratch
   (no warm start) so folds are independent.
4. Write `predictions.parquet`, `metrics.json` (§7.3), `manifest.json` (config, hashes,
   fold boundaries, timings, environment, git commit via the existing `services.runner.git_commit`).
5. Register the run in `RunStore` with `experiment = config.experiment` and `tags =
   ["ml", horizon.name, model.name]` so the UI lists it and `RunStore.trials` counts it.
   The run's `BacktestResult` slot is filled by the economic test (§7.1) when it is run; until
   then the record holds metrics only. **Flag:** `RunStore.complete` currently expects a
   `BacktestResult`; add a `complete_ml(run_id, metrics, manifest_path)` method that stores
   metrics without a result rather than fabricating an empty result.

Hyperparameter search is **not** in the template. Groups use `services.research.run_sweep`
patterns or plain loops in their own scripts; every evaluated config must be a separate
`run_experiment` call so that trial counting (§7.4) is honest.

### 6.4 Determinism

`ml/determinism.py::seed_everything(seed)`: `random`, `numpy` (`default_rng` is passed
explicitly everywhere; the global seed is set only for third-party code), `torch.manual_seed`,
`torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(HFTF_MAX_WORKERS)`,
`PYTHONHASHSEED` check (warn if unset), polars: all joins/group-bys followed by an explicit
`.sort(keys)` (polars is otherwise order-unstable across thread counts). sklearn models get
`random_state=seed`. Recorded in `environment.json`: Python, polars, numpy, torch, sklearn
versions, CPU count, thread settings, `pip freeze` hash.

**Flag (CPU now, GPU later):** when a GPU appears, add `CUBLAS_WORKSPACE_CONFIG=:4096:8`
and `torch.backends.cudnn.deterministic = True` in the same function; bit-exact
reproducibility across CPU and GPU is **not** achievable and the report must state that
results are reproducible on the hardware class recorded in `environment.json`.

Determinism test: `tests/unit/ml/test_determinism.py` runs `logit_ridge` twice on synthetic
data and asserts identical predictions; the torch stub is covered once a group implements it
(test is parametrized over registered models and skips stubs).

### 6.5 CLI (`cli.py`, group `hftf ml`)

- `hftf ml build-dataset <config.yaml>`
- `hftf ml run <config.yaml> [--model name] [--seed N]`
- `hftf ml econ <run_id> [--cross-sectional]` (§7.1)
- `hftf ml compare --experiment h30s --experiment h5m --experiment h30m --out report/`
- `hftf data pull-taq --manifest sp50_2024h1 --symbols-file symbols.txt --start --end --grid 5s`
  (runs `fetch_partitioned` for `taq_nbbo_bars` and `taq_trade_bars`, writes the manifest,
  prints the quality summary)

Everything else stays as today (`hftf run`, `hftf data pull/import/catalog`, `hftf serve`,
`hftf check-lookahead`, `hftf new ...`).

---

## 7. Evaluation

### 7.1 Economic test via the existing engine (`ml/economic.py`)

**Per-name (primary).** For each instrument, build a `BacktestConfig`:

- `data`: the manifest's `local_file`/cache entry for that instrument on the horizon's grid
  (`price_basis = mid`).
- `strategy`: `prediction_signal` with params `predictions_path`, `threshold` (trade when
  `max(p_up, p_down) − p_flat > threshold`, or when `|y_hat| > k · SPREAD_BPS` — both modes,
  `mode` parameter), `hold_bars = steps`, `overlap = "replace" | "average"` (a new signal
  replaces the position, or positions are averaged over the last `steps` signals, the
  standard way to use overlapping forecasts without a lookahead).
- `execution`: `vectorized`, `lag_bars = 1` (decision at bar end `t`, fill at bar end
  `t+1`), `price = close` (= mid), `rebalance = on_change`.
- `costs`/`slippage`: three variants per run — `none`, `quoted_spread` (half the observed
  spread at the trade bar; new model in §1.3), `quoted_spread + sqrt_impact`. The run tag
  records the variant.
- `benchmark`: none (or the instrument itself buy-and-hold for the UI).

Outputs per instrument: net/gross Sharpe (per bar, annualized with the grid's
`periods_per_year` — **flag:** annualized Sharpe on a 5 s grid is dominated by cost drag and
should be read with the per-bar mean/t-stat shown next to it), turnover, cost as a fraction
of gross PnL, `breakeven` cost multiple via `services.research.run_sensitivity`-style scaling
(`cost_scale ∈ {0, 0.5, 1, 2, 4}`, reusing `VectorizedEngine.cost_scale`), hit rate of
executed trades. Pooled: equal-weight average of per-instrument net returns (a portfolio of
the 50 per-name books) and the distribution across instruments.

**Cross-sectional (secondary).** One `BacktestConfig` on the 1 min grid (the 30 s group
aggregates 5 s predictions to 1 min first): `prediction_signal` with `output_kind = signals`
emitting `y_hat` per instrument, constructor `quantile` (long top decile, short bottom)
or `signal_proportional`, overlays `gross_cap` + `vol_target` from `portfolio/overlays/sizing.py`,
same cost variants. **Flag:** cross-sectional at 30 s with 50 names rebalanced every bar is
economically absurd under marketable-order costs; run it, report it, and let the result say so.

`prediction_signal` reads only out-of-sample rows (`fold >= 0`); the strategy asserts that
every timestamp it emits a target for has a prediction from a fold whose test window
contains it, and refuses files with duplicated keys. It declares `freq:subminute`,
`engine:vectorized`, `supports:short`.

Fill realism: fills at mid plus half the quoted spread assume a marketable order that fills at
the touch for a size below displayed depth. The event engine's limit orders exist but there
is **no queue-position model**, so passive fills are out of scope; the report states this.

### 7.2 New slippage model `quoted_spread` (`engine/costs/slippage.py`)

`vectorized(ctx)`: needs the `SPREAD_BPS` panel; `CostContext` does not carry arbitrary
fields, so add `extra_panels: dict[str, FloatArray]` to `CostContext` (populated by the
vectorized engine from `data` for the fields a cost model declares in a new
`CostModel.requires: ClassVar[tuple[str, ...]]`). Cost = `|trade weight| × 0.5 × SPREAD_BPS[t] × 1e-4`,
with a `multiplier` param (default 1.0) and a floor `min_bps` (default 0). The event-engine
`price_adjustment` uses the bar's spread via a new `BarSnapshot.spread` attribute (nullable).
This is the only engine interface change in the whole design.

### 7.3 Statistical metrics (`ml/evaluate.py`)

Computed per fold and pooled, from `predictions.parquet`, always alongside the same metrics
for `zero` and `last_sign` on identical rows:

- Direction: accuracy, balanced accuracy, macro F1, one-vs-rest AUC for up and down,
  log-loss, Brier score, a 10-bin reliability table (calibration), confusion matrix.
- Signed return: MAE, RMSE, `R²` vs the zero predictor, Spearman IC between `y_hat` and
  `fwd_logret`, IC t-stat computed on per-session ICs (Fama–MacBeth style, which respects
  the session structure), and the IC of `|y_hat|` vs `magnitude_bps`.
- Conditioned: the same metrics on the top-decile `|y_hat|` rows (where trades would happen).
- Uncertainty: block bootstrap by session (resample sessions with replacement, reuse
  `analytics.stats.stationary_bootstrap_indices` on the session index) → 95 % CI for
  accuracy, AUC, IC, and for net Sharpe in §7.1. 1 000 resamples, seeded.
- Sample sizes: `n_rows`, `n_eff` (§5.3), class balance, share of rows dropped by each label
  rule.

### 7.4 Deflated metrics with trial counting (`ml/deflate.py`)

- Every `run_experiment` is one trial of `config.experiment` (one experiment per horizon:
  `h30s`, `h5m`, `h30m`). `n_trials = RunStore.trials(experiment)` — the count of **all**
  runs ever registered under that experiment, including failed ideas, which is the point.
  The UI's existing experiment field shows it. Groups are told in the README not to delete
  runs.
- Deflated Sharpe of the economic test: `analytics.stats.deflated_sharpe(net_returns,
  trial_sharpes, n_trials)` with `trial_sharpes` = per-bar Sharpe of every trial's economic
  run under the `quoted_spread` cost variant.
- PBO: `analytics.robustness.pbo_cscv` on the `(T, n_trials)` matrix of per-bar net returns
  of all trials (aligned on the pooled test timestamps). Reported when `n_trials >= 4`.
- Deflated classification metric: a permutation-style baseline is **not** added (overdoing);
  instead the report shows accuracy against the bootstrap CI of the `zero` baseline and the
  Bonferroni-adjusted p-value of the IC t-stat for `n_trials` comparisons.

### 7.5 The single comparison table (`ml/report.py`)

`hftf ml compare` produces `report/comparison.csv` and `report/comparison.md`, one row per
`(model, cost_variant)` and one column group per horizon:

| | h30s (5 s grid) | h5m (1 min) | h30m (1 min) |
|---|---|---|---|
| rows / n_eff / sessions / folds | | | |
| accuracy [CI] vs zero / last_sign | | | |
| AUC up / AUC down | | | |
| IC [CI], IC t-stat | | | |
| MAE (bps), R² vs zero | | | |
| net Sharpe per-name pooled [CI] at `none` / `quoted_spread` / `+impact` | | | |
| breakeven cost multiple | | | |
| turnover per session | | | |
| DSR, PBO, n_trials | | | |
| cross-sectional net Sharpe at `quoted_spread` | | | |

The best model per horizon is chosen by **out-of-sample net Sharpe at `quoted_spread`** and
bolded; the table also always includes the two baselines. The markdown table is the one that
goes into the report and the presentation; nothing else is generated (no PDF, no slides).

---

## 8. Shared core vs per-horizon groups, milestones, risks

### 8.1 The frozen interface (shared core; changes need a PR reviewed by one person from each group)

1. Canonical bar schema and column names (§3.2) and the `bar_label = end` convention (§3.4).
2. `hftf data pull-taq` + manifests (§3.9) — the only way data enters the repo.
3. `HorizonSpec`, `LabelParams`, `make_labels` (§4.1–4.2).
4. `FeatureSet` protocol + registry + the causality test (§4.3, §4.5).
5. `CVSpec`, `split`, `label_uniqueness` (§5).
6. `Forecaster` protocol, `Prediction` schema, `predictions.parquet` layout (§6.1).
7. `run_experiment` outputs (`manifest.json`, `metrics.json`) and the `RunStore` registration (§6.3).
8. `ml/economic.py`, `prediction_signal`, `quoted_spread` (§7.1–7.2).
9. `ml/evaluate.py` metric names and `ml/report.py` table columns (§7.3, §7.5).

Everything a group owns lives in `ml/horizons/<h>/` (configs, README, notebooks), in
`ml/models/<their model>.py`, `ml/features/<their set>.py`, and `user_plugins/`. The
import-linter contracts in §2.2 make it impossible for a model or feature set to reach into
shared training/evaluation code, which is what keeps three parallel branches mergeable.

Ownership: the shared core is built by Arnav with the agent (milestones M0–M2). After the
freeze, each group nominates one "core contact" who reviews PRs touching the frozen list.
The 30 s group (3 people) also owns `taq_nbbo_raw` and anything event-time.

### 8.2 Milestones (final deliverable mid-December; weeks counted from 2026-10-20)

| Week | Milestone | Done when |
|---|---|---|
| 1 | **M0 Fork & prune**: rename, deletions, `make check` green, UI builds, CI (GitHub Actions running `make check` on push) | `buy_and_hold_synthetic.yaml` runs via CLI and shows in the UI |
| 1–2 | **M1 Data**: sub-minute `Frequency`, `wrds_taq` with NBBO bars + filters, intraday quality checks, synthetic quotes, `pull-taq` + manifest | 5 names × 1 month (5 s and 1 min) pulled; quality reports in the UI; DST and early-close sessions verified |
| 2–3 | **M2 Pipeline & freeze**: labels, `basic` features, CV, baselines, `run_experiment`, `evaluate`, `economic` (per-name), `report` | integration test green; `compare` table with `zero`/`last_sign`/`logit_ridge` for all three horizons on the M1 data; **interface frozen**; groups onboarded |
| 3 | **M1b Full pull** (background, one machine): 20–50 names × 3–6 months | manifest `sp50_<period>` complete; disk and pull time recorded |
| 3–5 | **M3 Models** (groups in parallel): `gbm`, `torch_seq` encoders, group feature sets | each group beats `logit_ridge` OOS on at least one statistical metric, or documents that it cannot |
| 5–6 | **M4 Economics & deflation**: cost variants, cross-sectional runs, DSR/PBO, breakeven | final `comparison.md`; per-instrument distributions |
| 6–7 | **M5 Robustness**: sensitivity to quote filters (§3.5 thresholds), to `tau`, to `warmup/cooldown`, to timestamp shift (+1 bar alignment error), to dropping stale bars; writes §"does it survive frictions" | one table per robustness axis |
| 7–7.5 | **M6 Report & presentation** | reproducible from `manifest` + `hftf ml run` + `hftf ml compare` on a clean clone |

### 8.3 Main risks (and what the template does about them)

| Risk | Likelihood | Mitigation in the template | Residual |
|---|---|---|---|
| **Data access**: NBBO column vocabulary differs by year; WRDS throttles multi-hour pulls; subscription scope narrower than expected | medium | per-year column config, per-(symbol, month) cache so partial pulls are usable, `pull-taq` resumable from the manifest, fake WRDS fixtures so everything else develops offline | if NBBO is unavailable, mid from trades (`taq_trade_bars` with `price_basis = trade`) is a documented fallback with a quality flag; results then measure trade-price, not quote, predictability |
| **Sample size / regimes**: 3–6 months is one regime; 30 min horizon has ~2–8 folds | high for 30 min | session-level folds, per-fold reporting, block-bootstrap CIs, `n_eff` | the report must present the 30 min result as under-powered; adding history is the only fix |
| **30 s cost drag**: typical large-cap spread 1–3 bps ≈ the 30 s mid move; marketable execution nets negative | high | three cost variants, breakeven multiple, trade-selectivity threshold tied to observed spread, conditioned metrics on top-decile rows | a negative net result at 30 s is a legitimate finding; passive execution is out of scope |
| Label leakage via features | medium | session-scoped rolling windows, negative-shift grep test, truncation lookahead test on every feature set, scaling inside the model | human review of group feature sets still needed |
| Overfitting across 7 people × many ideas | high | mandatory `experiment` tagging, `RunStore.trials` as `n_trials`, DSR/PBO in the table | only works if nobody deletes runs or runs outside `run_experiment` |
| Memory on laptops | medium | server-side aggregation, per-instrument lazy loading, `max_rows_in_memory` guard | raw NBBO work limited to the 30 s group on one machine |
| Timestamp alignment between NBBO bars and trade bars | medium | both from `time_m`, both bar-end labelled, `mid_consistency` and `session_bounds` errors | nanosecond fields unused; sub-ms alignment not claimed |
| Level 2 arrives late with an unknown format | high | wide book schema, importer mapping hook, `book` feature stub | replay/pre-processing script written when the format is known |
| Three branches diverge | medium | frozen interface, import-linter independence contracts, `user_plugins/` for experiments | weekly merge to `main` is a team rule, not a tool |

---

## 9. Items flagged as infeasible, ambiguous or deliberately not decided

1. **Full S&P 500 × 1 year** is not feasible through per-session WRDS SQL on one laptop;
   the design caps the manifest at 50 names × 6 months and says so in the README.
2. **NBBO column names and `qu_cond` vocabularies** must be verified on WRDS per year before
   SQL is written; the agent must not guess them.
3. **`MID_TWAP` in SQL** may be slow on WRDS; drop to a plain mean if a session query exceeds
   ~2 minutes, and record which was used in metadata.
4. **Dead zone `tau`** and the **trade threshold** are research parameters; the template
   gives defaults per horizon config and reports class balance, nothing more.
5. **Effective sample size** is reported as the overlap-based upper bound; bootstrap CIs are
   the primary uncertainty statement.
6. **Passive / limit-order execution** is out of scope (no queue model); all economic results
   are for marketable orders at the touch.
7. **GPU determinism** is not claimed; results are reproducible per recorded hardware class.
8. **Level 2 pre-processing** waits for the employer's format.
9. **`RunStore.complete` without a `BacktestResult`** needs the small `complete_ml` addition;
   do not fake a result.
10. **Monthly-TAQ-era data (pre-2015)** is not supported; `taqm_*` only.
11. **Frontend**: kept view-only; the three-horizon comparison table is a file, not a page.
    If the team later wants it in the UI, it is one new router + one page, left for later.
12. **Annualized Sharpe at 5 s** is reported because the metric plumbing exists, but the
    per-bar mean, t-stat and breakeven cost multiple are the numbers to read.

---

## 10. Order of work for the agent (one PR per step, `make check` green at each)

1. Rename + prune (§1.0, every **Delete** row, test deletions, config field removals, OpenAPI regen).
2. `Frequency` seconds + calendar session helpers + resample units + compatibility capability.
3. `core/columns.py` additions; synthetic `quotes` dataset; intraday validation checks.
4. `wrds_taq.py` (NBBO bars, trade bars renamed, raw opt-in) + fake WRDS fixtures + `pull-taq` + manifests + `fetch_partitioned`.
5. `ml/base.py`, `config.py`, `datasets.py`, `labels.py`, `features/` (`basic` only), causality tests.
6. `ml/cv.py` with invariants tests.
7. `ml/models/` registry + three implemented baselines + two stubs + scaffolds.
8. `ml/train.py`, `determinism.py`, `RunStore.complete_ml`.
9. `quoted_spread` slippage + `CostContext.extra_panels` + `prediction_signal` strategy + `ml/economic.py` + example configs.
10. `ml/evaluate.py`, `ml/deflate.py`, `ml/report.py`, `hftf ml compare`.
11. `tests/integration/test_ml_pipeline.py`, README, ADR 0008, `ml/horizons/*/` configs and READMEs.
