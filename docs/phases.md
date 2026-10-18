# Build phase log

One short summary per phase: what was built, what was deferred, known limitations, how to demo.

## Phase 0: Foundation

**Built.** `pyproject.toml` (editable install into the `hft_forecast` conda env), Makefile
(`make dev/test/lint/typecheck/imports/check/run`), pre-commit config, `.env.example`, ADRs.
`core`: immutable domain types (`Instrument`, `DataRequest`, `MarketData`, `TargetFrame`,
`Order`, `Fill`, `Position`, `PortfolioState`), canonical column names, error hierarchy rooted
at `BackboneError`, pydantic-settings config, structlog logging, trading calendar, interfaces
for all ten plugin kinds, the generic `Registry[T]` with `@register`, and discovery of
built-in packages, `user_plugins/` and the `hft_forecast.plugins` entry-point group. Import
failures are recorded (plugin health), duplicates raise.

**Demo.** `hft_forecast plugins`; `pytest tests/unit/core`.

## Phase 1: Data

**Built.** `DataService` (single entry point), versioned Parquet cache keyed by the request
hash without dates (partial overlaps fetch only the missing range; every write is an
immutable snapshot a run can pin), DuckDB-queried catalog (JSON records + Parquet),
validation (duplicates, ordering, non-positive prices, calendar gaps, extreme jumps),
corporate-action conversions (raw / split / total return), Yahoo source (recorded-response
tests), local-file import (preview, guessed mapping, YAML mapping saved for one-click
re-import, extra columns kept), a deterministic synthetic source, and two universe
providers (static, point-in-time liquidity top-N).

**Limitations.** Interior gaps in a cached range are not back-filled (only extensions at
either end). Yahoo intraday is limited by Yahoo's own windows.

**Demo.** `hftf data pull SPY TLT --start 2018-01-01 --end 2020-12-31`, then again with a
later `--end` (only the new range is fetched); `hftf data import my.csv`;
`hftf data catalog`.

## Phase 2: Vectorized engine

**Built.** Closed-form drift simulation (ADR 0005), lag >= 1 enforced, close or open
execution, rebalance rules (`on_change` default, every bar, weekly, monthly, quarterly,
every N), cash interest, round-trip trade extraction, orders/fills tables. Cost models
(zero, fixed, per share, bps, tiered, futures/options per contract, short borrow, margin
interest) and slippage (half spread, volatility-scaled, square-root impact). Reference
strategies: buy and hold, SMA crossover, time-series momentum. All metric groups (return,
risk, risk-adjusted incl. PSR/DSR, drawdown, benchmark-relative, trades, portfolio, costs,
statistical with stationary bootstrap, factor). SQLite + Parquet run store, runner with
compatibility checker, CLI `hftf run config.yaml`, lookahead truncation check.

**Tests.** Hand-computed engine cases, Hypothesis invariants (zero-cost buy and hold equals
the asset return; cash + positions reconcile to equity), metrics cross-checked against
empyrical, golden snapshots, lookahead on every shipped strategy plus a negative control,
determinism, generic plugin-contract suite.

**Demo.** `hftf run configs/examples/sma_synthetic.yaml` (offline) or
`configs/examples/sma_spy_yahoo.yaml`.

## Phase 3: API and UI

**Built.** FastAPI under `/api/v1` (plugins, data, runs/results, compare, jobs + WebSocket,
presets, settings), one structured error format, LTTB downsampling, localhost-only binding
and CORS. Background jobs in a `ProcessPoolExecutor` with progress events and cooperative
cancellation. React + Vite + TypeScript (strict) front end: generated OpenAPI client,
TanStack Query, Zustand, ECharts through a chart-type registry, auto-generated parameter
forms (rjsf with themed templates), dark theme from tokens with no purple (tested). Pages:
Dashboard, Data (catalog, pull, import wizard with mapping editor, quality report), Strategy
Lab (six steps, live validation, drag-to-reorder overlays, presets, lookahead check), Run
Detail (tabs, synchronized zoom, brushing recomputes metrics, exports and HTML tearsheet),
Runs (filters, bulk actions), Compare (best/worst highlighting, all comparison charts,
config-difference warnings, combined portfolio), Settings.

**Tests.** API end to end with TestClient; Vitest for SchemaForm, the chart registry and
formatting; Playwright smoke test (build a run, see results, compare two runs) and a
computed-style check that no purple is rendered.

**Demo.** `make run`, open http://127.0.0.1:5173.

## Phase 4: Pipeline

**Built.** Constructors: equal weight, signal-proportional, inverse volatility, risk parity,
minimum variance, mean-variance, hierarchical risk parity, quantile / top-N long-short (risk
estimates are trailing and recomputed on a schedule or when the active set changes).
Overlays: vol targeting, fractional Kelly, fixed fractional, ATR sizing; stop loss, trailing
stop, take profit, time exit; max position, max group (sector), exposure limits, leverage
cap, turnover cap, liquidity cap; drawdown de-risking, circuit breaker, CPPI; beta hedge,
dollar and sector neutrality, pairs hedge, currency placeholder; trend, volatility and
user-regime filters. Every overlay's effect is recorded generically (before/after zero-cost
returns, gross exposure, cells changed, notes) and shown on the Overlays tab. Reference
strategies added: cross-sectional momentum, low volatility, mean reversion, pairs trading,
factor composite, futures trend.

**Tests.** Hypothesis properties (limit overlays always respect caps), behavioural tests
for every overlay group, constructor maths (risk contributions, neutrality, bounds).

## Phase 5: Full analytics and Compare

**Built.** All metric groups and chart groups (performance, drawdown, distribution, rolling,
positions, trades, costs, overlays, comparison); Compare page with best/worst highlighting,
config-difference warnings, date alignment and a combined-portfolio builder; CSV/Parquet
exports and an HTML tearsheet (print to PDF).

**Acceptance.** `test_compare_five_runs_across_every_comparison_chart`.

## Phase 6: Event-driven engine

See ADR 0006. **Acceptance.** Parity tests (7 configurations + an overlay) within 1e-10; a
breakout strategy with resting stop orders runs (`breakout_stop`).

## Phase 7: WRDS

See ADR 0007. CRSP daily/monthly with delisting returns, Compustat (availability-dated),
Fama-French 5 + momentum, point-in-time S&P 500 membership, factor metrics and charts
(exposures, rolling betas, attribution). **Acceptance** (offline, fake WRDS):
`test_crsp_momentum_with_pit_universe_and_factor_attribution`. **Live check:** set
`WRDS_USERNAME`, create `~/.pgpass`, then `pytest -m live`.

## Phase 8: Research tools

Grid/random sweeps (threaded), walk-forward (rolling or anchored, stitched OOS), Monte Carlo
(stationary bootstrap and trade shuffling), cost / delay / capacity sensitivity with
breakeven, regime analysis, deflated Sharpe counting sweep trials per experiment, PBO via
CSCV, test-period lock with recorded unlocks. Results render generically on the Research page.

## Phase 9: Futures and intraday

Continuous futures with pluggable roll rules (fixed days, volume, open interest) and ratio or
difference back-adjustment; roll costs in both engines; futures margin reported by the
ledger; intraday synthetic bars, Yahoo intraday and TAQ bars; multi-frequency (`resampled`
returns only completed coarser bars). **Acceptance:** `futures_trend` in the event engine,
`intraday_orb` and `daily_trend_intraday` (daily signals, intraday execution).

## Phase 10: Options

Black-Scholes prices, Greeks and implied vol; chain helpers (select by delta, moneyness,
DTE); rolling option instruments added by plugins through `synthetic_instruments`, priced
from an OptionMetrics chain when configured (`data.options_chain`) or model-priced from
implied/realized vol (runs are then flagged and warned). Engines take option returns from an
explicit `ret` series so expiring legs are handled exactly; the ledger cash-settles expired
contracts at intrinsic value. Overlays: protective put, collar, put spread, tail hedge
(premium budget), covered-call overwrite. Strategies: covered call, protective put. Charts:
payoff at expiry, Greeks over time, hedge cost vs protection; metrics: option P&L, carry per
year, payoff in stress periods, hedge efficiency.

**Acceptance.** `test_protective_put_delivers_protection_in_stress` (shallower drawdown,
positive option P&L in stress) with hedge cost reporting.

**Limitations.** Model pricing uses a flat vol per bar (no skew). Chain-based pricing needs
underlying ids that match the chain's tickers (Yahoo tickers work; CRSP PERMNOs need
mapping).

## Phase 11: Polish

How-to guides for every plugin kind (`docs/howto/`), copyable templates (`templates/`),
`hftf new <kind> <name>` scaffolding with a generated test, hot reload of
`user_plugins/` (compiled from source so edits are never hidden by cached bytecode),
engine profiling (`scripts/profile_engines.py`, `docs/performance.md`), Playwright smoke
tests (run, results, compare, research sweep, no-purple check).

**Acceptance.** `tests/integration/test_extensibility.py`: a scaffolded user strategy,
metric and chart are discovered, run through the API, shown on Run Detail and compared
against a built-in run with zero edits outside `user_plugins/`.

## Factor portfolios and comparison report

WRDS: `crsp_common` share-code universe, delisting fill rule, monthly Fama-French factors,
`crsp_index`, `crsp_treasury`, Fama-French book equity. Universe `market_cap_top_n`.
Strategies `trailing_return`, `field_signal`, `composite_signal`, `fixed_weights`,
`trend_filter`. Value-weighted, capped legs in `quantile_long_short`. Run-level risk-free
series (cash interest and excess-return Sharpe), splices, `source:dataset:SYMBOL`
benchmarks, month matching across sources, CAPM metrics. Compare page report export (CSV +
PNG), `hft_forecast preset import`. See `docs/howto/08-factor-portfolios-and-reports.md`.

**Acceptance.** `tests/integration/test_factor_benchmarks.py` (fake WRDS).

**Limitations.** The legacy CRSP stock files end in December 2024 (`msf_v2` is not read yet).

## Deferred / known limitations (overall)

- Live or paper trading (non-goal for v1); the `Broker` protocol is the extension point.
- Multi-currency (instruments carry `currency`; only USD is handled).
- WRDS and Yahoo live paths are tested against recorded or fake responses; run
  `pytest -m live` with credentials to exercise them for real.
- Cached ranges are extended at either end; holes inside a cached range are not back-filled.
- The Compare page aligns dates on request but does not re-run strategies over the window.
