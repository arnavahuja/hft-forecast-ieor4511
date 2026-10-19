# hft-forecast-ieor4511

Short-horizon price forecasting on TAQ quote data, with the economic test run through a
backtesting engine. A fork of `backbone` pruned down to what this capstone needs: equity
quote bars, one reference strategy, and (from milestone M2) an `ml/` package for labels,
features, purged cross-validation, models and evaluation.

Three horizon groups share one pipeline: **30 s** on a 5 s mid-price grid, **5 min** and
**30 min** on a 1 min grid. The shared interfaces are frozen once M2 lands; each group then
works inside `ml/horizons/<h>/`, `ml/models/` and `ml/features/`.

Everything runs on your machine and binds to `127.0.0.1`.

## Quick start

Requirements: a Python 3.12+ conda environment, Node 20+ and pnpm.

```bash
make dev          # pip install -e ".[dev]" and pnpm install
cp .env.example .env
make run          # API on http://127.0.0.1:8000, UI on http://127.0.0.1:5173
```

Try it offline: in the Strategy Lab choose source **synthetic**, instruments `AAA, BBB, CCC`,
benchmark `MKT`, and press **Run backtest**.

From the command line:

```bash
hftf run configs/examples/buy_and_hold_synthetic.yaml
hftf data pull SPY QQQ --start 2015-01-01 --end 2024-12-31
hftf data pull-taq --manifest sp50_2024h1_5s --symbols-file symbols.txt \
  --start 2024-01-02 --end 2024-06-28 --grid 5s
hftf data import my_prices.csv
hftf new strategy my_strategy
hftf plugins
```

## What's inside

| Area | Highlights |
|---|---|
| Data | WRDS (CRSP daily with delisting returns, S&P 500 point-in-time membership, TAQ NBBO quote bars and trade bars), Yahoo for daily benchmarks, local CSV/Parquet/Excel/Feather import with a mapping wizard, synthetic data. Versioned Parquet cache, DuckDB catalog, data quality reports |
| Engines | Vectorized (closed-form drift, lag >= 1 enforced, open/close execution, rebalance schedules) and event-driven (market/limit/stop/stop-limit, TIF, partial fills, latency, fill models). Parity within 1e-10 at zero cost |
| Pipeline | `buy_and_hold` reference strategy, three portfolio constructors (equal weight, signal proportional, quantile long/short), sizing and limit overlays, before/after reporting per overlay |
| Analytics | Metric groups including deflated Sharpe, PSR and bootstrap CIs; charts, HTML tearsheet, CSV/Parquet export, comparison report, date-range brushing that recomputes metrics |
| Research | Grid/random sweeps, walk-forward, Monte Carlo, cost/delay/capacity sensitivity, regime analysis, PBO (CSCV), locked test period |
| Front end | View-only for this project: data quality reports and economic-test runs. No new pages |

## Layout

```
backend/hft_forecast/   core, data, engine, strategies, portfolio, analytics, services, api, cli
frontend/           React + Vite + TypeScript UI (generated API client in src/api)
tests/              unit, integration, golden, fixtures (no network needed)
user_plugins/       your plugins (auto-discovered, hot-reloaded in dev mode)
templates/          one copyable example per plugin kind
docs/               how-to guides, ADRs, phase log, performance notes
configs/examples/   example run configs
```

Layering (`core <- data <- engine <- analytics <- services <- api`, plugins depend only on
`core`) is enforced by import-linter. The `ml` package will sit between `analytics` and
`services` (ADR 0008).

## Development

```bash
make check        # ruff, mypy --strict, import-linter, pytest, eslint, prettier, tsc, vitest, build
make e2e          # Playwright smoke tests (starts API and UI)
make openapi      # regenerate the OpenAPI schema and the TypeScript client
pytest -m live    # opt-in tests against Yahoo / WRDS
```

## WRDS

Set `WRDS_USERNAME` in `.env`. The password is read from `~/.pgpass` by the `wrds` library;
create it once with `python -c "import wrds; wrds.Connection()"`. Settings -> Test
connection checks it. Nothing secret is stored by this package.

`hftf data pull-taq` aggregates server-side (one SQL per session), caches one Parquet file per
instrument-month, and records the result in `data/manifests/<name>.yaml`. Re-running the same
command resumes a partial pull. Expect 2-6 hours for 50 names x 6 months; run it as a
background job, not interactively.

The NBBO schema differs by year, so `wrds_taq.NBBO_COLUMNS` holds the column names that have
actually been verified on WRDS (2024 so far). Add a year only after checking it with
`wrds.Connection().describe_table("taqm_<year>", "nbbom_<yyyymmdd>")`; the source refuses to
guess. Verify the `qu_cond` vocabulary for that year at the same time.

Data scope is capped at roughly 50 names x 6 months of TAQ quote bars: a full S&P 500 year
through per-session WRDS SQL is not feasible on one laptop. Raw (unaggregated) NBBO is a
separate, uncached source capped at 5 symbols x 20 sessions.

## Documentation

- [How-to guides](docs/howto/README.md): add each kind of plugin
- [Architecture decisions](docs/adr/) — [ADR 0008](docs/adr/0008-hft-forecast-fork.md) records
  the decisions behind this fork
- [Build phase log](docs/phases.md): what the upstream platform built (history)
- [Performance](docs/performance.md)
