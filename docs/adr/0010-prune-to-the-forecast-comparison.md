# ADR 0010: Prune the platform to the forecast comparison

Status: accepted

Supersedes ADR 0006 (event-driven engine). Extends ADR 0008.

## Context
The repository exists for one thing: three groups forecast mid-price returns at three
horizons, and the results are compared statistically and economically in one table. ADR 0008
pruned the general platform once, but kept several subsystems because they were already
built and harmless. They turned out not to be harmless: every one of them is a second way to
do something, a page in the UI, or a config field that nobody on this project should touch,
and all of it is code the team has to read past.

The test applied here is: can a forecast reach it? If no prediction, label, fold or
comparison-table row can reach a subsystem, it is not part of this project.

## Decision
Removed:

- **The event-driven engine** (`engine/event/`, `engine/fills/`), with `FillModel`,
  `BarSnapshot`, the `Context` protocol, `Strategy.on_bar`, `Overlay.on_bar`,
  `CostModel.commission`, `CostModel.holding_cost` and `SlippageModel.price_adjustment`.
  `prediction_signal` is vectorized-only and `ml/economic.py` asks for the vectorized engine
  by construction, so no forecast could ever reach it. `ExecutionSpec` loses `engine`,
  `latency_bars` and `max_participation`, and `BacktestConfig` loses `fill_model`.
- **The research service** (`services/research.py`, its router, the Research page): sweeps,
  walk-forward, Monte Carlo, sensitivity, regimes and the locked test period. Walk-forward is
  `ml/cv.py` for this project, PBO is `ml/deflate.py`, and sweeps actively conflict with the
  trial-counting rule that every evaluated configuration is its own `run_experiment` call.
  `analytics/robustness` keeps the three functions `ml` imports: `walk_forward_windows`,
  `pbo_cscv` and `breakeven`.
- **The Strategy Lab** and the preset store behind it. ADR 0008 already said the front end is
  view-only for this project; the Lab contradicted that, and a backtest hand-built in the
  browser is a trial of an experiment that nobody recorded. With it go `SchemaForm`,
  `PluginPicker`, the `rjsf` and `dnd-kit` dependencies, and the `/runs/validate` and
  `/runs/lookahead-check` endpoints it alone called (`hftf check-lookahead` remains).
- **`synthetic_markets`** (`gbm_intraday`): unreferenced, and it produces OHLCV without
  quotes, which the feature sets cannot use. `synthetic`/`quotes` is the offline source.
- **`docs/phases.md`, `docs/performance.md` and `scripts/profile_engines.py`**: they describe
  strategies and engines that no longer exist, so they misdirect rather than inform.

Kept, against the first instinct to cut them:

- **`portfolio/`** in full. The design names `signal_proportional` and `vol_target` as
  options for the cross-sectional economic test, so they are plausible, not dead.
- **`breakeven` and the cost-multiple grid.** The comparison table has a
  `breakeven_cost_multiple` row, and deleting the research service removed its only
  implementation. `ml/economic.cost_multiple_configs` replaces it in nine lines: one backtest
  per multiple of the quoted spread, which is also why `VectorizedEngine.cost_scale` could go.
- **CRSP and the universes**: they are how the team picks the names it pulls TAQ quotes for.

## Consequences
- One engine, one way to launch a run (the CLI, or the API the CLI-equivalent endpoints
  expose), one place where cost sensitivity lives.
- The front end is what ADR 0008 said it was: the catalog and quality reports, the run list,
  run detail and Compare. Its bundle drops by about 300 kB.
- Anything removed here is one `git revert` away, and nothing removed is referenced by the
  frozen interface of ADR 0008.
