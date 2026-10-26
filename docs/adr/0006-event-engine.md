# ADR 0006: Event-driven engine

Status: superseded by [ADR 0010](0010-prune-to-the-forecast-comparison.md)

## Decision
- Loop per bar: holding costs and cash interest over the bar (start-of-bar positions) ->
  dividends/splits -> broker fills eligible orders -> mark to market at the close -> decisions.
- Orders become eligible `lag_bars + latency_bars` after the decision bar.
- **Target-weight orders** (`set_target_weights`) are sized when they execute, from the equity
  and price at that moment, exactly like the vectorized engine. With zero costs the two engines
  agree to 1e-10 per bar (parity tests: buy and hold, SMA incl. shorts and open execution,
  time-series momentum, every-bar mean reversion, monthly rebalancing with lag 2, cash
  interest, and a vol-target overlay).
- Explicit orders support market, limit, stop and stop-limit, DAY/GTC/IOC, partial fills via
  a participation cap, and latency. Stops fill at the stop or the gapped open; limits at the
  limit or the better open.
- Vectorized-form strategies run through an adapter that replays pipeline-processed targets on
  the vectorized rebalance schedule. `on_bar`-only strategies run natively; overlays with an
  `on_bar` form edit staged targets.
- The broker sits behind a `Broker` protocol so a live adapter can replace the simulator.
- `Context` exposes history views ending at the current bar (read-only); `value_at` raises
  `LookaheadError` for future bars.

## Limitations
Futures are carried at notional value (margin is reported, not financed). Margin-interest
spread is applied by the vectorized engine only; the event engine charges the cash rate on
negative cash.
