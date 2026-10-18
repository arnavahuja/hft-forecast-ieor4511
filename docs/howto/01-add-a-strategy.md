# Add a strategy

Template: [`templates/example_strategy.py`](../../templates/example_strategy.py)

1. Scaffold it:

   ```bash
   hftf new strategy my_strategy     # writes user_plugins/my_strategy.py + a test
   ```

2. Edit `user_plugins/my_strategy.py`:
   - `Params` is a pydantic model. Each field becomes a form input in the Strategy Lab
     (`Field(default, ge=..., le=..., description=...)` sets limits and help text).
   - Implement `generate_targets(data)` (whole history, vectorized) and/or `on_bar(ctx)`
     (one bar at a time, event engine).
   - Return **intent**: target weights (`output_kind = WEIGHTS`, the default) or signals
     (`output_kind = SIGNALS`, then pick a portfolio constructor in the Lab). Sizing, risk
     limits and hedging belong in overlays.
   - Declare `capabilities`, e.g. `{"engine:vectorized", "engine:event", "freq:daily"}`. The
     Lab uses them to block invalid combinations.
3. Restart `make run`, or set `HFTF_DEV_MODE=true` and the file hot-reloads on save.
4. Open the Strategy Lab. The strategy is in the list with an auto-generated form. Click
   **Run lookahead check** before trusting results.

## Rules

- Row `t` of `generate_targets` may use rows `<= t` only. Use the trailing helpers in
  `hft_forecast.core.numeric` (`rolling_mean`, `rolling_std`, `ewm_mean`, `pct_change`, `shift`).
  The engine applies the execution lag, so never shift targets yourself.
- In `on_bar`, `ctx.history(field, lookback)` ends at the current bar;
  `ctx.value_at(field, i)` raises `LookaheadError` for future bars.
- No global state and no I/O. In `on_bar`, keep state in `ctx.state`.
- `warmup()` returns the bars needed before the first valid decision.

## Test it

```bash
pytest user_plugins/tests/test_my_strategy.py
hftf check-lookahead my_config.yaml
```
