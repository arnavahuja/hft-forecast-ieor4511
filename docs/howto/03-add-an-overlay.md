# Add an overlay

Template: [`templates/example_overlay.py`](../../templates/example_overlay.py)

```bash
hftf new overlay my_overlay
```

Overlays take proposed targets and return modified targets. They run in the order chosen in
the Lab (drag to reorder):

```python
def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame: ...
def on_bar(self, ctx: Context) -> None: ...   # optional event-engine form
```

`OverlayContext` provides:

- `data`: market data. Row `t` of your output may only use rows `<= t`.
- `simulate(targets)`: zero-cost returns of any target frame on the engine's timing, for
  path-dependent rules (volatility targeting, drawdown control).
- `benchmark`, `periods_per_year`, `lag_bars`, `initial_capital`.
- `notes`: append a string and it appears on the Run Detail **Overlays** tab.

You don't need to write any reporting. The pipeline records what every overlay changed
(gross exposure, zero-cost return before and after, cells changed) and the Overlays tab
shows it.

The event form reads `ctx.target_weights()` and calls `ctx.set_target_weights(...)`. Declare
`capabilities` with `engine:event` when you implement it.
