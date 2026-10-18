# Add a cost, slippage or fill model

Template: [`templates/example_cost_model.py`](../../templates/example_cost_model.py)

```bash
hftf new cost_model my_fees
```

| Kind | Vectorized engine | Event engine |
|---|---|---|
| `cost_model` | `vectorized(ctx) -> (T, N)` cost as a fraction of equity | `commission(qty, price, instrument)`, `holding_cost(value, instrument, year_fraction)` |
| `slippage_model` | `vectorized(ctx)` | `price_adjustment(qty, price, bar)` (adverse, per unit) |
| `fill_model` | (not used) | `fill(order, bar, max_quantity) -> (price, quantity)` |

`CostContext` provides `trades` (signed weight traded), `holdings`, `prices`, `volumes`,
`equity`, `returns`, `multipliers`, `asset_classes` and `traded_units()`. Estimates such as
volatility must use rows `<= t`. Set `category` (`commission`, `fees`, `borrow`, `financing`)
to control how costs are reported. Declare `engine:vectorized` / `engine:event` in
`capabilities` for the engines you support.
