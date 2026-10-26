# Add a cost or slippage model

Template: [`templates/example_cost_model.py`](../../templates/example_cost_model.py)

```bash
hftf new cost_model my_fees
```

| Kind | What you implement |
|---|---|
| `cost_model` | `vectorized(ctx) -> (T, N)`: cost at each bar as a fraction of equity |
| `slippage_model` | `vectorized(ctx) -> (T, N)`: the same, for spread and impact |

`CostContext` provides `trades` (signed weight traded), `holdings`, `prices`, `volumes`,
`equity`, `returns`, `multipliers`, `asset_classes` and `traded_units()`. Estimates such as
volatility must use rows `<= t`. Set `category` (`commission`, `fees`, `borrow`, `financing`)
to control how costs are reported, and declare `engine:vectorized` in `capabilities`.

A model that needs a data field the context does not carry (the quoted spread, say) lists it
in `requires`; the engine then puts that panel in `ctx.extra_panels` and `ctx.panel(name)`
reads it. `quoted_spread` is the worked example, and it is the model the economic test is
judged on.
