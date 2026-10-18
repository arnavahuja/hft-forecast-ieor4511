# Add a portfolio constructor

Template: [`templates/example_portfolio_constructor.py`](../../templates/example_portfolio_constructor.py)

```bash
hftf new portfolio_constructor my_constructor
```

Constructors turn a strategy's **signals** into **weights**:

```python
def construct(self, signals: TargetFrame, ctx: ConstructionContext) -> TargetFrame: ...
```

`ctx.data` and `ctx.periods_per_year` are available. For risk-based sizing, reuse
`hft_forecast.core.portfolio_math`: `trailing_cov`, `inverse_vol_weights`,
`risk_parity_weights`, `min_variance_weights`, `mean_variance_weights`, `hrp_weights`, and
`rolling_construct`, which recomputes on a schedule and holds weights in between so turnover
stays realistic. Row `t` must use data up to `t` only; the lookahead check runs the
constructor too.
