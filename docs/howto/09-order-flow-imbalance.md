# Order flow imbalance (Cont, Kukanov and Stoikov, 2014)

The paper's claim is that the events at the **best quotes alone** explain most of the
short-horizon price move. For consecutive observations of the best bid and ask,

```
e_n = 1{P_b(n) >= P_b(n-1)} q_b(n) - 1{P_b(n) <= P_b(n-1)} q_b(n-1)
    - 1{P_a(n) <= P_a(n-1)} q_a(n) + 1{P_a(n) >= P_a(n-1)} q_a(n-1)
```

A better bid adds its whole size, a worse bid removes the size that was there, and an
unchanged price contributes the change in depth; the ask enters with the opposite sign. The
order flow imbalance (OFI) over an interval is the sum of `e_n` over it. Mid-price changes
are close to linear in OFI, with a slope that falls roughly as one over the displayed depth.

This needs level 1 and nothing deeper, which is why it is the right place to start on TAQ
NBBO bars.

## Two things ship

| Where | What |
|---|---|
| `ml/features/ofi.py` | the `ofi` feature set: `f_ofi_<window>` (imbalance summed over a window, divided by trailing depth) and `f_log_depth`. **Forecasting**: past OFI predicting a future return |
| `ml/impact.py`, `hftf ml impact` | the paper's regression: `dP = alpha + beta * OFI` per stock-day. **Contemporaneous**: how much of the move the flow accounts for |

The regression is the ceiling; the feature set is the thing that has to beat the baselines.
A high R-squared in `hftf ml impact` and a flat forecast are not a contradiction — the first
says the flow and the price move together, the second says you cannot know the flow in
advance.

## Run it

```bash
# the paper's regression, 10-minute intervals by default
hftf ml impact backend/hft_forecast/ml/horizons/h5m/config.yaml --out report/impact.csv

# the forecast, with the imbalance in the feature set (already in the 1 min configs)
hftf ml run backend/hft_forecast/ml/horizons/h5m/config.yaml
hftf ml econ <run_id>
```

`hftf ml impact` prints a line per instrument and then:

| Number | What it means | The paper |
|---|---|---|
| `median_r2` | share of the interval's price move the flow explains | about 0.65 |
| `median_beta` | dollars of mid move per share of imbalance | positive, small |
| `median_beta_depth` | `beta * depth`, the quantity that should be stable across names | roughly constant |
| `share_positive_beta` | days where buying pressure moved the price up | close to 1 |
| `depth_scaling_slope` | slope of `log beta` on `log depth` | about -1 |

## The one caveat that matters

The paper works in **event time**: one `e_n` per quote update. A bar carries only the last
NBBO of the bar, so everything here is computed bar to bar and whatever happened inside a bar
is invisible — an update that lifts the bid and another that puts it back nets to zero.
`n_quotes` tells you how many updates each bar hid, and the data quality report shows it.

So the numbers are a **lower bound**: the finer the grid, the closer to the paper. Expect a
lower R-squared than 0.65 on 1 min bars, and read a rise in R-squared when you move to a 5 s
grid as confirmation rather than as a new result. `taq_nbbo_raw` (5 symbols x 20 sessions,
uncached) is the only source here that gives true event time.

## Do not validate this on synthetic quotes

The `synthetic` source paints a bid and an ask around a mid that was generated first, so
every mid tick moves both quotes and the imbalance becomes a deterministic function of the
price move. `hftf ml impact` on synthetic bars returns a median R-squared near 0.65 — the
paper's headline number — and it means nothing: there is no order flow in that data, only a
price path with quotes drawn around it. The depth scaling comes out nonsense, which is the
giveaway.

Use synthetic bars to check that the pipeline runs. Judge the paper on TAQ bars.

## Tuning

```yaml
features:
  - name: ofi
    params:
      windows: [1, 5, 10]   # bars to sum the imbalance over; set them to your horizon
      depth_window: 30      # bars of trailing depth the imbalance is divided by
```

Dividing by depth is what makes the feature comparable across names: the slope is inversely
proportional to depth, so the same imbalance means twice as much to a book half as deep.
