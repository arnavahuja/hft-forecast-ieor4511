# h5m research log

## Research question

How much of the 5-minute predictability in Level-1 (TAQ NBBO) data can actually be traded?
Decay with execution delay, a model ladder trained on the tradeable target, and the cost hurdle.

## Data (fixed)

| Role | Sample | Use |
|---|---|---|
| Research sample | 47 large S&P 500 stocks, 2024-01-02 to 2024-06-28, 1-minute bars | all exploration, modelling, selection |
| Holdout period | same 47 stocks, 2024-07-01 to 2024-12-31, 1-minute bars | run once at the end; not looked at before |

Environment: lightgbm 4.7.0. Dataset of trials 1-2: `be3b07f87bf6d3f0`.

## Methodology decisions

- IC is reported cross-sectionally (same minute, across stocks) and within stock over the full
  sample. Within-session IC is biased for price-level variables (placebo 1/mid: 0.22).
- Tradeable target: `labels.start_lag_bars: 1`, i.e. log mid(t+6) - log mid(t+1). The economic
  test decides at t and trades at t+1, so this is the return the model is judged on.
- t-statistics: aggregate to one number per day before testing (overlapping 5-minute windows).
- Selective trading uses a rule fixed in advance: trade only when |y_hat| > 1 x half spread.
- Maker (passive) fills cannot be simulated from bars: reported only as an upper bound.

## Trial register (fixed in advance; every `hftf ml run` counts, failed runs included)

| Trial | Run id | Grid | Label lag | Model | Features | Question |
|---|---|---|---|---|---|---|
| T1 | a569afa2eba8 | 1 min | 0 | logit_ridge | basic(15,30) | baseline |
| T2 | 50ab714f9562 | 1 min | 0 | last_sign (f_logret_5) | - | naive baseline |
| T3 | dac307569977 | 1 min | 1 | logit_ridge | basic(15,30) | does training on the tradeable target help? |
| T4 | 538baa451040 | 1 min | 1 | logit_ridge | basic + h30s_flow | order-flow increment |
| T5 | a0626ba4be10 | 1 min | 1 | lgbm | basic + h30s_flow | non-linear increment |
| T6 | | 1 min | 1 | h5m_mlp | basic + h30s_flow | deep learning, tabular |
| T7 | | 1 min | 1 | h5m_gru | raw sequence | deep learning, sequence |
| T8 | | 1 min | 2 | best of T3-T7 | same | robustness to delay |

Holdout (2024H2): the T3-T8 pipeline with unchanged settings, run once, logged separately.

## Results so far

- Spread "signal" (within-session IC 0.078) is a measurement artefact.
- Imbalance / microprice: cross-sectional IC 0.034 / 0.039 at lag 0, about 0 at lag 1.
- Reversal (-f_logret_5): IC 0.011 at lag 0, 0.006 at lag 1, 0.004 at lag 2 (slow decay).
- T1 at lag 1: cross-sectional IC 0.003; cost-free 0.005 bps per trade vs 2.34 bps round trip.
- Figures: figures/latency_ic.png, figures/latency_pnl.png (reports/latency_curve.csv).
