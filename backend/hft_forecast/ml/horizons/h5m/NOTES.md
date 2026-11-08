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
| T6 | cac38dec1522 | 1 min | 1 | h5m_mlp | basic + h30s_flow | deep learning, tabular |
| T7 | 87aee1143379 | 1 min | 1 | h5m_gru | raw sequence | deep learning, sequence |
| T8 | b68c2478d1ed | 1 min | 2 | h5m_mlp (source T6) | basic + h30s_flow | robustness to delay |

Holdout (2024H2): the T3-T8 pipeline with unchanged settings, run once, logged separately.

## Results so far

- Spread "signal" (within-session IC 0.078) is a measurement artefact.
- Imbalance / microprice: cross-sectional IC 0.034 / 0.039 at lag 0, about 0 at lag 1.
- Reversal (-f_logret_5): IC 0.011 at lag 0, 0.006 at lag 1, 0.004 at lag 2 (slow decay).
- T1 at lag 1: cross-sectional IC 0.003; cost-free 0.005 bps per trade vs 2.34 bps round trip.
- Figures: figures/latency_ic.png, figures/latency_pnl.png (reports/latency_curve.csv).
- T3 (label lag 1): cross-sectional IC 0.0054 vs T1 0.0031 on the same tradeable target;
  latency profile turns from a cliff into a slope (learns slow signals instead of imbalance).
- T4 (+ order flow): cross-sectional IC 0.0066; IC at lag 0 is 0.0008, peak at lag 1 (0.0059).
- T5 (lgbm, default h30s params, untuned by design): IC 0.0026 but highest accuracy 0.3931;
  negative IC at lag 4-5 (-0.0034): likely overfitting. Pre-stated expectation (|T5-T4| < 0.002) was wrong.
- All t-statistics above are per minute (overlapping windows); day-level t in eval_h5m.
- eval_h5m (day-level t): T4 y_hat IC 0.0066 (t 3.4) is the most reliable; T1 borderline (t 2.0).
- Direction head (p_up - p_down) beats the return head: T4 0.0098 (t 3.9), T5 0.0082 (t 3.8),
  T5 within-stock 0.0085 (t 5.9). Primary metric stays y_hat (pre-registered); 10 comparisons.
- Long-short cost 7-12 bps, far above the median spread: the extreme forecasts sit in the
  widest-spread, most volatile names. The signal is strongest where trading is most expensive.
- Selective rule (|y_hat| > half spread) trades 3-7.5% of rows; T4 net -1.08 bps per trade.
- T7 (GRU, T6 + raw sequence): y_hat IC 0.0086 (day-level t 3.07), below T6's 0.0090
  (t 3.27); the pre-registered T8 source rule therefore selects T6.
- T8 (T6 MLP, trained with label lag 2): y_hat IC 0.0096 (day-level t 3.27), evaluated on
  the fixed lag-1 tradeable target for comparison; long-short net -7.14 bps.

Pre-registered before T6-T8:
- every trial reports both heads; y_hat stays primary.
- step 10 adds ONE cost-aware ranking, score / quoted spread, decided now (not after seeing results).
- T7 clarified before running: T6 + h5m_seq (6 raw L1 variables x 20 lags) read by a GRU; the
  engineered features enter through a dense branch, so T7 - T6 = value of the raw sequence.
- T8 source rule: the trial among T3-T7 with the highest day-level cross-sectional IC of y_hat
  in eval_h5m; T8 = that config with labels.start_lag_bars 2.
- eval_h5m reports a third head, y_hat / quoted spread (cost-aware ranking).
