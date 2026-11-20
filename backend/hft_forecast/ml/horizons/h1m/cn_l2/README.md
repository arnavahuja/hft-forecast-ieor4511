# h1m on China A-share level 2 trades: 60 s forecasts

The 60 s pipeline from the US NBBO study (multi-horizon pretraining, 60 s fine-tuning,
per-stock heads, seed ensembles), applied to the course-provided level 2 tick-by-tick
trade data for 50 large China A-shares, January to June 2025.

| | |
|---|---|
| Data | `l2tick_2025MM.csv.gz` (trades only: price, size, aggressor side, order sequence numbers). 300M trades, aggregated to 14.6M 5 s bars |
| Universe | 50 stocks, Shanghai and Shenzhen (main boards, ChiNext, STAR) |
| Sessions | Continuous auction only, 09:30 to 11:30 and 13:00 to 14:57. Nothing crosses lunch or the overnight gap |
| Split | Train Jan to Apr (77 sessions), validation May (19), locked test June (20) |
| Label | 60 s log change of a 15 s VWAP, starting just after the forecast time: VWAP of (t, t+15 s] to VWAP of (t+60 s, t+75 s] |
| Classes | Down, flat, up with a 5 bps dead zone (36 / 31 / 33% on training labels) |
| Inputs | 60 bars (5 min) of 27 causal features: VWAP returns, volatility, signed order flow, activity, distance to price limits, time of day |
| Models | GRU, LSTM, causal TCN; 3 seeds each; 407k training windows |
| Compute | One L4 GPU (GCP), 36 min end to end |

## Results

Locked June test, LSTM seed ensemble with per-stock heads (selected on May):

| Metric | Model | Zero forecast |
|---|---|---|
| Spearman IC | **0.103** (t = 25.2 over 20 sessions) | |
| Balanced accuracy (3 classes) | **0.441** | 0.333 |
| Sign accuracy when confident (23% of windows) | 0.577 | |
| MAE (bps) | 7.78 | 7.80 |

May validation (seed ensembles, per-stock heads):

| | Spearman IC | IC t-stat | Balanced accuracy |
|---|---|---|---|
| GRU | 0.098 | 21.2 | 0.425 |
| LSTM | 0.100 | 23.3 | 0.425 |
| TCN | 0.093 | 18.8 | 0.423 |

Every one of the 50 stocks has positive validation IC (median 0.099, range 0.02 to 0.21;
43 with a t-statistic above 2). Seeds agree to within 0.01. The model beats the zero
forecast on MAE on 14 of 19 May days (sign test p = 0.06); the MAE gain is small because
the magnitude head is deliberately conservative, so IC and direction metrics are the
headline numbers.

## Out of sample: July to December 2025

The course data runs to December, so the frozen models (no retraining, no tuning) were
scored on six more months with `oos/oos_eval.py` and `oos/oos_newstocks.py`. These run
the notebook's own code; recomputing June on CPU reproduces the saved GPU predictions
(all 105,057 windows, correlation 0.99999997).

The universe was reconstituted on 1 July 2025: 22 of the 50 stocks stayed and 28 were
new. The 22 continuing stocks use their trained per-stock models. The 28 new stocks were
never seen in training, so they use the pooled model with the average stock embedding
(a cold start); July is their burn-in month, used only for volume and return scaling.

| Month | Continuing stocks: IC | t-stat | Balanced acc. | New stocks (cold start): IC | t-stat | Balanced acc. |
|---|---|---|---|---|---|---|
| Jun (all 50, locked test) | 0.103 | 25.2 | 0.441 | | | |
| Jul | 0.100 | 15.9 | 0.423 | burn-in | | |
| Aug | 0.117 | 15.2 | 0.444 | 0.140 | 14.7 | 0.457 |
| Sep | 0.115 | 20.9 | 0.436 | 0.102 | 15.9 | 0.445 |
| Oct | 0.094 | 11.1 | 0.429 | 0.098 | 14.5 | 0.440 |
| Nov | 0.085 | 15.5 | 0.428 | 0.077 | 12.2 | 0.430 |
| Dec | 0.091 | 21.0 | 0.435 | 0.092 | 13.4 | 0.442 |

The signal holds for six months without retraining and transfers to stocks the model has
never seen, which points to a general microstructure effect rather than stock-specific
fitting. IC drifts down in the fourth quarter (0.08 to 0.09 versus 0.10 to 0.12), so
periodic retraining would likely help.

After costs (`oos/oos_backtest.py`, May thresholds, continuing stocks), the picture is the
same as in June: round trips lose money at every threshold when crossing the spread (top
1%: -1.8 to -5.9 bps per trade), and only come near break-even with fees alone (-2.2 to
+0.6 bps). Execution timing saves 0.30 to 0.71 bps per order in every month (t between 8.9
and 17.4). Monthly tables are in `results/oos_2025h2/`.

## Replication: retrained on July to December

The whole notebook was rerun unchanged on the reconstituted July universe (50 stocks, 28 of
them new): train July to October, validate November, locked test December, same features,
label, architectures, seeds and selection rule (`vm/run.env` sets the months and splits).

| | First half run | Second half run |
|---|---|---|
| Train / validate / test | Jan to Apr / May / Jun | Jul to Oct / Nov / Dec |
| Validation IC by architecture | 0.093 to 0.100 | 0.094 to 0.095 |
| Model chosen on validation | LSTM, per-ticker heads | GRU, per-ticker heads |
| Locked test IC (t-stat) | 0.1025 (25.2) | 0.0965 (24.6) |
| Locked test balanced accuracy | 0.441 | 0.440 |
| Stocks with positive test IC | 50 of 50 | 50 of 50 |

In December the retrained ensemble scores 0.093 on the 22 continuing stocks and 0.099 on the
28 new ones, against 0.091 and 0.092 for the frozen first half model. The result does not
depend on the first half's stocks or months, and the three architectures again finish within
0.01 of each other. Tables are in `results_h2_jul_dec/`.

## Linear baseline

A least-squares model on the same 27 features, fit on January to April with the
forward-start label (`oos/label_and_linear.py`, rows sampled every 2 minutes), against the
LSTM seed ensemble. July to December use the 22 continuing stocks for both.

| Month | Linear IC | LSTM ensemble IC |
|---|---|---|
| May (validation) | 0.062 | 0.100 |
| Jun (locked test) | 0.073 | 0.103 |
| Jul | 0.070 | 0.100 |
| Aug | 0.086 | 0.117 |
| Sep | 0.090 | 0.115 |
| Oct | 0.058 | 0.094 |
| Nov | 0.064 | 0.084 |
| Dec | 0.062 | 0.091 |

The sequence model beats the linear model in every month, by 0.02 to 0.04 IC (about 1.3 to
1.6 times), so the 5 minutes of input history and the nonlinear model add predictive value
beyond the current feature values. The linear model also stays positive out of sample,
which supports the features themselves carrying the signal.

## Other horizons

The models also carry 15, 30 and 120 s heads from multi-horizon pretraining (only the 60 s
heads were fine-tuned). `oos/horizons.py` scores all four on the same windows, May to
December (July onward on the 22 continuing stocks); the 60 s column reproduces the tables above.

| Horizon | Mean IC | Range by month | Mean balanced accuracy |
|---|---|---|---|
| 15 s | 0.106 | 0.086 to 0.130 | 0.461 |
| 30 s | 0.108 | 0.090 to 0.130 | 0.445 |
| 60 s | 0.101 | 0.084 to 0.117 | 0.433 |
| 120 s | 0.086 | 0.076 to 0.099 | 0.427 |

Predictability is highest at 15 to 30 s and fades slowly with the horizon; balanced
accuracy is not strictly comparable across horizons because the class mix changes (15 s
moves are mostly inside the 5 bps dead zone). At 120 s the moves are larger, so the cost
picture improves a little (June to December means, top 1% of forecasts, May thresholds):

| | 60 s | 120 s |
|---|---|---|
| Gross per trade (bps) | 7.3 | 8.5 |
| Net after fees and spread (bps) | -4.4 | -3.2 |
| Net after fees only (bps) | -0.7 | +0.5 |
| Months positive after fees and spread | 0 of 7 | 1 of 7 |
| Execution-timing saving per order (bps) | 0.44 | 0.49 |

A longer horizon narrows the gap but does not close it when crossing the spread; with fees
alone the most confident 120 s forecasts are roughly break-even. Tables are in
`results/horizons/` (consecutive 120 s trades in one stock overlap by 15 s).

## Feature importance (June locked test)

Each feature's whole 60-bar input history was shuffled across windows of the same stock (one
fixed shuffle, frozen LSTM ensemble, 60 s), and the drop in pooled IC from 0.1025 is its
importance. Groups are shuffled jointly, which also captures features that stand in for one
another (`oos/feature_importance.py`, `results/feature_importance_june.csv`).

| Group shuffled | IC after | Share of IC lost |
|---|---|---|
| Price and volatility (returns, range, volatility, VWAP gap, tick) | 0.037 | 64% |
| Order flow (signed volume and count imbalances) | 0.042 | 59% |
| Activity (volume, trades, trade size, intensity, stock-relative volume) | 0.079 | 23% |
| Context (time of day, afternoon flag, distance to price limits) | 0.097 | 6% |

Largest single features: 5 s flow imbalance (19% of IC), 30 s flow imbalance (11%), 60 s and
120 s returns (10% each). The tick size, 1 m volatility and seconds since the last trade add
almost nothing on their own. Shuffled alone, the afternoon flag and `tod_sin` look important
(15% and 9%) but the context group as a whole loses only 6%: shuffling one of them alone
gives the model contradictory times of day, so their single-feature numbers overstate them.
The shares add up to more than 100% because the two large groups overlap in what they carry.

## Cost-aware backtest

`backtest_costs.py` scores the forecasts after costs. Thresholds are chosen on May and
applied unchanged to June. Assumed costs: stamp duty 5 bps on the sell leg, commission
1 bp and exchange fees 0.5 bp per leg (8 bps fixed per round trip), plus half the spread
per leg (one tick round trip, 6.1 bps on average in June) when taking liquidity.

**Intraday round trips on a base position** (sell first or buy first against existing
inventory, so T+1 does not bind), trading only the most confident forecasts. June, locked test:

| Trades taken | Trades per stock-day | Hit rate | Gross (bps) | Net, taker (bps) | Net, fees only (bps) |
|---|---|---|---|---|---|
| Top 50% | 53.8 | 0.554 | 1.16 | -12.3 | -6.8 |
| Top 20% | 22.9 | 0.578 | 1.92 | -10.8 | -6.1 |
| Top 10% | 12.1 | 0.592 | 2.49 | -10.0 | -5.5 |
| Top 5% | 6.3 | 0.615 | 3.59 | -8.8 | -4.4 |
| Top 1% | 1.4 | 0.649 | 6.55 | -5.8 | -1.5 |

The gross edge rises steadily with confidence and June matches May at every threshold,
which supports the signal being real. But even the top 1% of forecasts (6.5 bps gross,
65% hit rate) does not cover the 8 bps of fixed costs, so **the forecast is not a
profitable standalone round-trip strategy** at these costs. Closing the gap would need
passive execution (earning rather than paying the spread), a longer horizon, or lower fees.

**Execution timing.** An order that must be filled now or 60 s later pays the same fees
either way. Waiting when the forecast is adverse (a buyer waits on a predicted fall, a
seller on a predicted rise) saves **0.36 bps per order in June** (t = 18.8; May 0.35),
about 9% of what perfect foresight would save. Small per order, but it costs nothing,
which is where a 60 s signal of this size is most useful.

## Execution backtests: steered schedule and passive orders

`backtest_execution.py` tests the uses that fit A-shares (T+1, restricted shorting): doing
orders a desk must do anyway more cheaply. Every policy trades the same quantity, so fees
and stamp duty cancel and only the price differs. Settings are chosen on each run's
validation month (May, or November for the second half model) and applied unchanged later.
Each test is repeated with the forecast shuffled within stock ("random"), which keeps how
often the policy acts but removes the information. Buy and sell parents are scored together,
so market drift cancels. Outputs are in `results/execution/`.

**Steered schedule.** A parent order is spread over the day's 2 minute decision points.
TWAP does one slice at each. The steered schedule falls behind TWAP by up to K slices when
the forecast is adverse and runs ahead by up to K when it is favourable, and finishes at the
last decision point. Chosen on validation: K = 8 slices (at most 16 minutes off the TWAP path)
and act when |forecast| is above its median. Saving against TWAP, bps of the parent order:

| Month | Stocks | Steered | t-stat | Random | Days positive |
|---|---|---|---|---|---|
| Jun (locked test) | all 50 | +0.84 | 3.5 | -0.06 | 80% |
| Jul to Dec | 22 continuing | +0.92 to +2.59 | 2.0 to 5.3 | -0.42 to +0.55 | 74% to 91% |
| Aug to Dec | 28 new (cold start) | +1.01 to +2.22 | 2.8 to 4.2 | -0.46 to +0.25 | 70% to 91% |
| Dec (second half model, locked test) | all 50 | +1.03 | 4.0 | -0.32 | 78% |

The saving is positive in all 13 test month and stock-group cells, with both models, and the
shuffled forecast gives about zero. On validation the saving grows with K (May: 0.51 bps at
K = 1, 0.71 at 2, 0.87 at 4, 0.94 at 8), so even staying within one slice of TWAP helps.
Market impact is ignored, which is fair only for parent orders that are small next to
2 minute volume.

**Passive orders.** A child order that must be done within 60 s either crosses now or rests
one tick behind the last trade, and crosses at 60 s if not filled. With trades only there is
no queue position, so a resting order counts as filled either only when a later trade prints
through its price (conservative) or when one prints at it (optimistic). The switched policy
rests unless the forecast says waiting is costly. Saving against crossing now, bps per order:

| Fill rule | Always rest | Forecast-switched | Random-switched |
|---|---|---|---|
| Trade-through (conservative), June | -0.82 | -0.19 | -0.41 |
| Trade-through, Jul to Dec, all groups | -0.60 to -1.52 | -0.07 to -0.51 | -0.27 to -0.73 |
| Touch (optimistic), June | +3.35 | +3.24 | +3.22 |
| Touch, Jul to Dec, all groups | +2.17 to +4.19 | +1.97 to +4.07 | +1.97 to +4.06 |

Whether resting beats crossing depends on the fill rule, which trades alone cannot settle.
Under the conservative rule, resting loses a little and the forecast recovers 0.15 to 0.3 bps
over random switching in every month. Under the optimistic rule, resting saves 2 to 4 bps and
the forecast adds almost nothing. So for passive execution the forecast is a small refinement,
and order book data (queue position) would be needed to price the passive leg properly.

**Takeaway.** The forecast pays where it decides *when* to trade: steering a day's schedule
saves about 1 to 2.5 bps of the parent order against TWAP, every month, on stocks the model
never saw, and with the retrained model. That is the clearest economic use of this 60 s signal.

## Comparison with the team's US runs

The team's US Colab runs (v4 and v5 notebooks) used an earlier version of this pipeline
(single 60 s head, no multi-horizon pretraining or per-stock heads) on WRDS TAQ NBBO data:
12 S&P 500 megacaps, 5 s grid, snapshot midquote label, trained January to April 2024
and validated on May 2024.

| May validation | US megacaps (TAQ, midquote) | China A-shares (L2 trades, forward VWAP) |
|---|---|---|
| Spearman IC, per model and seed | 0.031 to 0.045 | 0.086 to 0.098 |
| Spearman IC, best seed ensemble | | 0.100 |
| Directional skill | Sign accuracy 51.2 to 51.9% vs 50.8% majority | Balanced accuracy 0.425 vs 0.333 |
| MAE vs zero forecast (bps) | 3.88 to 3.89 vs 3.875 (worse) | 7.55 to 7.58 vs 7.576 (slightly better) |
| Locked test evaluated | No | Yes (June IC 0.103) |

This is not a like-for-like comparison. The markets differ (US megacaps are among the most
efficient stocks anywhere; A-shares have heavy retail flow and a 1-cent tick worth 3 to 11
bps of price), the labels differ (midquote versus forward trade VWAP, and the team found
much of the snapshot-midquote IC to be bounce), and the US runs used an earlier pipeline.
A matched comparison needs the US data rerun through this notebook by a WRDS account holder.

## Why the label starts after the forecast time

There are no quotes in this feed, so there is no midquote. The first run used the
trailing 15 s VWAP at t as the starting price. It scored IC 0.30, but that was mostly
bid-ask bounce: the trailing VWAP still carries the latest prints' side of the spread,
and that error reverts. The checks in `results/label_diagnostic.csv`:

| Check (May) | Trailing label | Forward-start label |
|---|---|---|
| Single feature: last trade vs VWAP, IC | 0.129 | -0.003 |
| Linear model fit on Jan to Apr, IC | 0.145 | 0.062 |
| Best sequence ensemble, IC | 0.301 | 0.101 |
| Rank corr(per-stock IC, tick size) | 0.70 | 0.24 |

Under the forward-start label the starting price contains no trade the model has seen,
and it is roughly the price a trader acting at t could obtain. The first run's tables are
kept in `results_trailing_label/` for reference. `oos/label_and_linear.py` reproduces the
feature and linear rows (`results/label_diagnostic.csv`).

## Files

- `L2_China_60s_Trade_Forecasting.ipynb`: the executed notebook (outputs included). It builds
  the 5 s bars from the raw files, runs integrity tests (bar edges and auction exclusion, a
  lookahead scramble test, label recomputation), trains, evaluates and runs the June test.
- `results/`: validation and test metrics, per-stock and daily tables, training histories,
  bar coverage per stock-day, run configuration.
- `results_h2_jul_dec/`: the same tables for the July to December replication, plus its December test.
- `results_trailing_label/`: the same tables for the first (bounce-affected) label.
- `backtest_execution.py`: the steered schedule and passive order backtests (writes `results/execution/`).
- `backtest_costs.py`: the cost-aware backtest above (reads the notebook's per-window predictions and the bar cache; writes `results/backtest_*`).
- `oos/`: the July to December out-of-sample scripts, the label diagnostic and linear baseline, the all-horizon evaluation and the permutation feature importance (they load the notebook's code and the saved checkpoints; set `L2_RAW_DIR` and `RESULTS_DIR`).
- `vm/`: scripts to run the notebook headless on a GCP GPU VM (`launch_cn.sh` creates the VM,
  copies the notebook, runs it with papermill, uploads results and powers the VM off).

Raw tick files, the bar cache, per-window predictions and checkpoints are not committed.

## Reproduce

Put the monthly `l2tick_2025MM.csv.gz` files in `cn_l2_60s_research/raw/` (or set `L2_RAW_DIR`)
and run the notebook top to bottom. On Colab set `STORAGE = "drive"`. `NOTEBOOK_SMOKE=1` runs
a small end-to-end check (8 stocks, two months, tiny models) in a few minutes on a CPU.
`RUN_FINAL_TEST=1` evaluates the June test once.

## Caveats

1. Trades only: no spread or depth, so the label is a forward VWAP change, not a midquote change.
2. Shenzhen does not report the aggressor; it is inferred from order sequence numbers (the later order).
3. The previous close for the price-limit features is the last continuous-session trade of the prior day in the sample, not the official close.
4. No transaction costs. A-shares settle T+1 and short selling is restricted, so a down forecast is not directly tradeable; an economic test should be long-only or inventory-based.
5. One validation month and seven out-of-sample months (June to December 2025), all from one year.
