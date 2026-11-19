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
| Linear model fit on April, IC | 0.142 | 0.044 |
| Best sequence ensemble, IC | 0.301 | 0.101 |
| Rank corr(per-stock IC, tick size) | 0.70 | 0.24 |

Under the forward-start label the starting price contains no trade the model has seen,
and it is roughly the price a trader acting at t could obtain. The first run's tables are
kept in `results_trailing_label/` for reference.

## Files

- `L2_China_60s_Trade_Forecasting.ipynb`: the executed notebook (outputs included). It builds
  the 5 s bars from the raw files, runs integrity tests (bar edges and auction exclusion, a
  lookahead scramble test, label recomputation), trains, evaluates and runs the June test.
- `results/`: validation and test metrics, per-stock and daily tables, training histories,
  bar coverage per stock-day, run configuration.
- `results_trailing_label/`: the same tables for the first (bounce-affected) label.
- `backtest_costs.py`: the cost-aware backtest above (reads the notebook's per-window predictions and the bar cache; writes `results/backtest_*`).
- `oos/`: the July to December out-of-sample scripts (they load the notebook's code and the saved checkpoints; set `L2_RAW_DIR` and `RESULTS_DIR`).
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
