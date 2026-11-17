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
5. One validation month and one test month. The second half of 2025 is available for walk-forward validation.
