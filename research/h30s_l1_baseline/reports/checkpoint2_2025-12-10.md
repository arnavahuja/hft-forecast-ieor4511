# Checkpoint 2 — feature/label panel, 2025-12-10

- Panel: **684,030 rows** = 30 symbols (29 targets + SPY) × 22,801 seconds (09:35:00–15:55:00); 85 features; labels for h ∈ [5, 10, 30, 60] s.
- Files: `data/features/date=2025-12-10/part.parquet`, `data/labels/date=2025-12-10/part.parquet`.

## Missing values

Features: **no missing values** in any of the 85 columns.
Labels: **no missing values** (15:55 + 60 s < 16:00, so no label crosses the close).

## Zero mid-change fractions by stock

Share of grid seconds with exactly unchanged mid over h seconds; `flat_band_30s` = share with |Δmid| < half spread (dir = 0). Sorted by `zero_30s`.

| symbol | role | price | med_spread_ticks | zero_5s | zero_10s | zero_30s | zero_60s | flat_band_30s | std_r30_bps |
|---|---|---|---|---|---|---|---|---|---|
| F | target | 13.14 | 1.0 | 94.2% | 89.2% | 75.4% | 62.1% | 75.4% | 4.38 |
| T | target | 24.41 | 1.0 | 90.0% | 82.8% | 63.8% | 47.9% | 63.8% | 3.82 |
| PFE | target | 25.55 | 1.0 | 84.2% | 73.4% | 50.5% | 38.6% | 50.5% | 4.20 |
| VZ | target | 39.86 | 1.0 | 83.1% | 71.7% | 47.5% | 33.2% | 47.5% | 3.22 |
| KMI | target | 26.76 | 1.0 | 80.5% | 68.4% | 45.1% | 32.7% | 45.6% | 4.49 |
| BAC | target | 53.84 | 1.0 | 62.7% | 50.1% | 31.5% | 23.1% | 31.6% | 4.99 |
| KO | target | 70.09 | 1.0 | 64.8% | 48.9% | 27.1% | 18.3% | 27.7% | 2.87 |
| CSCO | target | 79.84 | 1.0 | 51.1% | 35.5% | 17.1% | 10.7% | 19.2% | 3.95 |
| INTC | target | 40.14 | 1.0 | 40.7% | 28.0% | 13.7% | 10.1% | 14.1% | 9.66 |
| WFC | target | 89.70 | 1.0 | 32.9% | 20.9% | 11.2% | 7.0% | 16.4% | 5.03 |
| XOM | target | 119.25 | 1.0 | 29.8% | 16.8% | 8.7% | 6.4% | 18.8% | 3.69 |
| BRK.B | target | 493.36 | 17.0 | 51.1% | 32.2% | 8.6% | 2.5% | 59.8% | 2.70 |
| WMT | target | 114.12 | 1.0 | 31.0% | 17.3% | 7.5% | 4.7% | 15.0% | 4.50 |
| HD | target | 346.95 | 15.0 | 47.5% | 28.7% | 6.5% | 1.8% | 50.3% | 4.60 |
| MRK | target | 97.30 | 2.0 | 36.2% | 20.2% | 6.3% | 3.6% | 25.5% | 4.80 |
| PG | target | 140.49 | 3.0 | 33.8% | 17.9% | 5.8% | 3.9% | 33.2% | 3.24 |
| CVX | target | 150.68 | 2.0 | 23.8% | 12.6% | 5.7% | 3.3% | 26.2% | 3.54 |
| JNJ | target | 204.06 | 6.0 | 35.0% | 19.4% | 5.6% | 2.8% | 43.2% | 3.70 |
| NVDA | target | 183.06 | 1.0 | 12.6% | 8.6% | 5.0% | 3.5% | 6.3% | 5.47 |
| PEP | target | 149.00 | 4.0 | 35.9% | 18.9% | 5.0% | 2.6% | 36.8% | 4.83 |
| SPY | benchmark | 683.40 | 1.0 | 9.5% | 6.7% | 3.3% | 2.4% | 3.7% | 2.63 |
| AAPL | target | 278.43 | 3.0 | 12.3% | 6.1% | 3.0% | 2.0% | 18.3% | 3.45 |
| AMZN | target | 231.28 | 3.0 | 9.0% | 5.2% | 2.5% | 1.7% | 15.6% | 4.80 |
| AMD | target | 220.14 | 8.0 | 9.9% | 4.6% | 2.1% | 1.3% | 29.2% | 7.47 |
| GOOGL | target | 317.61 | 4.0 | 6.2% | 3.6% | 1.9% | 1.2% | 16.1% | 4.90 |
| JPM | target | 306.29 | 10.0 | 25.6% | 10.9% | 1.9% | 1.3% | 30.3% | 5.64 |
| MSFT | target | 479.16 | 5.0 | 6.3% | 2.8% | 1.4% | 0.8% | 15.4% | 4.41 |
| TSLA | target | 445.99 | 6.0 | 3.2% | 2.0% | 1.1% | 0.7% | 12.3% | 6.64 |
| AVGO | target | 403.91 | 12.0 | 7.7% | 2.7% | 1.1% | 0.7% | 24.2% | 6.65 |
| META | target | 649.75 | 16.0 | 11.7% | 3.9% | 1.0% | 0.6% | 28.2% | 4.79 |

## Labels (29 targets pooled)

| horizon_s | mean_bps | std_bps | p1_bps | p99_bps | zero | dir_up | dir_flat | dir_down |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.01951 | 2.042 | -6.082 | 6.288 | 38.4% | 19.6% | 61.3% | 19.1% |
| 10 | 0.03945 | 2.863 | -8.178 | 8.26 | 27.7% | 26.0% | 48.9% | 25.1% |
| 30 | 0.1169 | 4.924 | -13.69 | 14.87 | 16.0% | 35.3% | 30.9% | 33.7% |
| 60 | 0.2342 | 6.895 | -18.42 | 20.21 | 11.3% | 39.8% | 22.1% | 38.1% |

## Feature summary (29 targets pooled)

| feature | mean | std | p1 | p50 | p99 | zero_share |
|---|---|---|---|---|---|---|
| spread_bps | 2.609 | 1.772 | 0.5431 | 2.135 | 7.637 | 0.0% |
| spread_ticks | 4.918 | 6.16 | 1 | 2 | 29 | 0.0% |
| quote_imbalance | 0.006289 | 0.386 | -0.8305 | 0 | 0.8261 | 20.0% |
| microprice_dev | 3.261e-07 | 5.932e-05 | -0.0001746 | 0 | 0.0001697 | 14.6% |
| time_since_last_trade | 0.516 | 0.7705 | 0.001587 | 0.2393 | 3.64 | 0.0% |
| time_since_last_mid_change | 13.6 | 47.99 | 0.005839 | 1.762 | 215.5 | 0.0% |
| secs_since_open | 1.17e+04 | 6582 | 528 | 1.17e+04 | 2.287e+04 | 0.0% |
| ret_1 | 3.759e-07 | 9.183e-05 | -0.0002854 | 0 | 0.0002942 | 65.8% |
| ofi_1 | 0.005213 | 3.171 | -10.14 | 0 | 10.2 | 35.4% |
| trade_imb_1 | -0.008429 | 0.7142 | -1 | 0 | 1 | 15.8% |
| trade_cnt_imb_1 | 0.07412 | 0.607 | -1 | 0 | 1 | 23.7% |
| n_trades_1 | 1.775 | 1.281 | 0 | 1.609 | 4.927 | 14.9% |
| volume_1 | 4.358 | 2.735 | 0 | 4.787 | 9.36 | 14.9% |
| n_quote_updates_1 | 1.671 | 1.465 | 0 | 1.386 | 5.118 | 27.5% |
| avg_spread_1 | 2.609 | 1.772 | 0.5431 | 2.135 | 7.637 | 0.0% |
| ret_2 | 7.526e-07 | 0.0001301 | -0.0003951 | 0 | 0.0004037 | 54.2% |
| ofi_2 | 0.01091 | 4.632 | -14.61 | 0 | 14.44 | 22.2% |
| trade_imb_2 | -0.03387 | 0.6784 | -1 | 0 | 1 | 5.3% |
| trade_cnt_imb_2 | 0.06703 | 0.5457 | -1 | 0 | 1 | 13.1% |
| n_trades_2 | 2.471 | 1.31 | 0 | 2.398 | 5.553 | 4.7% |
| volume_2 | 5.693 | 2.382 | 0 | 6.089 | 9.949 | 4.7% |
| n_quote_updates_2 | 2.359 | 1.551 | 0 | 2.398 | 5.717 | 14.2% |
| avg_spread_2 | 2.609 | 1.752 | 0.544 | 2.134 | 7.637 | 0.0% |
| ret_5 | 1.881e-06 | 0.000204 | -0.0006082 | 0 | 0.0006271 | 38.3% |
| ofi_5 | 0.02566 | 7.674 | -23.84 | 0 | 23.33 | 9.9% |
| trade_imb_5 | -0.05198 | 0.5584 | -1 | -0.06199 | 1 | 0.5% |
| trade_cnt_imb_5 | 0.05756 | 0.4178 | -1 | 0.06667 | 1 | 5.3% |
| n_trades_5 | 3.492 | 1.222 | 0.6931 | 3.497 | 6.428 | 0.3% |
| volume_5 | 7.256 | 1.742 | 1.946 | 7.421 | 10.72 | 0.3% |
| n_quote_updates_5 | 3.396 | 1.505 | 0 | 3.497 | 6.578 | 4.0% |
| avg_spread_5 | 2.609 | 1.728 | 0.5465 | 2.141 | 7.637 | 0.0% |
| ret_10 | 3.709e-06 | 0.0002863 | -0.0008182 | 0 | 0.0008229 | 27.7% |
| ofi_10 | 0.0464 | 11.41 | -35.32 | 0 | 34.57 | 4.8% |
| trade_imb_10 | -0.05519 | 0.4569 | -0.9652 | -0.06068 | 0.9429 | 0.1% |
| trade_cnt_imb_10 | 0.05174 | 0.3396 | -0.75 | 0.05928 | 0.7922 | 2.7% |
| n_trades_10 | 4.272 | 1.122 | 1.792 | 4.277 | 7.108 | 0.0% |
| volume_10 | 8.23 | 1.411 | 4.575 | 8.28 | 11.32 | 0.0% |
| n_quote_updates_10 | 4.201 | 1.393 | 0 | 4.263 | 7.246 | 1.1% |
| avg_spread_10 | 2.609 | 1.708 | 0.547 | 2.158 | 7.631 | 0.0% |
| rv_10 | 8.435e-08 | 2.443e-07 | 0 | 2.472e-08 | 8.583e-07 | 22.0% |
| ret_30 | 1.084e-05 | 0.0004933 | -0.001381 | 0 | 0.001485 | 16.0% |
| ofi_30 | 0.1513 | 22.03 | -68 | 0.0122 | 66.69 | 1.5% |
| trade_imb_30 | -0.05608 | 0.3233 | -0.7989 | -0.05854 | 0.7211 | 0.0% |
| trade_cnt_imb_30 | 0.04427 | 0.2516 | -0.5741 | 0.04249 | 0.6391 | 1.0% |
| n_trades_30 | 5.468 | 0.9888 | 3.434 | 5.394 | 8.204 | 0.0% |
| volume_30 | 9.563 | 1.146 | 6.849 | 9.536 | 12.3 | 0.0% |
| n_quote_updates_30 | 5.427 | 1.215 | 2.303 | 5.398 | 8.324 | 0.0% |
| avg_spread_30 | 2.611 | 1.675 | 0.5477 | 2.202 | 7.631 | 0.0% |
| rv_30 | 2.533e-07 | 5.503e-07 | 0 | 1.121e-07 | 2.244e-06 | 9.8% |
| ret_60 | 2.165e-05 | 0.0006943 | -0.001859 | 0 | 0.002029 | 11.3% |
| ofi_60 | 0.404 | 34.05 | -104.8 | 0.1547 | 105.3 | 0.8% |
| trade_imb_60 | -0.05617 | 0.2574 | -0.6812 | -0.05785 | 0.5842 | 0.0% |
| trade_cnt_imb_60 | 0.04137 | 0.2093 | -0.4783 | 0.033 | 0.5844 | 0.6% |
| n_trades_60 | 6.196 | 0.9369 | 4.394 | 6.089 | 8.891 | 0.0% |
| volume_60 | 10.33 | 1.064 | 7.879 | 10.29 | 12.94 | 0.0% |
| n_quote_updates_60 | 6.161 | 1.151 | 3.434 | 6.105 | 9.005 | 0.0% |
| avg_spread_60 | 2.613 | 1.657 | 0.5559 | 2.228 | 7.631 | 0.0% |
| rv_60 | 5.077e-07 | 9.835e-07 | 0 | 2.409e-07 | 4.276e-06 | 5.4% |
| ret_120 | 4.364e-05 | 0.0009834 | -0.002621 | 0 | 0.002953 | 7.9% |
| ofi_120 | 1.001 | 54.25 | -167.9 | 0.4374 | 172.2 | 0.5% |
| trade_imb_120 | -0.05629 | 0.2064 | -0.5707 | -0.05706 | 0.4693 | 0.0% |
| trade_cnt_imb_120 | 0.03932 | 0.1773 | -0.3828 | 0.02672 | 0.5529 | 0.4% |
| n_trades_120 | 6.91 | 0.9057 | 5.288 | 6.787 | 9.595 | 0.0% |
| volume_120 | 11.07 | 1.013 | 8.764 | 11.04 | 13.61 | 0.0% |
| n_quote_updates_120 | 6.879 | 1.116 | 4.304 | 6.809 | 9.678 | 0.0% |
| avg_spread_120 | 2.618 | 1.646 | 0.5558 | 2.251 | 7.631 | 0.0% |
| rv_120 | 1.02e-06 | 1.817e-06 | 0 | 5.074e-07 | 8.524e-06 | 2.4% |
| ret_300 | 0.0001058 | 0.001545 | -0.003923 | 0 | 0.00488 | 4.9% |
| ofi_300 | 3.447 | 101.6 | -304.4 | 1.102 | 344 | 0.3% |
| trade_imb_300 | -0.05694 | 0.1581 | -0.4498 | -0.05782 | 0.341 | 0.0% |
| trade_cnt_imb_300 | 0.03724 | 0.1513 | -0.282 | 0.01978 | 0.5321 | 0.2% |
| n_trades_300 | 7.844 | 0.8832 | 6.354 | 7.707 | 10.53 | 0.0% |
| volume_300 | 12.03 | 0.9735 | 9.83 | 11.99 | 14.49 | 0.0% |
| n_quote_updates_300 | 7.815 | 1.092 | 5.361 | 7.731 | 10.58 | 0.0% |
| avg_spread_300 | 2.637 | 1.666 | 0.5551 | 2.29 | 7.632 | 0.0% |
| rv_300 | 2.62e-06 | 4.544e-06 | 1.129e-07 | 1.339e-06 | 2.118e-05 | 0.6% |
| ret_600 | 0.000205 | 0.002197 | -0.004846 | 2.423e-05 | 0.006715 | 3.4% |
| ofi_600 | 7.79 | 163.8 | -462.1 | 2.316 | 548.9 | 0.2% |
| trade_imb_600 | -0.05723 | 0.1329 | -0.3955 | -0.05801 | 0.258 | 0.0% |
| trade_cnt_imb_600 | 0.03638 | 0.1401 | -0.2388 | 0.01609 | 0.5123 | 0.1% |
| n_trades_600 | 8.543 | 0.8726 | 7.106 | 8.397 | 11.23 | 0.0% |
| volume_600 | 12.74 | 0.9521 | 10.59 | 12.7 | 15.16 | 0.0% |
| n_quote_updates_600 | 8.515 | 1.08 | 6.091 | 8.426 | 11.22 | 0.0% |
| avg_spread_600 | 2.678 | 1.757 | 0.5555 | 2.316 | 7.686 | 0.0% |
| rv_600 | 5.421e-06 | 9.165e-06 | 4.221e-07 | 2.758e-06 | 4.235e-05 | 0.1% |

## No look-ahead bias: safeguards

Every stage of the pipeline is built so that a prediction at time t uses only information available at t, and each safeguard is verified by an automated test or check:

| Stage | Safeguard | Verified by |
|---|---|---|
| Universe | S&P 500 membership taken as of each date from CRSP (no end-of-sample constituent list, so no survivorship bias); TAQ symbols linked by that day's CUSIP. | `build_universe` per date; members and links logged per day. |
| Cleaning | Filters use only each row's own fields (prices, sizes, condition codes); no filter looks at later data. | `tests/test_clean.py` |
| Trade signing | Lee–Ready uses the last NBBO strictly *before* the trade (as-of join on trade time − 1 ns), so a quote at the same nanosecond, possibly caused by the trade, is never used. | `tests/test_signing.py` (same-timestamp quote case) |
| Features | Every feature at grid time t uses only events with timestamp ≤ t (backward as-of joins); windows are exact (t − w, t]; windows near the open are truncated at 09:30 rather than filled with later data. | Shift test (perturbing all events after t leaves features at t unchanged) and a brute-force recomputation of window features (`tests/test_features_labels.py`). |
| Labels | Target = mid-to-mid return from the last mid ≤ t to the last mid ≤ t + h; never trade prices; labels that would cross 16:00 are dropped. | Label alignment test on a known mid path. |
| Day boundaries | Features and labels use only the same day's 09:30–16:00 events. | Day-boundary test (adding the neighbouring days' and after-close events changes nothing). |
| Harness check | A deliberately leaked feature set (features from t + 30 s) must look absurdly good. | Real-data placebo (`tests/test_placebo_real.py`); leaked R² ≈ 100% vs small R² for correct features. |
| Normalization, splits, tuning (Steps 8–10) | Planned: per-stock scaling and winsorizing fitted on the training window only; walk-forward by whole days (no train/test day overlap); hyper-parameters tuned on a validation day inside the training window; volume weights known as of the prior day. | To be added with Step 8 (assertions on split dates and scaler fit ranges). |

Latest test run: **13 passed in 5.01s**

Real-data placebo for 2025-12-10: pooled OLS of r_30 on all features, trained 09:35–12:45, tested 12:45–15:55, R² vs zero forecast: correct features **-1.52%**, features leaked from t+30s **100.00%**, so the harness detects leakage.
