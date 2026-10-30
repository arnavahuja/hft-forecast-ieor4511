# h30s — Phase A baseline: 30-second mid-price prediction from TAQ L1 data

Work of the **30-second horizon group** (IEOR 4511 capstone, sponsored by Occudo Quantitative Strategies).
Phase A builds a leak-free baseline in the spirit of Aït-Sahalia, Fan, Xue & Zhou, *How and When Are
High-Frequency Stock Returns Predictable?* (NBER w30366). We use TAQ level-1 data (NBBO + trades),
hand-crafted order-flow features over multiple lookback windows, and the **mid-price return over the
next 30 seconds** as the target.

This folder is a **standalone pipeline**. It doesn't import the main `hft_forecast` framework, and the
framework's checks (`make check`) don't cover it. It uses its own 1-second grid and point-in-time S&P 500
universe. Porting the features into the framework's `FeatureSet` interface is possible later.

**No data is included.** TAQ/CRSP data is licensed through WRDS and is not redistributed; `data/` is
git-ignored. Everything can be rebuilt with a WRDS account.

## Status

| Step | | Status |
|---|---|---|
| 0–1 | Setup, verify WRDS tables and schemas | done |
| 2 | Point-in-time universe (S&P 500 membership from CRSP, linked to TAQ by CUSIP) | done |
| 3 | Clean NBBO and trades (Holden & Jacobsen-style filters) | done |
| 4 | Lee–Ready trade signing | done |
| 5 | 85 features on a 1-second grid (09:35–15:55) | done |
| 6 | Mid-to-mid labels at 5/10/30/60 s | done |
| 7 | Leakage tests | done (13 tests pass) |
| 8–10 | Walk-forward splits, models (LASSO, LightGBM, MLP), evaluation (R², IC, Rank IC, directional accuracy) | next |

Developed on one full day (2025-12-10, all S&P 500 members); the pilot month is March 2025
(29 large caps + SPY).

## Layout

```
config/pilot.yaml     all parameters: dates, tickers, cleaning rules, windows, horizons
src/hft/              wrds_io (WRDS access), universe, clean, signing, features, labels, leakage, reporting
src/wrds_login.py     WRDS connection helper (credentials in the standard pgpass file, never in code)
scripts/              one CLI per pipeline stage + run_days.py (whole range) + reports
tests/                pytest: unit tests, leakage tests, brute-force feature check, real-data placebo
reports/              DECISIONS.md (every design choice + reason), checkpoint and findings reports, figures
```

## Setup

```bash
cd research/h30s_l1_baseline
pip install -e ".[dev]"          # Python >= 3.11 (developed on 3.14, pandas 3, polars 1.44)
python src/wrds_login.py         # once: prompts for WRDS username/password, saves them to pgpass, tests the connection
pytest                           # 12 synthetic-data tests; the real-data placebo runs once a day has been built
```

## Running the pipeline (one day)

```bash
python scripts/download_day.py --date 2025-03-03 --group pilot --group benchmarks   # raw L1 + trades (WRDS)
python scripts/clean_day.py    --date 2025-03-03                                    # Step 3
python scripts/sign_day.py     --date 2025-03-03                                    # Step 4
python scripts/features_day.py --date 2025-03-03                                    # Steps 5-6
python scripts/panel_report.py --date 2025-03-03                                    # report incl. leakage placebo
# or the whole pilot month, resumable:
python scripts/run_days.py --pilot-range
```

A pilot day is about 190 MB of raw Parquet and takes 8–10 minutes end to end (most of it the WRDS download).

## No look-ahead bias: safeguards

The sponsor requires zero look-ahead bias. Each safeguard is enforced in code and verified by a test:

| Stage | Safeguard | Verified by |
|---|---|---|
| Universe | S&P 500 membership as of each date (CRSP), not an end-of-sample list, so no survivorship bias | per-day universe files |
| Cleaning | Filters use only each row's own fields | `tests/test_clean.py` |
| Trade signing | Lee–Ready uses the last NBBO strictly *before* the trade, so a quote at the same nanosecond is never used | `tests/test_signing.py` |
| Features | Every feature at t uses only events with timestamp ≤ t; windows are exactly (t − w, t]; windows near the open are truncated at 09:30, not filled with later data | Shift test (changing all events after t leaves features at t unchanged) and a brute-force recomputation (`tests/test_features_labels.py`) |
| Labels | Mid-to-mid return from the last mid ≤ t to the last mid ≤ t + h (never trade prices); no label crosses 16:00 | Label-alignment test on a known mid path |
| Day boundaries | Only the same day's 09:30–16:00 events are used | Day-boundary test |
| Harness | Deliberately leaked features (from t + 30 s) must look absurdly good | Real-data placebo: R² = 100% leaked vs −1.5% correct (2025-12-10) |
| Normalization, splits, tuning | Planned for Steps 8–10: scaling fitted on training days only, walk-forward by whole days, tuning inside the training window | to be added with Step 8 |

## First results (one development day, descriptive only)

See `reports/dev_day_findings_2025-12-10.md`. Highlights:

- **Quote imbalance** is the strongest signal: average per-stock rank IC with the next 30 s mid return is
  0.06, positive in 27–28 of 29 stocks, and it decays with horizon (0.098 at 5 s, 0.047 at 60 s).
- **Predictability is concentrated in large-tick stocks:** IC is 0.30 for F and 0.14–0.18 for T, PFE, KMI
  and VZ, versus about 0 for AAPL, MSFT and TSLA.
- Spread and volatility features looked predictive only because of the day's market drift (FOMC rally);
  after removing each stock's market component they largely disappear. Quote imbalance doesn't.

These are single-day, overlapping-label statistics; the walk-forward evaluation on the pilot month comes next.

## Design decisions

All non-obvious choices (cleaning filters, odd lots / ISO / off-exchange trades, OFI normalization,
handling of the open, flat-band labels, and how our setup differs from the NBER paper) are logged with
reasons in [`reports/DECISIONS.md`](reports/DECISIONS.md).
