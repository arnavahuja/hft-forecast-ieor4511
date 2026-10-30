# Decisions log — Phase A baseline

Each entry: the choice, and a one-line reason. Newest sections at the bottom.

## Data access

| Decision | Reason |
|---|---|
| Connect via `wrds_login.connect()` (SQLAlchemy 2 + psycopg2; credentials from pgpass) instead of the official `wrds` package | Current `wrds` releases require pandas < 2.3; the only pandas-3-compatible release (3.1.2) fails with SQLAlchemy 2. Same method names, no credentials in code. |
| One SQL query per day for all needed symbols, filtered by symbol and session on the server | One pass over the day's table; avoids repeated scans. |
| Raw window 09:30:00–16:00:00 (SIP time) | Regular hours; the 09:35–15:55 sample grid and 16:00 label cutoff sit inside it. |
| Store raw data as Parquet (zstd), `data/raw/{nbbo,trades}/date=YYYY-MM-DD/{group}.parquet` | ~5.6 B/row (L1) and ~8.7 B/row (trades) on disk vs ~180 B/row in pandas memory. |
| Keep the full S&P 500 raw day 2025-12-10 locally as the development day | Build and debug cleaning/features on real data before downloading the pilot month. |

## WRDS data facts (Step 1 / CHECKPOINT 1)

**Libraries.** Millisecond TAQ is in yearly libraries `taqm_2003` … `taqm_2026` (also `taqmsec`, which holds all years). Each has one table per trading day:

| Table | Content | Use |
|---|---|---|
| `complete_nbbo_YYYYMMDD` | WRDS complete NBBO (one row per NBBO change) | **source for the mid** |
| `ctm_YYYYMMDD` | consolidated trades | trades |
| `cqm_YYYYMMDD`, `nbbom_YYYYMMDD` | all exchange quotes / raw NBBO messages | reference only |
| `mastm_YYYYMMDD` | daily master file (`symbol_15`, `cusip`, `cta_symbol`, `sec_type`, `test_symbol_flag`, …) | symbol ↔ CUSIP |
| `wrds_iid_YYYY` | WRDS daily intraday indicators per stock (spreads, volumes, open/close) | cross-checks |

The daily tables have no indexes.

**`complete_nbbo` columns used:** `date` (date), `time_m` (time, µs), `time_m_nano` (smallint), `sym_root`, `sym_suffix` (text), `qu_cond`, `natbbo_ind`, `nbbo_qu_cond` (varchar), `best_bid`, `best_ask` (numeric → cast to float8), `best_bidsizeshares`, `best_asksizeshares` (integer, shares). Not downloaded: `qu_source`.

**`ctm` columns used:** `date`, `time_m`, `time_m_nano`, `ex`, `sym_root`, `sym_suffix`, `tr_scond` (sale conditions), `size` (int), `price` (numeric → float8), `tr_corr` (correction indicator), `tr_seqnum` (bigint), `tr_source`, `tr_rf`. Not downloaded: `tr_stop_ind`, `tr_id`, `part_time(_nano)`, `trf_time(_nano)`, `tte_ind`.

**Timestamp:** `ts = date + time_m` (SIP time, microsecond precision), plus `time_m_nano` for nanoseconds. Participant (`part_time`) and TRF (`trf_time`) timestamps are not used.

**AAPL on 2025-12-10 (09:30–16:00):** 355,025 L1 rows, 502,024 trades (21.7M shares); last trade $278.72, last mid $278.75.

**Dev day 2025-12-10, all 503 members, 09:30–16:00:**
- L1: 37.0M rows, 207 MB, 6.2 min download. Trades: 38.7M rows, 335 MB, 5.4 min. SPY: 6.1M L1 rows (31 MB), 1.1M trades (8.5 MB).
- All L1 quotes ≥ $1 are on the 1-cent grid (no half-penny ticks) → tick size $0.01.
- Crossed quotes 0.02% of rows, locked 2.7%, no zero bid/ask; no duplicate (symbol, ts, nano) rows.
- `qu_cond`: `R` 37,008,440, `O` 285. `natbbo_ind`: `U` 13.9M, `2` 13.5M, `4` 5.0M, `G` 4.5M, `3` 0.1M. `nbbo_qu_cond` is always null.
- `tr_corr`: `00` 38,736,437; `08` 588; `10` 67; `01` 6; `12` 4.
- `tr_scond` (top codes): `I` 9.2M, `@  I` 9.2M, null 4.1M, `F I` 4.0M, `@` 3.7M, `@F I` 3.4M, `F` 1.6M, `@F` 1.5M, `4 I` 0.8M, `@4 I` 0.6M, `4 B` 0.26M, `@4 W` 0.18M, `B` 33k, `@  W` 15k, `7 V`/`@7 V` 8k, `Q` 1k, `T`/`@ T` 1k, `O` 340, `C  I` 381. (`I` = odd lot, `F` = ISO, `4` = derivatively priced, `W` = average price, `T` = extended hours, `Q`/`O` = opening prints.) Filters are decided in Step 3.

## Universe and symbol linking

| Decision | Reason |
|---|---|
| Membership from CRSP `crsp_a_indexes.dsp500list_v2`, as of each date (`mbrstartdt <= date <= mbrenddt`) | Point-in-time membership, no survivorship bias. Covers dates through 2025-12-31; 2026 would need another source. |
| Link PERMNO → TAQ symbol by CUSIP: CRSP `stksecurityinfohist.cusip9` (valid that day) = TAQ `mastm.cusip`; `symbol_15` = root + space + suffix | The WRDS link table `wrdsapps.taqmclink` ends 2024-12-31 and leaves out REITs and foreign-incorporated members (68 of 503 on 2024-10-03, e.g. ACN, CB, AON, AMT). CUSIP linking matched all 503 members on 2025-03-03 and 2025-12-10. |
| Fallback to CRSP ticker + share class only when the CUSIP is missing (logged as `link = "ticker"`) | Safety net; not needed so far. |
| Pilot tickers written in CTA format (`BRK.B`, `GOOGL`) and mapped via `mastm.cta_symbol` | TAQ splits some symbols into root + suffix: `BRK B`, `GOOG L` (= GOOGL), `CMCS A` (= CMCSA). Mapping through the master file avoids guessing. All 29 pilot tickers map and are members on both checked dates. |
| SPY pulled as a separate `benchmarks` group | Not an S&P 500 member and not a prediction target; needed later for cross-asset work. |

## Data scope

| Decision | Reason |
|---|---|
| Keep raw L1 + trades locally for the pilot (29 names + SPY, 2025-03-03 … 2025-03-31) | ≈190 MB/day, ≈4 GB in total; lets feature definitions change without re-downloading. Pilot names are ~30% of S&P 500 L1 and trade rows. |
| No multi-year or full-universe multi-day downloads without Jeff's go-ahead | Five years of full-universe raw data is ~0.3–0.7 TB. |

## Step 2 — Universe

| Decision | Reason |
|---|---|
| `data/universe/date=YYYY-MM-DD/universe.parquet` holds the pilot targets that are S&P 500 members that day (`role = target`) plus SPY (`role = benchmark`); `symbol` = CTA ticker | One file per day drives cleaning and features. Membership is checked per date, so a stock that leaves the index drops out from that day on. |
| Built for 2025-12-10 and 2025-03-03: 503 members, 29/29 targets, SPY found, no ticker fallbacks | — |

## Step 3 — Cleaning (config `clean:` in `config/pilot.yaml`)

**NBBO** (`src/hft/clean.py: clean_nbbo`), applied in this order; each dropped row gets one reason:

| Rule | Reason |
|---|---|
| Keep `qu_cond` ∈ {A, B, H, O, R, W} | Holden & Jacobsen (2014) normal quote conditions. On the dev day only `R` and 16 `O` rows (just after 09:30) occur. |
| Drop missing or ≤ 0 bid/ask | Not a valid quote (none on the dev day). |
| Drop crossed (bid > ask) | Invalid NBBO state. |
| Drop locked (bid = ask) from the series | Spread 0 gives a degenerate mid and microprice. The previous valid quote stays in force until the next valid update. |
| Drop spread > 1000 bps | Sanity cap only; the widest pilot spread on the dev day was 202 bps, so it never triggers. |
| No filter on `natbbo_ind` | `complete_nbbo` is already WRDS's NBBO series. Crossed and locked rows occur only with `natbbo_ind` ∈ {U, 2}, and the rules above remove them. |
| Drop rows identical to the previous kept row (bid, ask, both sizes) | Repeated states add no information and would inflate quote-update counts. |
| Keep size-only updates (same prices, new sizes) | 53% of AAPL L1 rows; needed for quote imbalance and OFI. `time_since_last_mid_change` must ignore them. |
| `ts` = SIP time (`date + time_m`) + `time_m_nano`, in ns; no ties per symbol on the dev day | One consistent clock for quotes and trades. |

**Trades** (`clean_trades`):

| Rule | Reason |
|---|---|
| Keep `tr_corr = '00'` | Uncorrected trades only (Holden & Jacobsen 2014; WRDS ITD code). |
| Drop price or size ≤ 0 or missing | Invalid (none on the dev day). |
| Keep a trade only if every character of `tr_scond` is in {`@`, `F`, `I`, space}, or `tr_scond` is null | Allowlist = regular-way trades, including ISO (`F`) and odd lots (`I`). This drops opening/reopening/closing and auction prints (`O`, `Q`, `5`, `6`, `M`), extended-hours/late (`T`, `U`), out-of-sequence (`Z`, `L`), average-price (`W`), derivatively priced (`4`), qualified contingent (`7`, `V`), cash/next-day/seller (`C`, `N`, `R`), bunched (`B`), cross (`X`) and prior-reference (`P`) prints, which don't reflect the current market. |
| **Keep odd lots** (`I`), flag `is_oddlot` | About two-thirds of pilot prints (7.6M of 11.5M on the dev day), though a small share of volume. Since 2015 odd lots carry price information (O'Hara, Yao & Ye 2014), especially in high-priced stocks; dropping them would lose most trade events. |
| **Keep ISO trades** (`F`), flag `is_iso` | Intermarket sweep orders are aggressive, informed liquidity-taking trades, so they matter for imbalance features. |
| **Keep off-exchange trades** (FINRA TRF, `ex = 'D'`), flag `is_trf` | 49% of pilot trades and volume. The flag lets features be computed with or without them. Caveat: TRF prints are reported to the SIP with a delay (up to 10 s allowed), and the participant timestamp (`part_time`) is not downloaded. |

**Dev day 2025-12-10 result (29 targets + SPY):** NBBO 17.40M → 16.85M kept (locked 487k, duplicate 62k, crossed 4k). Trades 11.45M → 11.00M kept (sale condition 454k, corrected 125). AAPL keeps 99.4% of quotes and 95.6% of trades; F keeps 96.7% (3.2% locked) and 88.5% (mostly derivatively priced `4 I`/`4 B` prints). Plots and tables: `reports/clean_sanity_2025-12-10.md`.

**Note:** the dev day is an FOMC day (statement 14:00, press conference 14:30); AAPL's spread widens from 14:00 and spikes to 40 cents around 14:40. The pilot month includes FOMC on 2025-03-19.

## Step 4 — Trade signing (`src/hft/signing.py`)

| Decision | Reason |
|---|---|
| Prevailing quote = last clean NBBO with `ts` strictly before the trade (as-of join on `trade ts − 1 ns`) | No look-ahead: a quote stamped at the same nanosecond may have been caused by the trade. |
| Above mid → +1, below → −1; at mid (\|price − mid\| < $10⁻⁶) or no prior quote → tick test | Lee & Ready (1991). The tolerance only absorbs float error; prices and mids sit on a half-cent grid. |
| Tick test over all clean trades of the symbol (exchange and TRF): sign of the change from the previous different price; 0 if there is none | Standard zero-tick rule. |
| Kept as columns: `pre_mid`, `sign`, `sign_method` (quote / tick / none) | Lets features use trade location and filter by signing method. |

**Dev day result (29 targets + SPY, 11.0M trades):** 78.6% signed by quote, 21.4% by tick test, 56 unsigned. Exchange trades: 48.1% buy / 51.9% sell, 18% at the mid. TRF trades: 60.7% buy / 39.3% sell, 25% at the mid. The buy skew of off-exchange trades may be genuine (retail flow) or come from TRF reporting delays; Step 5 can compute imbalance with and without TRF trades to check.

## Reference: Aït-Sahalia, Fan, Xue & Zhou (NBER w30366) setup, verified from the paper (2026-10-30)

| Item | Paper | Our plan |
|---|---|---|
| Data / universe | TAQ 2019–2020; 101 S&P 100 constituents **as of 2020-12-31** (chosen at sample end; GOOGL dropped, GOOG kept) | point-in-time S&P 500 membership; pilot 29 names, March 2025 |
| Session | strictly 09:30:00–16:00:00; no pre-open data | grid 09:35–15:55 |
| Cleaning | only regular hours, Lee–Ready signing, odd lots included; no other filters described | Holden–Jacobsen-style filters (Step 3) |
| Prediction times | every trade/quote event timestamp T (summary stats use 1,000 sampled per stock-day) | 1-second grid |
| **Target** | **transaction return**: average trade price in (T, T+Δ] divided by the mid at T, minus 1 (Eq. 2.4); footnote 5 says mid-to-mid can be studied the same way | mid-to-mid return (sponsor requirement) |
| Lookback windows | 9 **disjoint** windows (0, .1], (.1, .2], (.2, .4], …, (12.8, 25.6] seconds; also transaction-clock (0–1 … 128–256 trades) and volume-clock (0–100 … 12,800–25,600 shares) versions | cumulative windows 1–600 s |
| Predictors | 13 × 9 windows = 117 per clock: Breadth, Immediacy, VolumeAll, VolumeAvg, VolumeMax, Lambda, LobImbalance, TxnImbalance (Lee–Ready), PastReturn, Turnover (volume / shares outstanding), AutoCov, QuotedSpread, EffectiveSpread | own set (OFI, microprice, RV, …) |
| Models | LASSO and random forest (main); horse race with OLS, ridge, FarmPredict, GBT | Ridge/LASSO, LightGBM, MLP |
| Training | fit on previous 5 days, predict next day; hyper-parameters re-tuned every 20 days (two-layer rolling window); **training target clipped at 5th/95th percentile** | 5-day walk-forward; features winsorized 0.5/99.5 |
| R² benchmark | **out-of-sample mean of the test data** (Eq. 3.3) | zero forecast |
| Direction accuracy | sign of (prediction − training mean) vs sign of (actual − training mean) | sign accuracy with flat band |
| Headline results | median R² ≈ 10% (5 s) and ≈ 4% (30 s) for transaction returns; ~64% direction accuracy; higher for low-priced stocks; INTC 5 s R² falls from 14% to 2.5% with a 10 ms delay | — |

The suggested replication repo (fanyiyang/…) implements the paper on Chinese A-share (Shenzhen) tick data with its own matching engine. It is not TAQ-based, so its code isn't directly reusable.

**Jeff's decision (2026-10-30): keep the original Phase A plan settings.** That means the planned feature set (cumulative 1–600 s windows), the mid-to-mid target only, the 1-second grid, a zero-forecast R² benchmark, feature winsorizing at 0.5/99.5%, and Ridge/LASSO/LightGBM/MLP. Consequence: our R² is **not directly comparable** to the paper's (different target, sampling times and benchmark). The comparison paragraph in the final report must say so. The paper's features, event-time sampling and transaction-return target remain possible later extensions.

## Step 5 — Features (`src/hft/features.py`)

| Decision | Reason |
|---|---|
| Windows near the open are **truncated at 09:30** (option a): only events from 09:30 on count, and `ret_w` uses the first mid of the day when t − w is before it | Matches the paper (strictly 09:30–16:00, no pre-open data) and keeps the full sample. |
| Add `secs_since_open` | Lets models tell when a long window is truncated near the open. |
| Method: running totals per event stream + backward as-of join to each grid second (events with ts ≤ t); window (t−w, t] = cum[t] − cum[t−w] | Exact half-open windows, verified against a brute-force loop over raw events (`tests/test_features_labels.py`). |
| `ofi_w` = sum of Cont–Kukanov–Stoikov L1 order-flow contributions of the clean NBBO updates in the window ÷ their average L1 depth ((bid_size + ask_size)/2); 0 if no updates | CKS normalize OFI by average depth. The first update of the day contributes 0. |
| `trade_imb_w`, `trade_cnt_imb_w` = 0 when the window has no trades | Neutral value; `n_trades_w` = 0 tells the model the window was empty. |
| Trade features use all clean signed trades (exchange and TRF, odd lots, ISO) | Original settings; the `is_trf` flag remains available for a with/without comparison. |
| `n_quote_updates_w` counts size-only updates; `time_since_last_mid_change` ignores them | Size changes are information; mid changes are a different event. |
| `avg_spread_w` = average of the 1-second `spread_bps` samples in the window (time-weighted at 1 s resolution) | Simple and leak-free. Note: `avg_spread_1` equals `spread_bps` by construction (redundant; harmless for regularized models). |
| `rv_w` = sum of squared 1-second log mid changes in the window (w ≥ 10) | Realized variance; seconds before the first quote contribute 0. |
| `time_since_last_trade` / `_mid_change`: measured from 09:30 if no event yet | Avoids missing values right after the open. |
| `n_trades`, `volume`, `n_quote_updates` as log1p | Heavy-tailed counts. |

## Step 6 — Labels (`src/hft/labels.py`)

| Decision | Reason |
|---|---|
| `mid_{t+h}` = last clean mid with ts ≤ t+h (a quote stamped exactly at t+h counts) | As specified; tested on a synthetic path with jumps placed 1 µs on either side of t+h. |
| Δmid in ticks rounded to 4 decimals; `r_h` = rounded Δmid / mid_t | Makes unchanged mids exactly zero despite float arithmetic. |
| `dir_h` = 0 if \|Δmid\| < half the spread at t, otherwise sign(Δmid); `sign_h` = pure sign | Flat band as specified; a move of exactly half the spread counts as a move. |
| Labels with t + h after 16:00 are null; with the 15:55 sample end and h ≤ 60 s, none occur | — |

## Step 7 — Leakage tests (all passing; `pytest`, 13 tests)

1. Shift test: perturbing every quote and trade after 12:00 (and adding trades) leaves all features at t ≤ 12:00 unchanged, and changes later ones.
2. Label alignment on a known mid path, plus the flat-band rule and the 16:00 cutoff.
3. Day boundary: adding the previous and next trading days' events, plus events after 16:00, changes no feature or label.
4. Real-data placebo (dev day): pooled OLS of r_30, trained 09:35–12:45 and tested 12:45–15:55. Correct features give R² −1.5%; features shifted 30 s into the future give R² 100.0%.
5. A brute-force check of the window features against loops over raw events.

**Dev-day panel:** 684,030 rows (30 symbols × 22,801 s), 85 features, no missing values. Report: `reports/checkpoint2_2025-12-10.md`.
