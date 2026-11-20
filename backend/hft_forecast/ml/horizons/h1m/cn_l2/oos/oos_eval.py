"""Out-of-sample check of the frozen June models on July to December 2025.

Runs the committed notebook's own code (bar building, features, labels, sequences, model
classes, prediction and metrics), loads the saved LSTM checkpoints (the ensemble selected
on May), and scores every later month without retraining. June is recomputed first and
compared with the notebook's saved June predictions to prove the pipeline is identical.
"""
import json
import os
import time
from pathlib import Path

from nbcode import before, src

RAW = Path(os.environ.get("L2_RAW_DIR", "raw"))                      # monthly l2tick_2025MM.csv.gz
RUN = Path(os.environ.get("RESULTS_DIR", "cn_l2_60s_research/results_60s_multihorizon"))  # checkpoints
OUT = Path("out"); OUT.mkdir(exist_ok=True)
ALL_MONTHS = [f"2025{m:02d}" for m in range(1, 13)]
EVAL_MONTHS = ALL_MONTHS[5:]           # June (reproduction check) then July to December

g = {"__name__": "nb"}
exec(src(4), g)                          # imports
exec(src(6), g)                          # configuration (DEVICE, horizons, dead zone, ...)
exec(before(src(8), "\n\nfor month in MONTHS"), g)
exec(before(src(10), "grid = full_grid(bars)"), g)
exec(before(src(12), "started = time.perf_counter()"), g)
s16 = src(16); exec(s16[s16.index("@dataclass"):s16.index("samples = make_sequences(")], g)
exec(before(src(18), "N_STOCKS = len(STOCK_VOCAB)"), g)
exec(before(src(20), "STOCK_STD_MAP = {"), g)
pl, np, pd, torch = g["pl"], g["np"], g["pd"], g["torch"]
H = g["PRETRAIN_HORIZONS_S"]; PRIMARY = g["PRIMARY_HORIZON_S"]

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---- 1. Bars for all twelve months (cached) ----
bars_dir = Path(os.environ.get("BARS_DIR", "bars")); bars_dir.mkdir(exist_ok=True)
for m in ALL_MONTHS:
    out = bars_dir / f"bars_{m}.parquet"
    if not out.exists():
        t0 = time.time(); nb, nt = g["build_month_bars"](RAW / f"l2tick_{m}.csv.gz", out)
        log(f"bars {m}: {nt:,} trades -> {nb:,} bars in {time.time() - t0:.0f}s")
read = lambda m: pl.read_parquet(bars_dir / f"bars_{m}.parquet")

# ---- 2. Training-window statistics the notebook computed on Jan to Apr ----
cfg = json.loads((RUN / "experiment_config.json").read_text())
STOCK_VOCAB = cfg["stock_vocab"]
FEATURE_COLUMNS = cfg["features"]
assert FEATURE_COLUMNS == g["FEATURE_COLUMNS"]
stock_std_all = {int(h): v for h, v in cfg["stock_std_bps_by_horizon"].items()}
g.update(STOCK_VOCAB=STOCK_VOCAB, STOCK_TO_IDX={t: i for i, t in enumerate(STOCK_VOCAB)},
         N_STOCKS=len(STOCK_VOCAB),
         STOCK_STD_MAP={t: stock_std_all[PRIMARY][t] for t in STOCK_VOCAB})
parts = []
for m in ALL_MONTHS[:4]:
    f = g["add_features_labels"](g["full_grid"](read(m)), [PRIMARY], g["MAX_TRADE_AGE_SECONDS"])
    parts.append(f.select("ticker", pl.col("log_volume_1m").cast(pl.Float32).cast(pl.Float64)))
vol_stats = (pl.concat(parts).group_by("ticker")
             .agg(pl.col("log_volume_1m").mean().alias("mean"), pl.col("log_volume_1m").std().alias("std"))
             .to_pandas().set_index("ticker"))
log("volume z-score statistics from Jan to Apr:", len(vol_stats), "stocks")

# ---- 3. Frozen models: the May-selected LSTM per-ticker seed ensemble ----
fits = []
for seed in cfg["seeds"]:
    ck = torch.load(RUN / f"model_LSTM_seed{seed}.pt", map_location="cpu", weights_only=True)
    pooled = g["build_model"](ck["kind"], ck["input_dim"], ck["n_stocks"]); pooled.load_state_dict(ck["pooled_state"])
    per = {}
    for t, st in ck["per_ticker_state"].items():
        mdl = g["build_model"](ck["kind"], ck["input_dim"], ck["n_stocks"]); mdl.load_state_dict(st); per[t] = mdl
    fits.append({"pooled": pooled, "per_ticker": per,
                 "x_mean": np.asarray(ck["x_mean"], dtype=np.float32),
                 "x_std": np.asarray(ck["x_std"], dtype=np.float32)})
log("loaded", len(fits), "LSTM checkpoints")

def month_frame(m):
    prev = ALL_MONTHS[ALL_MONTHS.index(m) - 1]
    bars = pl.concat([read(prev), read(m)])                    # prior month gives the previous close
    f = g["add_features_labels"](g["full_grid"](bars), H, g["MAX_TRADE_AGE_SECONDS"])
    f = f.filter(pl.col("date").str.starts_with(m))
    base = [c for c in FEATURE_COLUMNS if c not in g["STOCK_RELATIVE"]]
    keep = (["ticker", "date", "segment", "bucket", "usable", "px", "log_volume_1m"]
            + [c for c in base if c != "log_volume_1m"]
            + [f"target_{h}s_bps" for h in H] + [f"target_{h}s_valid" for h in H])
    f32 = [c for c in keep if f[c].dtype == pl.Float64 and c != "px"]
    fr = f.select(keep).with_columns(pl.col(f32).cast(pl.Float32)).to_pandas()
    fr["session_date"] = pd.to_datetime(fr["date"], format="%Y%m%d")
    fr["timestamp"] = fr["session_date"] + pd.to_timedelta(fr["bucket"], unit="s")
    mu = fr.ticker.map(vol_stats["mean"]).astype("float64")
    sd = fr.ticker.map(vol_stats["std"]).astype("float64").fillna(1.0).clip(lower=1e-6)
    fr["volume_1m_stock_z"] = ((fr["log_volume_1m"] - mu) / sd).astype("float32")
    fr[FEATURE_COLUMNS] = fr[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    for h in H:
        fr[f"target_{h}s_std"] = (fr[f"target_{h}s_bps"] / fr.ticker.map(stock_std_all[h])).astype("float32")
        y = fr[f"target_{h}s_bps"].to_numpy()
        cls = np.full(len(fr), 1, dtype=np.int64); cls[y > g["DEAD_ZONE_BPS"]] = 2; cls[y < -g["DEAD_ZONE_BPS"]] = 0
        fr[f"target_{h}s_cls"] = cls
    fr = fr[fr.ticker.isin(STOCK_VOCAB)]
    fr["stock_idx"] = fr.ticker.map(g["STOCK_TO_IDX"]).astype(np.int64)
    return fr

rows = []
for m in EVAL_MONTHS:
    t0 = time.time()
    fr = month_frame(m)
    ds = g["make_sequences"](fr, FEATURE_COLUMNS, g["LOOKBACK_STEPS"], g["STRIDE_STEPS"])
    del fr
    preds = [g["predict_per_ticker"](fit, ds) for fit in fits]
    cls = np.mean([p[0] for p in preds], axis=0); reg = np.mean([p[1] for p in preds], axis=0)
    y = ds.y_bps[PRIMARY]; ses = ds.metadata.session_date.to_numpy()
    res = g["evaluate_detailed"](y, cls, reg, ds.stock, g["STOCK_STD_MAP"], None, sessions=ses)
    out = ds.metadata.copy()
    out["actual_bps"] = y
    out["predicted_bps"] = reg * np.array([g["STOCK_STD_MAP"][STOCK_VOCAB[s]] for s in ds.stock])
    out["p_down"], out["p_flat"], out["p_up"] = cls[:, 0], cls[:, 1], cls[:, 2]
    out.to_csv(OUT / f"predictions_{m}.csv", index=False)
    pd.DataFrame(res["per_stock"]).T.to_csv(OUT / f"per_stock_{m}.csv")
    rows.append({"month": m, **res["pooled"]})
    log(f"{m}: n={len(y):,} ic={res['pooled']['ic_spearman']:.4f} t={res['pooled']['ic_t_stat']:.1f} "
        f"bal_acc={res['pooled']['balanced_accuracy']:.3f} ({time.time() - t0:.0f}s)")
    if m == "202506":                                           # reproduction check
        saved = pd.read_csv(RUN / "FINAL_TEST_predictions.csv", parse_dates=["timestamp"])
        j = saved.merge(out, on=["ticker", "timestamp"], suffixes=("_saved", "_new"))
        diff = np.abs(j.predicted_bps_saved - j.predicted_bps_new)
        corr = np.corrcoef(j.predicted_bps_saved, j.predicted_bps_new)[0, 1]
        log(f"June reproduction: {len(j):,} of {len(saved):,} windows matched, max |diff| {diff.max():.2e} bps, "
            f"median {np.median(diff):.2e}, corr {corr:.8f} (GPU vs CPU float rounding)")
        assert len(j) == len(saved) and diff.max() < 0.05 and corr > 0.99999, \
            "pipeline does not reproduce the saved June predictions"
pd.DataFrame(rows).to_csv(OUT / "monthly_metrics.csv", index=False)
log("done")
