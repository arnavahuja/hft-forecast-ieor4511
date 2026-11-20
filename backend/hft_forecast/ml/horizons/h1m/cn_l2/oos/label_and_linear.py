"""Label diagnostic and linear baseline for the China L2 60 s study.

Uses the notebook's own feature and label code. Two labels are compared:
  trailing:      log change of the 15 s VWAP from t to t+60 s (starting price includes the
                 trades just before t; the first run's label)
  forward-start: log change from the VWAP of (t, t+15 s] to that of (t+60 s, t+75 s]
                 (the notebook's label)
Rows are sampled every 2 minutes, like the model's input windows, and need a valid label
and finite features. A least-squares linear model on clipped, standardized features is fit
on the training window (January to April) and scored by pooled Spearman IC, with a t-stat
over per-session ICs.

Outputs in RESULTS_OUT (default `results`):
  label_diagnostic.csv   single-feature ICs and linear ICs under both labels (May)
  linear_baseline.csv    linear IC per month, May to December, next to the LSTM ensemble

Environment: BARS_DIR (5 s bar cache, all 12 months), RESULTS_DIR (the notebook's results
with experiment_config.json), OOS_DIR (outputs of oos_eval.py), RESULTS_OUT.
"""
import json
import os
from pathlib import Path

from nbcode import before, src

BARS = Path(os.environ.get("BARS_DIR", "bars"))
RUN = Path(os.environ.get("RESULTS_DIR", "cn_l2_60s_research/results_60s_multihorizon"))
OOS = Path(os.environ.get("OOS_DIR", "out"))
OUT = Path(os.environ.get("RESULTS_OUT", "results")); OUT.mkdir(parents=True, exist_ok=True)
MONTHS = [f"2025{m:02d}" for m in range(1, 13)]
TRAIN, EVAL = MONTHS[:4], MONTHS[4:]

g = {"__name__": "nb"}
exec(src(4), g); exec(src(6), g)
exec(before(src(8), "\n\nfor month in MONTHS"), g)          # session windows (MORNING, AFTERNOON)
exec(before(src(10), "grid = full_grid(bars)"), g)
exec(before(src(12), "started = time.perf_counter()"), g)
pl, np, pd = g["pl"], g["np"], g["pd"]
cfg = json.loads((RUN / "experiment_config.json").read_text())
FEATURES, VOCAB = cfg["features"], cfg["stock_vocab"]
BASE = [c for c in FEATURES if c not in g["STOCK_RELATIVE"]]
K = g["PRIMARY_HORIZON_S"] // g["GRID_SECONDS"]
read = lambda m: pl.read_parquet(BARS / f"bars_{m}.parquet").filter(pl.col("ticker").is_in(VOCAB))


def month(m):
    """Features and both labels for one month, every row (prior month kept for the previous close)."""
    i = MONTHS.index(m)
    bars = pl.concat([read(MONTHS[i - 1]), read(m)]) if i else read(m)
    f = g["add_features_labels"](g["full_grid"](bars), [g["PRIMARY_HORIZON_S"]], g["MAX_TRADE_AGE_SECONDS"])
    seg = g["SEG_KEYS"]; lp = pl.col("px").log()
    f = f.with_columns(
        (1e4 * (lp.shift(-K).over(seg) - lp)).alias("y_trailing"),
        (pl.col("usable") & pl.col("usable").shift(-K).over(seg).fill_null(False)).alias("v_trailing"),
        pl.col("target_60s_bps").alias("y_forward"), pl.col("target_60s_valid").alias("v_forward"))
    return f.filter(pl.col("date").str.starts_with(m))


# Training-window volume statistics (all rows, as in the notebook), then 2-minute samples.
samples, vol_parts = {}, []
for m in MONTHS:
    f = month(m)
    if m in TRAIN:
        vol_parts.append(f.select("ticker", pl.col("log_volume_1m").cast(pl.Float32).cast(pl.Float64)))
    samples[m] = f.filter(pl.col("bucket") % 120 == 0).select(
        ["ticker", "date"] + BASE + ["y_trailing", "v_trailing", "y_forward", "v_forward"]).to_pandas()
    print(m, len(samples[m]), "sampled rows", flush=True)
vol = (pl.concat(vol_parts).group_by("ticker")
       .agg(pl.col("log_volume_1m").mean().alias("mu"), pl.col("log_volume_1m").std().alias("sd")).to_pandas().set_index("ticker"))
for m, d in samples.items():
    d["volume_1m_stock_z"] = (d.log_volume_1m - d.ticker.map(vol.mu)) / d.ticker.map(vol.sd).clip(lower=1e-6)
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan)


def rows(d, label):
    ok = d[f"v_{label}"] & np.isfinite(d[f"y_{label}"]) & np.isfinite(d[FEATURES]).all(axis=1)
    return d.loc[ok]


def ic(df, x, y):
    """Pooled Spearman IC and a t-stat over per-session ICs."""
    r = lambda a, b: np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]
    daily = np.array([r(s[x], s[y]) for _, s in df.groupby("date") if len(s) >= 20])
    return r(df[x], df[y]), daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))


train = {lab: pd.concat([rows(samples[m], lab) for m in TRAIN]) for lab in ("trailing", "forward")}
models = {}
for lab, d in train.items():
    mu, sd = d[FEATURES].mean(), d[FEATURES].std().replace(0, 1)
    X = lambda df: np.column_stack([np.clip((df[FEATURES] - mu) / sd, -5, 5), np.ones(len(df))])
    w = np.linalg.lstsq(X(d), np.clip(d[f"y_{lab}"], -100, 100), rcond=None)[0]
    models[lab] = lambda df, X=X, w=w: X(df) @ w

may = {lab: rows(samples["202505"], lab) for lab in ("trailing", "forward")}
for lab in may:
    may[lab] = may[lab].assign(linear=models[lab](may[lab]))
diag = []
for feat in FEATURES:
    a, _ = ic(may["trailing"], feat, "y_trailing"); b, _ = ic(may["forward"], feat, "y_forward")
    diag.append({"check": f"single feature: {feat}", "trailing_label": a, "forward_start_label": b})
diag = sorted(diag, key=lambda r: -abs(r["trailing_label"]))
lin = {lab: ic(may[lab], "linear", f"y_{lab}")[0] for lab in may}
diag.insert(0, {"check": "linear model fit on Jan to Apr", "trailing_label": lin["trailing"], "forward_start_label": lin["forward"]})
pd.DataFrame(diag).round(4).to_csv(OUT / "label_diagnostic.csv", index=False)

lstm = {"202505": pd.read_csv(RUN / "ensemble_metrics.csv").set_index(["model", "variant"]).loc[("LSTM", "pt"), "ic_spearman"]}
oos = OOS / "monthly_metrics.csv"
if oos.exists():
    lstm.update({str(r.month): r.ic_spearman for r in pd.read_csv(oos).itertuples()})
continuing = set(pd.read_csv(OOS / "predictions_202512.csv").ticker.unique()) if (OOS / "predictions_202512.csv").exists() else set(VOCAB)
out = []
for m in EVAL:
    d = rows(samples[m], "forward")
    if m >= "202507":
        d = d[d.ticker.isin(continuing)]
    d = d.assign(linear=models["forward"](d))
    r, t = ic(d, "linear", "y_forward")
    out.append({"month": m, "stocks": d.ticker.nunique(), "rows": len(d), "linear_ic": r, "linear_t": t,
                "lstm_ensemble_ic": lstm.get(m, np.nan)})
pd.DataFrame(out).round(4).to_csv(OUT / "linear_baseline.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(diag).round(3).head(8).to_string(index=False))
print(pd.DataFrame(out).round(3).to_string(index=False))
