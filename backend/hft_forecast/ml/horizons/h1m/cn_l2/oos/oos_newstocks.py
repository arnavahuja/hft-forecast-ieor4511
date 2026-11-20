"""Cold-start test: the frozen model on the 28 stocks that entered the universe on 1 July 2025.

These stocks were never seen in training. The pooled LSTM models (3 seeds) are used with the
average of the 50 trained stock embeddings. July is a burn-in month used only for each new
stock's volume normalisation and 60 s return scale; August to December are scored.
"""
import json
import os
import time
from pathlib import Path

from nbcode import before, src

RUN = Path(os.environ.get("RESULTS_DIR", "cn_l2_60s_research/results_60s_multihorizon"))
OUT = Path("out")
MONTHS = [f"2025{m:02d}" for m in range(7, 13)]

g = {"__name__": "nb"}
exec(src(4), g); exec(src(6), g)
exec(before(src(8), "\n\nfor month in MONTHS"), g)
exec(before(src(10), "grid = full_grid(bars)"), g)
exec(before(src(12), "started = time.perf_counter()"), g)
s16 = src(16); exec(s16[s16.index("@dataclass"):s16.index("samples = make_sequences(")], g)
exec(before(src(18), "N_STOCKS = len(STOCK_VOCAB)"), g)
exec(before(src(20), "STOCK_STD_MAP = {"), g)
pl, np, pd, torch = g["pl"], g["np"], g["pd"], g["torch"]
H = g["PRETRAIN_HORIZONS_S"]; PRIMARY = g["PRIMARY_HORIZON_S"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

cfg = json.loads((RUN / "experiment_config.json").read_text())
FEATURES = cfg["features"]
read = lambda m: pl.read_parquet(Path(os.environ.get("BARS_DIR", "bars")) / f"bars_{m}.parquet")
new = sorted(set(read("202507")["ticker"].unique().to_list()) - set(cfg["stock_vocab"]))
log(f"{len(new)} new stocks")

def features(m):
    prev = f"2025{int(m[4:]) - 1:02d}"
    bars = pl.concat([read(prev), read(m)]).filter(pl.col("ticker").is_in(new))
    f = g["add_features_labels"](g["full_grid"](bars), H, g["MAX_TRADE_AGE_SECONDS"])
    return f.filter(pl.col("date").str.starts_with(m))

# Burn-in month: per-stock scales from July only.
jul = features("202507")
vol = (jul.select("ticker", pl.col("log_volume_1m").cast(pl.Float32).cast(pl.Float64)).group_by("ticker")
       .agg(pl.col("log_volume_1m").mean().alias("mean"), pl.col("log_volume_1m").std().alias("std"))
       .to_pandas().set_index("ticker"))
std_h = {h: (jul.filter(pl.col(f"target_{h}s_valid")).group_by("ticker")
             .agg(pl.col(f"target_{h}s_bps").std()).to_pandas().set_index("ticker")[f"target_{h}s_bps"]
             .clip(lower=1e-3).to_dict()) for h in H}
del jul

VOCAB = new
g.update(STOCK_VOCAB=VOCAB, STOCK_TO_IDX={t: i for i, t in enumerate(VOCAB)}, N_STOCKS=len(VOCAB),
         STOCK_STD_MAP={t: std_h[PRIMARY][t] for t in VOCAB})

fits = []
for seed in cfg["seeds"]:
    ck = torch.load(RUN / f"model_LSTM_seed{seed}.pt", map_location="cpu", weights_only=True)
    state = dict(ck["pooled_state"])
    emb = state["stock_emb.weight"]
    state["stock_emb.weight"] = torch.cat([emb, emb.mean(dim=0, keepdim=True)])   # index 50 = average stock
    model = g["build_model"](ck["kind"], ck["input_dim"], ck["n_stocks"] + 1); model.load_state_dict(state)
    fits.append({"pooled": model, "x_mean": np.asarray(ck["x_mean"], dtype=np.float32),
                 "x_std": np.asarray(ck["x_std"], dtype=np.float32)})
AVG = fits[0]["pooled"].stock_emb.num_embeddings - 1

rows = []
for m in MONTHS[1:]:
    t0 = time.time()
    f = features(m)
    base = [c for c in FEATURES if c not in g["STOCK_RELATIVE"]]
    keep = (["ticker", "date", "segment", "bucket", "usable", "px", "log_volume_1m"]
            + [c for c in base if c != "log_volume_1m"]
            + [f"target_{h}s_bps" for h in H] + [f"target_{h}s_valid" for h in H])
    f32 = [c for c in keep if f[c].dtype == pl.Float64 and c != "px"]
    fr = f.select(keep).with_columns(pl.col(f32).cast(pl.Float32)).to_pandas(); del f
    fr["session_date"] = pd.to_datetime(fr["date"], format="%Y%m%d")
    fr["timestamp"] = fr["session_date"] + pd.to_timedelta(fr["bucket"], unit="s")
    mu = fr.ticker.map(vol["mean"]).astype("float64")
    sd = fr.ticker.map(vol["std"]).astype("float64").fillna(1.0).clip(lower=1e-6)
    fr["volume_1m_stock_z"] = ((fr["log_volume_1m"] - mu) / sd).astype("float32")
    fr[FEATURES] = fr[FEATURES].replace([np.inf, -np.inf], np.nan)
    for h in H:
        fr[f"target_{h}s_std"] = (fr[f"target_{h}s_bps"] / fr.ticker.map(std_h[h])).astype("float32")
        y = fr[f"target_{h}s_bps"].to_numpy()
        c = np.full(len(fr), 1, dtype=np.int64); c[y > g["DEAD_ZONE_BPS"]] = 2; c[y < -g["DEAD_ZONE_BPS"]] = 0
        fr[f"target_{h}s_cls"] = c
    fr["stock_idx"] = fr.ticker.map(g["STOCK_TO_IDX"]).astype(np.int64)
    ds = g["make_sequences"](fr, FEATURES, g["LOOKBACK_STEPS"], g["STRIDE_STEPS"]); del fr
    as_avg = g["SequenceData"](X=ds.X, y_bps=ds.y_bps, y_std=ds.y_std, y_cls=ds.y_cls,
                               stock=np.full(len(ds.stock), AVG, dtype=np.int64), metadata=ds.metadata)
    preds = [g["predict_pooled"](fit, as_avg) for fit in fits]
    cls = np.mean([p[0] for p in preds], axis=0); reg = np.mean([p[1] for p in preds], axis=0)
    y = ds.y_bps[PRIMARY]
    res = g["evaluate_detailed"](y, cls, reg, ds.stock, g["STOCK_STD_MAP"], None,
                                 sessions=ds.metadata.session_date.to_numpy())
    out = ds.metadata.copy(); out["actual_bps"] = y
    out["predicted_bps"] = reg * np.array([g["STOCK_STD_MAP"][VOCAB[s]] for s in ds.stock])
    out["p_down"], out["p_flat"], out["p_up"] = cls[:, 0], cls[:, 1], cls[:, 2]
    out.to_csv(OUT / f"predictions_new_{m}.csv", index=False)
    pd.DataFrame(res["per_stock"]).T.to_csv(OUT / f"per_stock_new_{m}.csv")
    rows.append({"month": m, **res["pooled"]})
    log(f"{m} new stocks: n={len(y):,} ic={res['pooled']['ic_spearman']:.4f} t={res['pooled']['ic_t_stat']:.1f} "
        f"bal_acc={res['pooled']['balanced_accuracy']:.3f} ({time.time() - t0:.0f}s)")
pd.DataFrame(rows).to_csv(OUT / "monthly_metrics_new_stocks.csv", index=False)
log("done")
