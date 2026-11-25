"""Permutation feature importance of the frozen LSTM ensemble on the June 2025 locked test (60 s).

For each feature (and each feature group), the feature's whole 60-bar input history is
shuffled across windows of the same stock, which breaks its link with the label while
keeping its distribution and each stock's level. The drop in pooled Spearman IC is the
feature's importance. Groups are permuted jointly, which also captures features that
substitute for one another.

Environment: BARS_DIR (bars for May and June), RESULTS_DIR (checkpoints and
experiment_config.json), RESULTS_OUT (default `results`).
"""
import json
import os
import time
from pathlib import Path

from nbcode import before, src

BARS = Path(os.environ.get("BARS_DIR", "bars"))
RUN = Path(os.environ.get("RESULTS_DIR", "cn_l2_60s_research/results_60s_multihorizon"))
OUT = Path(os.environ.get("RESULTS_OUT", "results")); OUT.mkdir(parents=True, exist_ok=True)
ALL = [f"2025{m:02d}" for m in range(1, 13)]
MONTH = "202506"

g = {"__name__": "nb"}
exec(src(4), g); exec(src(6), g)
exec(before(src(8), "\n\nfor month in MONTHS"), g)
exec(before(src(10), "grid = full_grid(bars)"), g)
exec(before(src(12), "started = time.perf_counter()"), g)
s16 = src(16); exec(s16[s16.index("@dataclass"):s16.index("samples = make_sequences(")], g)
exec(before(src(18), "N_STOCKS = len(STOCK_VOCAB)"), g)
exec(before(src(20), "STOCK_STD_MAP = {"), g)
pl, np, pd, torch = g["pl"], g["np"], g["pd"], g["torch"]
H = g["PRETRAIN_HORIZONS_S"]; P = g["PRIMARY_HORIZON_S"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

cfg = json.loads((RUN / "experiment_config.json").read_text())
VOCAB, FEATURES = cfg["stock_vocab"], cfg["features"]
std_all = {int(h): v for h, v in cfg["stock_std_bps_by_horizon"].items()}
g.update(STOCK_VOCAB=VOCAB, STOCK_TO_IDX={t: i for i, t in enumerate(VOCAB)}, N_STOCKS=len(VOCAB))
GROUPS = {"price and volatility": g["TRADE_FEATURES"], "order flow": g["FLOW_FEATURES"],
          "activity": g["ACTIVITY_FEATURES"] + g["STOCK_RELATIVE"], "context": g["CONTEXT_FEATURES"]}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES)
read = lambda m: pl.read_parquet(BARS / f"bars_{m}.parquet")

vol = (pl.concat([g["add_features_labels"](g["full_grid"](read(m)), [P], g["MAX_TRADE_AGE_SECONDS"])
                  .select("ticker", pl.col("log_volume_1m").cast(pl.Float32).cast(pl.Float64)) for m in ALL[:4]])
       .group_by("ticker").agg(pl.col("log_volume_1m").mean().alias("mean"), pl.col("log_volume_1m").std().alias("std"))
       .to_pandas().set_index("ticker"))

fits = []
for seed in cfg["seeds"]:
    ck = torch.load(RUN / f"model_LSTM_seed{seed}.pt", map_location="cpu", weights_only=True)
    mk = lambda st: (lambda m: (m.load_state_dict(st), m.eval(), m)[2])(g["build_model"](ck["kind"], ck["input_dim"], ck["n_stocks"]))
    fits.append({"pooled": mk(ck["pooled_state"]), "per_ticker": {t: mk(st) for t, st in ck["per_ticker_state"].items()},
                 "x_mean": np.asarray(ck["x_mean"], dtype=np.float32), "x_std": np.asarray(ck["x_std"], dtype=np.float32)})

# June windows, built exactly as in the notebook.
i = ALL.index(MONTH)
f = g["add_features_labels"](g["full_grid"](pl.concat([read(ALL[i - 1]), read(MONTH)])), H, g["MAX_TRADE_AGE_SECONDS"])
f = f.filter(pl.col("date").str.starts_with(MONTH))
base = [c for c in FEATURES if c not in g["STOCK_RELATIVE"]]
keep = (["ticker", "date", "segment", "bucket", "usable", "px", "log_volume_1m"] + [c for c in base if c != "log_volume_1m"]
        + [f"target_{h}s_bps" for h in H] + [f"target_{h}s_valid" for h in H])
f32 = [c for c in keep if f[c].dtype == pl.Float64 and c != "px"]
fr = f.select(keep).with_columns(pl.col(f32).cast(pl.Float32)).to_pandas(); del f
fr["session_date"] = pd.to_datetime(fr["date"], format="%Y%m%d")
fr["timestamp"] = fr["session_date"] + pd.to_timedelta(fr["bucket"], unit="s")
fr["volume_1m_stock_z"] = ((fr.log_volume_1m - fr.ticker.map(vol["mean"]).astype("float64"))
                           / fr.ticker.map(vol["std"]).astype("float64").clip(lower=1e-6)).astype("float32")
fr[FEATURES] = fr[FEATURES].replace([np.inf, -np.inf], np.nan)
for h in H:
    fr[f"target_{h}s_std"] = (fr[f"target_{h}s_bps"] / fr.ticker.map(std_all[h])).astype("float32")
    y = fr[f"target_{h}s_bps"].to_numpy()
    c = np.full(len(fr), 1, dtype=np.int64); c[y > g["DEAD_ZONE_BPS"]] = 2; c[y < -g["DEAD_ZONE_BPS"]] = 0
    fr[f"target_{h}s_cls"] = c
fr["stock_idx"] = fr.ticker.map(g["STOCK_TO_IDX"]).astype(np.int64)
ds = g["make_sequences"](fr, FEATURES, g["LOOKBACK_STEPS"], g["STRIDE_STEPS"]); del fr
y = pd.Series(ds.y_bps[P]).rank().to_numpy()
log(f"June windows: {len(y):,}")


def ensemble_ic(X):
    d = g["SequenceData"](X=X, y_bps=ds.y_bps, y_std=ds.y_std, y_cls=ds.y_cls, stock=ds.stock, metadata=ds.metadata)
    reg = np.mean([g["predict_per_ticker"](fit, d)[1] for fit in fits], axis=0)
    pred = reg * np.array([std_all[P][VOCAB[s]] for s in ds.stock])
    return float(np.corrcoef(pd.Series(pred).rank().to_numpy(), y)[0, 1])


base_ic = ensemble_ic(ds.X)
log(f"baseline IC {base_ic:.4f}")
rng = np.random.default_rng(0)
by_stock = [np.flatnonzero(ds.stock == s) for s in np.unique(ds.stock)]
perm = np.empty(len(ds.stock), dtype=np.int64)
for idx in by_stock:
    perm[idx] = rng.permutation(idx)                      # one within-stock shuffle, reused for every feature

rows = []
for name, cols in [(c, [c]) for c in FEATURES] + [(f"group: {k}", v) for k, v in GROUPS.items()]:
    j = [FEATURES.index(c) for c in cols]
    X = ds.X.copy(); X[:, :, j] = ds.X[perm][:, :, j]
    ic = ensemble_ic(X); del X
    group = next(k for k, v in GROUPS.items() if cols[0] in v)
    rows.append({"feature": name, "group": group if not name.startswith("group:") else "", "ic_permuted": ic,
                 "ic_drop": base_ic - ic, "share_of_ic": (base_ic - ic) / base_ic})
    log(f"{name:32s} IC {ic:.4f} drop {base_ic - ic:+.4f}")
res = pd.DataFrame(rows)
res.insert(0, "baseline_ic", base_ic)
res = pd.concat([res[~res.feature.str.startswith("group:")].sort_values("ic_drop", ascending=False),
                 res[res.feature.str.startswith("group:")].sort_values("ic_drop", ascending=False)])
res.round(4).to_csv(OUT / "feature_importance_june.csv", index=False)
print(res.round(4).to_string(index=False))
