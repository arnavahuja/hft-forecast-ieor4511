"""Every horizon head of the frozen LSTM ensemble, May to December 2025, plus a 120 s cost backtest.

The models carry 15, 30, 60 and 120 s heads from multi-horizon pretraining; only the 60 s
heads were fine-tuned. This scores all four on the same windows (May validation, June locked
test, July to December on the 22 continuing stocks) and reruns the cost-aware backtest at
120 s, where moves are larger relative to the fixed cost per round trip. Thresholds are
chosen on May, as in backtest_costs.py.

Environment: L2_RAW_DIR is not needed (bars must exist in BARS_DIR, see oos_eval.py),
RESULTS_DIR (checkpoints and experiment_config.json), RESULTS_OUT (default `results`).
"""
import json
import os
import sys
import time
from pathlib import Path

from nbcode import before, src

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backtest_costs as bt  # noqa: E402

BARS = Path(os.environ.get("BARS_DIR", "bars"))
RUN = Path(os.environ.get("RESULTS_DIR", "cn_l2_60s_research/results_60s_multihorizon"))
OUT = Path(os.environ.get("RESULTS_OUT", "results")); OUT.mkdir(parents=True, exist_ok=True)
ALL = [f"2025{m:02d}" for m in range(1, 13)]
EVAL = ALL[4:]

g = {"__name__": "nb"}
exec(src(4), g); exec(src(6), g)
exec(before(src(8), "\n\nfor month in MONTHS"), g)
exec(before(src(10), "grid = full_grid(bars)"), g)
exec(before(src(12), "started = time.perf_counter()"), g)
s16 = src(16); exec(s16[s16.index("@dataclass"):s16.index("samples = make_sequences(")], g)
exec(before(src(18), "N_STOCKS = len(STOCK_VOCAB)"), g)
exec(before(src(20), "STOCK_STD_MAP = {"), g)
pl, np, pd, torch = g["pl"], g["np"], g["pd"], g["torch"]
H = g["PRETRAIN_HORIZONS_S"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

cfg = json.loads((RUN / "experiment_config.json").read_text())
VOCAB, FEATURES = cfg["stock_vocab"], cfg["features"]
std_all = {int(h): v for h, v in cfg["stock_std_bps_by_horizon"].items()}
g.update(STOCK_VOCAB=VOCAB, STOCK_TO_IDX={t: i for i, t in enumerate(VOCAB)}, N_STOCKS=len(VOCAB))
read = lambda m: pl.read_parquet(BARS / f"bars_{m}.parquet")

vol = (pl.concat([g["add_features_labels"](g["full_grid"](read(m)), [60], g["MAX_TRADE_AGE_SECONDS"])
                  .select("ticker", pl.col("log_volume_1m").cast(pl.Float32).cast(pl.Float64)) for m in ALL[:4]])
       .group_by("ticker").agg(pl.col("log_volume_1m").mean().alias("mean"), pl.col("log_volume_1m").std().alias("std"))
       .to_pandas().set_index("ticker"))

fits = []
for seed in cfg["seeds"]:
    ck = torch.load(RUN / f"model_LSTM_seed{seed}.pt", map_location="cpu", weights_only=True)
    mk = lambda st: (lambda m: (m.load_state_dict(st), m.eval(), m)[2])(g["build_model"](ck["kind"], ck["input_dim"], ck["n_stocks"]))
    fits.append({"pooled": mk(ck["pooled_state"]), "per_ticker": {t: mk(st) for t, st in ck["per_ticker_state"].items()},
                 "x_mean": np.asarray(ck["x_mean"], dtype=np.float32), "x_std": np.asarray(ck["x_std"], dtype=np.float32)})


def predict_all(fit, ds):
    """predict_per_ticker from the notebook, returning every horizon head."""
    X = np.clip((ds.X - fit["x_mean"]) / fit["x_std"], -15, 15).astype(np.float32)
    cls = {h: np.zeros((len(X), 3)) for h in H}; reg = {h: np.zeros(len(X)) for h in H}
    for ticker, idx in g["STOCK_TO_IDX"].items():
        mask = ds.stock == idx
        if not mask.any():
            continue
        model = fit["per_ticker"].get(ticker, fit["pooled"])
        with torch.no_grad():
            c, r, _ = model(torch.from_numpy(X[mask]), torch.from_numpy(ds.stock[mask]))
        for h in H:
            cls[h][mask] = torch.softmax(c[h], dim=-1).numpy(); reg[h][mask] = r[h].numpy()
    return cls, reg


def month_frame(m):
    i = ALL.index(m)
    f = g["add_features_labels"](g["full_grid"](pl.concat([read(ALL[i - 1]), read(m)])), H, g["MAX_TRADE_AGE_SECONDS"])
    f = f.filter(pl.col("date").str.starts_with(m) & pl.col("ticker").is_in(VOCAB))
    base = [c for c in FEATURES if c not in g["STOCK_RELATIVE"]]
    keep = (["ticker", "date", "segment", "bucket", "usable", "px", "log_volume_1m"] + [c for c in base if c != "log_volume_1m"]
            + [f"target_{h}s_bps" for h in H] + [f"target_{h}s_valid" for h in H])
    f32 = [c for c in keep if f[c].dtype == pl.Float64 and c != "px"]
    fr = f.select(keep).with_columns(pl.col(f32).cast(pl.Float32)).to_pandas()
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
    return fr


metrics, preds120 = [], {}
for m in EVAL:
    t0 = time.time()
    ds = g["make_sequences"](month_frame(m), FEATURES, g["LOOKBACK_STEPS"], g["STRIDE_STEPS"])
    outs = [predict_all(fit, ds) for fit in fits]
    ses = ds.metadata.session_date.to_numpy()
    for h in H:
        cls = np.mean([o[0][h] for o in outs], axis=0); reg = np.mean([o[1][h] for o in outs], axis=0)
        res = g["evaluate_detailed"](ds.y_bps[h], cls, reg, ds.stock, {t: std_all[h][t] for t in VOCAB}, None, sessions=ses)
        metrics.append({"month": m, "horizon_s": h, **res["pooled"]})
        if h == 120:
            out = ds.metadata.copy(); out["actual_bps"] = ds.y_bps[h]
            out["predicted_bps"] = reg * np.array([std_all[h][VOCAB[s]] for s in ds.stock])
            preds120[m] = out
    row = {r["horizon_s"]: round(r["ic_spearman"], 4) for r in metrics if r["month"] == m}
    log(f"{m}: IC by horizon {row} ({time.time() - t0:.0f}s)")
met = pd.DataFrame(metrics)
met.round(4).to_csv(OUT / "horizon_metrics.csv", index=False)

# Cost backtest at 120 s: thresholds from May, then every later month.
ticks = bt.tick_bps(BARS, EVAL)
loaded = {m: preds120[m].assign(session_date=pd.to_datetime(preds120[m].session_date)).merge(ticks, on=["ticker", "session_date"])
          for m in EVAL}
thr = {q: float(loaded["202505"].predicted_bps.abs().quantile(q)) for q in (0.90, 0.95, 0.99)}
bt_rows = []
for m, df in loaded.items():
    row = {"month": m, **{f"timing_{k}": v for k, v in bt.timing(df).items() if k in ("saving_bps_per_order", "saving_t")}}
    for q, t in thr.items():
        top = f"top{round((1 - q) * 100)}"; tk = bt.round_trips(df, t, 1.0); fe = bt.round_trips(df, t, 0.0)
        row.update({f"{top}_trades": tk["trades"], f"{top}_hit": tk["hit_rate"], f"{top}_gross": tk["gross_bps"],
                    f"{top}_net_taker": tk["net_bps"], f"{top}_net_fees_only": fe["net_bps"]})
    bt_rows.append(row)
btdf = pd.DataFrame(bt_rows); btdf.round(4).to_csv(OUT / "backtest_120s.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(met.pivot(index="month", columns="horizon_s", values="ic_spearman").round(3).to_string())
print(met.pivot(index="month", columns="horizon_s", values="balanced_accuracy").round(3).to_string())
print("May 120 s thresholds (bps):", {k: round(v, 2) for k, v in thr.items()})
print(btdf.round(3).to_string(index=False))
