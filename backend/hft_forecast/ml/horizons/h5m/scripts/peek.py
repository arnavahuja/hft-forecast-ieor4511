"""Print a few real rows of every data layer for one stock (read-only)."""
import os
import polars as pl

pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)
ROOT = r"E:\4511_5min"
TICKER = "AAPL"

quotes = pl.read_parquet(rf"{ROOT}\drive_upload\taq_nbbo_bars\{TICKER}.parquet")
trades = pl.read_parquet(rf"{ROOT}\drive_upload\taq_trade_bars\{TICKER}.parquet")
print("QUOTE BARS", quotes.shape)
print(quotes.schema)
print(quotes.slice(60, 3))           # 60 bars after the open = 10:31 New York
print("\nTRADE BARS", trades.shape)
print(trades.slice(60, 3))

feats = pl.read_parquet(rf"{os.environ['HFTF_DATA_DIR']}\features\be3b07f87bf6d3f0\{TICKER}.parquet")
print("\nMODEL DATASET", feats.shape)
print([c for c in feats.columns if c.startswith("f_")])
cols = ["timestamp", "mid", "f_imbalance", "f_logret_5", "fwd_logret", "direction", "label_valid"]
sample = feats.select(cols).slice(60, 8)
print(sample)

# check the label by hand: fwd_logret at row i equals log(mid[i+5]) - log(mid[i])
m = feats["mid"].to_list()
i = 60
print("\nhand check:", (pl.Series([m[i + 5]]).log() - pl.Series([m[i]]).log())[0],
      "vs stored", feats["fwd_logret"][i])
print("rows per session:", feats.group_by("session_id").len()["len"].unique().to_list())