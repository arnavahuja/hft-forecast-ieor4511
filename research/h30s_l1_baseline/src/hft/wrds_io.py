"""WRDS access: S&P 500 membership, TAQ symbol mapping, and raw TAQ L1 / trade downloads.

Raw files are written one day at a time, one file per symbol group:
    data/raw/nbbo/date=2025-12-10/sp500.parquet     every L1 update (taqm complete_nbbo)
    data/raw/trades/date=2025-12-10/sp500.parquet   every trade (taqm ctm)
    data/raw/nbbo/date=2025-12-10/SPY.parquet       ...

Notes (details in reports/DECISIONS.md):
- One query per day for all requested symbols, filtered in SQL by symbol and session.
- Membership: CRSP crsp_a_indexes.dsp500list_v2. TAQ symbols are linked by 9-digit CUSIP
  through the TAQ daily master file (mastm), because wrdsapps.taqmclink ends 2024-12-31
  and leaves out REITs / foreign-incorporated members.
- `ts` is the SIP timestamp (date + time_m, microseconds); `time_m_nano` adds nanoseconds.
"""

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMAS = {
    "nbbo": pa.schema([
        ("ts", pa.timestamp("us")), ("time_m_nano", pa.int16()),
        ("sym_root", pa.string()), ("sym_suffix", pa.string()),
        ("qu_cond", pa.string()), ("natbbo_ind", pa.string()), ("nbbo_qu_cond", pa.string()),
        ("best_bid", pa.float64()), ("best_bidsizeshares", pa.int32()),
        ("best_ask", pa.float64()), ("best_asksizeshares", pa.int32()),
    ]),
    "trades": pa.schema([
        ("ts", pa.timestamp("us")), ("time_m_nano", pa.int16()), ("ex", pa.string()),
        ("sym_root", pa.string()), ("sym_suffix", pa.string()),
        ("tr_scond", pa.string()), ("size", pa.int32()), ("price", pa.float64()),
        ("tr_corr", pa.string()), ("tr_seqnum", pa.int64()), ("tr_source", pa.string()), ("tr_rf", pa.string()),
    ]),
}

_SELECT = {
    "nbbo": """t.date + t.time_m AS ts, t.time_m_nano, t.sym_root, s.sym_suffix,
               t.qu_cond, t.natbbo_ind, t.nbbo_qu_cond,
               t.best_bid::float8 AS best_bid, t.best_bidsizeshares,
               t.best_ask::float8 AS best_ask, t.best_asksizeshares""",
    "trades": """t.date + t.time_m AS ts, t.time_m_nano, t.ex, t.sym_root, s.sym_suffix,
                 t.tr_scond, t.size, t.price::float8 AS price, t.tr_corr, t.tr_seqnum, t.tr_source, t.tr_rf""",
}
_SOURCE = {"nbbo": "complete_nbbo", "trades": "ctm"}


def taq_table(kind: str, date) -> str:
    d = pd.Timestamp(date)
    return f"taqm_{d.year}.{kind}_{d:%Y%m%d}"


def is_trading_day(db, date) -> bool:
    q = "SELECT to_regclass(%(t)s) IS NOT NULL AS ok"
    return bool(db.raw_sql(q, params={"t": taq_table("complete_nbbo", date)})["ok"].iloc[0])


def split_symbol(symbol_15: pd.Series) -> pd.DataFrame:
    """TAQ master 'BRK B' -> sym_root 'BRK', sym_suffix 'B' ('' when there is no suffix)."""
    parts = symbol_15.str.strip().str.split(" ", n=1, expand=True).reindex(columns=[0, 1])
    return pd.DataFrame({"sym_root": parts[0], "sym_suffix": parts[1].fillna("")}, index=symbol_15.index)


def taq_symbols(db, date, tickers) -> pd.DataFrame:
    """Map CTA tickers ('BRK.B', 'GOOGL', 'SPY') to TAQ sym_root/sym_suffix and CUSIP on `date`.

    Tickers not found in that day's master file are returned with null symbols.
    """
    found = db.raw_sql(
        f"""SELECT cta_symbol AS ticker, symbol_15, cusip, sec_desc
            FROM {taq_table('mastm', date)}
            WHERE cta_symbol = ANY(%(tk)s) AND coalesce(test_symbol_flag, '') <> 'Y'""",
        params={"tk": list(tickers)},
    )
    out = pd.DataFrame({"ticker": list(tickers)}).merge(found, on="ticker", how="left")
    out = pd.concat([out, split_symbol(out["symbol_15"].fillna(""))], axis=1)
    out["sym_root"] = out["sym_root"].mask(out["sym_root"] == "")
    return out


def sp500_members(db, date) -> pd.DataFrame:
    """S&P 500 members on `date` (CRSP) with their TAQ sym_root / sym_suffix (linked by CUSIP)."""
    d = pd.Timestamp(date)
    df = db.raw_sql(f"""
        WITH m AS (
            SELECT s.permno, h.cusip9, h.ticker, h.shareclass, h.issuernm
            FROM crsp_a_indexes.dsp500list_v2 s
            JOIN crsp_a_stock.stksecurityinfohist h
              ON h.permno = s.permno AND %(d)s BETWEEN h.secinfostartdt AND h.secinfoenddt
            WHERE %(d)s BETWEEN s.mbrstartdt AND s.mbrenddt
        ), t AS (
            SELECT DISTINCT ON (cusip) cusip, symbol_15, cta_symbol
            FROM {taq_table('mastm', d)}
            WHERE coalesce(test_symbol_flag, '') <> 'Y'
            ORDER BY cusip, symbol_15
        )
        SELECT m.*, t.symbol_15, t.cta_symbol FROM m LEFT JOIN t ON t.cusip = m.cusip9
        ORDER BY m.ticker""", params={"d": d.date()})
    # fall back to CRSP ticker (+ share class) only if the CUSIP is not in the master file
    fallback = (df["ticker"] + " " + df["shareclass"].fillna("")).str.strip()
    df["link"] = df["symbol_15"].isna().map({True: "ticker", False: "cusip"})
    df = pd.concat([df, split_symbol(df["symbol_15"].fillna(fallback))], axis=1)
    df.insert(0, "date", d.date())
    return df


def raw_path(data_dir, kind: str, date, name: str) -> Path:
    return Path(data_dir) / "raw" / kind / f"date={pd.Timestamp(date):%Y-%m-%d}" / f"{name}.parquet"


def _download(db, kind, date, symbols, out_path, session, chunksize):
    sql = f"""
        WITH s(sym_root, sym_suffix) AS (SELECT * FROM unnest(%(roots)s::text[], %(sufs)s::text[]))
        SELECT {_SELECT[kind]}
        FROM {taq_table(_SOURCE[kind], date)} t
        JOIN s ON t.sym_root = s.sym_root AND coalesce(t.sym_suffix, '') = s.sym_suffix
        {"WHERE t.time_m BETWEEN %(t0)s AND %(t1)s" if session else ""}"""
    params = {"roots": symbols["sym_root"].tolist(), "sufs": symbols["sym_suffix"].tolist()}
    if session:
        params |= {"t0": session[0], "t1": session[1]}

    schema, rows = SCHEMAS[kind], 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for chunk in db.raw_sql(sql, params=params, chunksize=chunksize, dtype_backend="pyarrow"):
            w.write_table(pa.Table.from_pandas(chunk, preserve_index=False).select(schema.names).cast(schema))
            rows += len(chunk)
    tmp.replace(out_path)  # only complete files get the final name
    return rows


def download_day(db, date, symbols: pd.DataFrame, name: str, data_dir="data",
                 datasets=("nbbo", "trades"), session=("09:30:00", "16:00:00"),
                 chunksize=2_000_000, overwrite=False, verbose=True) -> pd.DataFrame:
    """Download raw TAQ data for one day and a set of symbols (DataFrame with sym_root, sym_suffix).

    Writes data/raw/{dataset}/date=YYYY-MM-DD/{name}.parquet. Existing files are skipped unless
    overwrite=True, so re-running resumes. session=None downloads the full day incl. pre/post market.
    """
    symbols = symbols.dropna(subset=["sym_root"])[["sym_root", "sym_suffix"]].drop_duplicates()
    summary = []
    for kind in datasets:
        path = raw_path(data_dir, kind, date, name)
        if path.exists() and not overwrite:
            summary.append({"dataset": kind, "status": "exists", "rows": pq.read_metadata(path).num_rows,
                            "seconds": 0.0, "mb": round(path.stat().st_size / 1e6, 1)})
            continue
        t0 = time.time()
        rows = _download(db, kind, date, symbols, path, session, chunksize)
        summary.append({"dataset": kind, "status": "downloaded", "rows": rows,
                        "seconds": round(time.time() - t0, 1), "mb": round(path.stat().st_size / 1e6, 1)})
        if verbose:
            s = summary[-1]
            print(f"  {pd.Timestamp(date):%Y-%m-%d} {name:8s} {kind:7s} {rows:>12,} rows {s['seconds']:7.1f}s {s['mb']:8.1f} MB")
    return pd.DataFrame(summary).assign(date=pd.Timestamp(date).date(), name=name)
