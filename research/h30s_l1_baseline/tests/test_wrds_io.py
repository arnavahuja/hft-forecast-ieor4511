import pandas as pd

from hft.wrds_io import raw_path, split_symbol, taq_table


def test_split_symbol_root_and_suffix():
    out = split_symbol(pd.Series(["AAPL", "BRK B", "GOOG L", "CMCS A", " SPY "]))
    assert out["sym_root"].tolist() == ["AAPL", "BRK", "GOOG", "CMCS", "SPY"]
    assert out["sym_suffix"].tolist() == ["", "B", "L", "A", ""]


def test_table_and_path_names():
    assert taq_table("complete_nbbo", "2025-03-03") == "taqm_2025.complete_nbbo_20250303"
    p = raw_path("data", "nbbo", "2025-03-03", "pilot")
    assert p.as_posix() == "data/raw/nbbo/date=2025-03-03/pilot.parquet"
