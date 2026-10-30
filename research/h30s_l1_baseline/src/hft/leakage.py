"""Step 7.4: real-data placebo. A deliberately leaked feature set must look absurdly good."""

import numpy as np
import polars as pl


def _ols_r2_oos(train: np.ndarray, test: np.ndarray) -> float:
    """OLS on standardized features (train stats), R² on test vs a zero forecast."""
    Xtr, ytr, Xte, yte = train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    A = lambda X: np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(A(Xtr), ytr, rcond=None)
    pred = A(Xte) @ beta
    return float(1 - np.sum((yte - pred) ** 2) / np.sum(yte ** 2))


def placebo(features: pl.DataFrame, labels: pl.DataFrame, horizon: int = 30, split: str = "12:45:00") -> dict:
    """R²_OOS of r_h on (a) correct features at t, (b) features from t + h (leaked).

    Pooled OLS over target stocks; train = morning, test = afternoon of the same day.
    """
    feat_cols = [c for c, t in features.schema.items() if t.is_numeric()]
    y = f"r_{horizon}"
    df = (features.filter(pl.col("role") == "target")
          .join(labels.select("symbol", "t", y), on=["symbol", "t"])
          .sort("symbol", "t"))
    leaked = df.with_columns(pl.col(c).shift(-horizon).over("symbol") for c in feat_cols)
    out = {}
    for name, d in [("correct", df), ("leaked_+%ds" % horizon, leaked)]:
        d = d.drop_nulls([y, *feat_cols]).filter(pl.all_horizontal(pl.col(feat_cols).is_finite()))
        morning = d["t"].dt.time() < pl.Series([split]).str.to_time()[0]
        arr = d.select(y, *feat_cols).to_numpy()
        out[name] = _ols_r2_oos(arr[morning.to_numpy()], arr[~morning.to_numpy()])
    return out
