"""Statistical metrics from ``predictions.parquet`` (§7.3).

Every number is computed for the configured model and for both baselines on identical rows,
so the comparison needs no alignment. Uncertainty is a block bootstrap over sessions, not a
normal approximation: labels overlap, and sessions are the only unit that is close to
independent.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import polars as pl

from hft_forecast.analytics import stats as S
from hft_forecast.core.types import BoolArray, FloatArray, IntArray
from hft_forecast.ml.base import DIRECTION, FWD_LOGRET, MAGNITUDE_BPS, SESSION_ID

CLASSES: Final = (-1, 0, 1)
CLASS_NAMES: Final = ("down", "flat", "up")
RELIABILITY_BINS: Final = 10
BOOTSTRAP_SAMPLES: Final = 1000
BOOTSTRAP_BLOCK: Final = 2.0
TOP_DECILE: Final = 0.9
MIN_SESSIONS: Final = 3
EPS: Final = 1e-12


def evaluate(predictions: pl.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Per-model, per-fold and pooled metrics for a run's predictions."""
    models = sorted(predictions.get_column("model").unique().to_list())
    out: dict[str, Any] = {
        "n_rows": predictions.height // max(len(models), 1),
        "n_sessions": predictions.get_column(SESSION_ID).n_unique(),
        "n_folds": predictions.get_column("fold").n_unique(),
        "models": {},
    }
    for model in models:
        rows = predictions.filter(pl.col("model") == model)
        folds = [
            {"fold": int(index), **_metrics(rows.filter(pl.col("fold") == index))}
            for index in sorted(rows.get_column("fold").unique().to_list())
        ]
        out["models"][model] = {
            "pooled": {**_metrics(rows), **_bootstrap(rows, seed)},
            "folds": folds,
        }
    return out


def _metrics(rows: pl.DataFrame) -> dict[str, Any]:
    probabilities = rows.select("p_down", "p_flat", "p_up").to_numpy()
    truth = rows.get_column(DIRECTION).to_numpy().astype(int)
    y_hat = rows.get_column("y_hat_logret").to_numpy()
    y_true = rows.get_column(FWD_LOGRET).to_numpy()
    sessions = rows.get_column(SESSION_ID).to_numpy()
    magnitude = rows.get_column(MAGNITUDE_BPS).to_numpy()
    top = np.abs(y_hat) >= np.quantile(np.abs(y_hat), TOP_DECILE) if len(y_hat) else y_hat > 0
    return {
        "n_rows": len(truth),
        **_direction_metrics(probabilities, truth),
        **_return_metrics(y_hat, y_true, magnitude, sessions),
        "class_balance": {
            name: float((truth == value).mean())
            for name, value in zip(CLASS_NAMES, CLASSES, strict=True)
        },
        "confusion": _confusion(probabilities, truth),
        "reliability": _reliability(probabilities, truth),
        "top_decile": {
            "n_rows": int(top.sum()),
            **_direction_metrics(probabilities[top], truth[top]),
            **_return_metrics(y_hat[top], y_true[top], magnitude[top], sessions[top]),
        },
    }


def _predicted(probabilities: FloatArray) -> IntArray:
    index: IntArray = probabilities.argmax(axis=1)
    return np.array(CLASSES)[index]


def _direction_metrics(probabilities: FloatArray, truth: IntArray) -> dict[str, float]:
    if not len(truth):
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan")}
    predicted = _predicted(probabilities)
    per_class = [
        float((predicted[truth == c] == c).mean()) if (truth == c).any() else float("nan")
        for c in CLASSES
    ]
    return {
        "accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float(np.nanmean(per_class)),
        "macro_f1": _macro_f1(predicted, truth),
        "auc_up": _auc(probabilities[:, 2], truth == 1),
        "auc_down": _auc(probabilities[:, 0], truth == -1),
        "log_loss": _log_loss(probabilities, truth),
        "brier": _brier(probabilities, truth),
    }


def _return_metrics(
    y_hat: FloatArray, y_true: FloatArray, magnitude: FloatArray, sessions: IntArray
) -> dict[str, float]:
    if not len(y_hat):
        return {"mae_bps": float("nan"), "ic": float("nan")}
    error = y_hat - y_true
    variance = float(np.var(y_true))
    session_ic = _per_session(y_hat, y_true, sessions)
    return {
        "mae_bps": float(np.abs(error).mean() * 1e4),
        "rmse_bps": float(np.sqrt((error**2).mean()) * 1e4),
        "r2_vs_zero": float(1.0 - (error**2).mean() / variance) if variance > EPS else float("nan"),
        "ic": _spearman(y_hat, y_true),
        "ic_t_stat": float(S.mean_t_stat(session_ic)) if len(session_ic) > 1 else float("nan"),
        "ic_sessions": len(session_ic),
        "magnitude_ic": _spearman(np.abs(y_hat), magnitude),
    }


def _per_session(y_hat: FloatArray, y_true: FloatArray, sessions: IntArray) -> FloatArray:
    """Spearman IC within each session (Fama-MacBeth style, respecting the session blocks)."""
    out = [
        _spearman(y_hat[sessions == session], y_true[sessions == session])
        for session in np.unique(sessions)
    ]
    values = np.array([v for v in out if np.isfinite(v)])
    return values


def _bootstrap(rows: pl.DataFrame, seed: int) -> dict[str, Any]:
    """95% intervals from resampling whole sessions.

    Sessions, not rows: six consecutive 5 s labels share five of their six bars, so a row
    bootstrap would report intervals several times too narrow.
    """
    sessions = rows.get_column(SESSION_ID).to_numpy()
    unique = np.unique(sessions)
    if len(unique) < MIN_SESSIONS:
        return {"ci": {}, "ci_note": "too few sessions to bootstrap"}
    probabilities = rows.select("p_down", "p_flat", "p_up").to_numpy()
    truth = rows.get_column(DIRECTION).to_numpy().astype(int)
    y_hat = rows.get_column("y_hat_logret").to_numpy()
    y_true = rows.get_column(FWD_LOGRET).to_numpy()
    correct = (_predicted(probabilities) == truth).astype(float)
    per_session = np.array(
        [
            [correct[sessions == s].mean(), _spearman(y_hat[sessions == s], y_true[sessions == s])]
            for s in unique
        ]
    )
    rng = np.random.default_rng(seed)
    index = S.stationary_bootstrap_indices(len(unique), BOOTSTRAP_SAMPLES, BOOTSTRAP_BLOCK, rng)
    draws = np.nanmean(per_session[index], axis=1)
    return {
        "ci": {
            "accuracy": _interval(draws[:, 0]),
            "ic": _interval(draws[:, 1]),
        },
        "n_bootstrap": BOOTSTRAP_SAMPLES,
    }


def _interval(draws: FloatArray) -> list[float]:
    """95% interval, or NaNs when the statistic is undefined (``zero`` has no IC)."""
    if not np.isfinite(draws).any():
        return [float("nan"), float("nan")]
    return [float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))]


def _confusion(probabilities: FloatArray, truth: IntArray) -> list[list[int]]:
    predicted = _predicted(probabilities)
    return [[int(((truth == a) & (predicted == b)).sum()) for b in CLASSES] for a in CLASSES]


def _reliability(probabilities: FloatArray, truth: IntArray) -> list[dict[str, float]]:
    """Calibration of ``p_up``: predicted versus realized frequency, in ten bins."""
    p_up = probabilities[:, 2]
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    bins = np.clip(np.digitize(p_up, edges[1:-1]), 0, RELIABILITY_BINS - 1)
    out = []
    for b in range(RELIABILITY_BINS):
        where = bins == b
        if not where.any():
            continue
        out.append(
            {
                "bin": float(edges[b]),
                "predicted": float(p_up[where].mean()),
                "observed": float((truth[where] == 1).mean()),
                "n_rows": int(where.sum()),
            }
        )
    return out


def _macro_f1(predicted: IntArray, truth: IntArray) -> float:
    scores = []
    for c in CLASSES:
        true_positive = float(((predicted == c) & (truth == c)).sum())
        precision = true_positive / max((predicted == c).sum(), 1)
        recall = true_positive / max((truth == c).sum(), 1)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def _auc(score: FloatArray, positive: BoolArray) -> float:
    """One-vs-rest AUC via the rank identity (no sklearn round trip)."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _rank(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _log_loss(probabilities: FloatArray, truth: IntArray) -> float:
    index = np.searchsorted(np.array(CLASSES), truth)
    chosen = np.clip(probabilities[np.arange(len(truth)), index], EPS, 1.0)
    return float(-np.log(chosen).mean())


def _brier(probabilities: FloatArray, truth: IntArray) -> float:
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(truth)), np.searchsorted(np.array(CLASSES), truth)] = 1.0
    return float(((probabilities - onehot) ** 2).sum(axis=1).mean())


def _rank(values: FloatArray) -> FloatArray:
    """Average ranks, 1-based (ties share their mean rank)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < len(values):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _spearman(a: FloatArray, b: FloatArray) -> float:
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])
