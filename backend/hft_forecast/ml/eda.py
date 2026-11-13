"""Feature research: does a feature lead the label, and is that lead stable?

What this answers, before any model is fitted: how much usable data each name contributes,
which features correlate with the forward return and how reliably across stock-days, whether
the relationship is monotone or only shows up in the tails, how fast it decays as the
horizon lengthens, which features are duplicates of each other, and which are missing.

Three properties make it shared infrastructure rather than one group's notebook:

* It runs on a **built dataset**, so it sees whatever feature sets and horizon the config
  asked for. A group adds a feature set and its research appears here with no code change.
* Every statistic is computed **per instrument-session block** and then aggregated across
  blocks, so one volatile name or one wild day cannot carry a ranking. The block count is
  reported next to every number.
* It reads the **training sessions only** by default. Ranking features on sessions that
  later serve as test folds is how a feature study leaks, and the leak does not show up in
  the model's own cross-validation.

The forward returns here are deliberately computed from future bars: this is measurement of
a label, not a feature. Nothing in this module feeds a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import (
    BPS,
    DIRECTION,
    FWD_LOGRET,
    LABEL_VALID,
    SESSION_ID,
    HorizonSpec,
    require_columns,
)
from hft_forecast.ml.features.registry import feature_columns

FEATURE: Final = "feature"
BUCKET: Final = "bucket"
HORIZON_SECONDS: Final = "horizon_seconds"
SPEARMAN: Final = "spearman"
PEARSON: Final = "pearson"
BLOCK: Final = (C.INSTRUMENT, SESSION_ID)
EPS: Final = 1e-12

DEFAULT_MIN_BLOCK_ROWS: Final = 50
DEFAULT_BUCKETS: Final = 5
DEFAULT_DECAY_MULTIPLES: Final = (0.25, 0.5, 1.0, 2.0, 4.0)
NEEDS: Final = (C.TIMESTAMP, C.INSTRUMENT, SESSION_ID, C.MID, FWD_LOGRET, LABEL_VALID)


class EdaParams(BaseModel):
    """Knobs of the study. Defaults follow the group's feature-research notebook."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_sessions: int | None = Field(
        None,
        ge=1,
        description="Use only the first n sessions; None uses every session in the dataset",
    )
    min_block_rows: int = Field(
        DEFAULT_MIN_BLOCK_ROWS,
        ge=2,
        description="Stock-days with fewer usable rows than this are skipped, not imputed",
    )
    buckets: int = Field(
        DEFAULT_BUCKETS, ge=2, description="Rank buckets of the monotonicity profile"
    )
    decay_multiples: list[float] = Field(
        default_factory=lambda: list(DEFAULT_DECAY_MULTIPLES),
        description="Horizons to re-measure the IC at, as multiples of the configured one",
    )


@dataclass(frozen=True)
class EdaReport:
    """Everything the study computed, as tables ready to serialize.

    Attributes:
        horizon: Name of the horizon the labels were built for.
        horizon_seconds: That horizon in seconds.
        bar_seconds: Grid bar length in seconds.
        summary: Scalars for the header (row counts, class balance, strongest feature).
        coverage: One row per instrument: sessions, bars and usable shares.
        ranking: One row per feature: mean IC across blocks, its t-statistic and spread.
        buckets: Mean forward return per rank bucket per feature.
        decay: IC of every feature re-measured at several horizons.
        correlation: Pairwise feature correlation, long form.
        missing: Share of rows each feature is missing, per instrument.
        blocks: The per-block ICs the ranking aggregates (kept for export).
    """

    horizon: str
    horizon_seconds: int
    bar_seconds: int
    summary: dict[str, Any]
    coverage: pl.DataFrame
    ranking: pl.DataFrame
    buckets: pl.DataFrame
    decay: pl.DataFrame
    correlation: pl.DataFrame
    missing: pl.DataFrame
    blocks: pl.DataFrame

    def tables(self) -> dict[str, list[dict[str, Any]]]:
        """The frames as JSON-ready records, keyed by table name."""
        return {
            "coverage": self.coverage.to_dicts(),
            "ranking": self.ranking.to_dicts(),
            "buckets": self.buckets.to_dicts(),
            "decay": self.decay.to_dicts(),
            "correlation": self.correlation.to_dicts(),
            "missing": self.missing.to_dicts(),
        }


def train_slice(frame: pl.DataFrame, n_sessions: int | None) -> pl.DataFrame:
    """The first ``n_sessions`` sessions of a dataset, or all of it when ``None``.

    The cut is in whole sessions and the same for every instrument, which is what makes the
    slice reproducible when the instrument list changes.
    """
    if n_sessions is None:
        return frame
    sessions = frame.get_column(SESSION_ID).unique().sort().head(n_sessions)
    if sessions.is_empty():
        return frame
    # the cut is a session id, so a comparison is enough and does not care about gaps
    return frame.filter(pl.col(SESSION_ID) <= sessions[-1])


def coverage(frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Per-instrument data quality: sessions, bars, and the share that is usable.

    ``model_ready_share`` is the share of bars with a valid label *and* every feature
    present, which is the number that decides how much data a model actually trains on. It
    is always lower than ``label_valid_share``, and the gap is the cost of the feature
    windows.
    """
    ready = pl.col(LABEL_VALID) & pl.all_horizontal(
        [_present(f) for f in features] or [pl.lit(value=True)]
    )
    aggregates = [
        pl.col(SESSION_ID).n_unique().alias("sessions"),
        pl.len().alias("bars"),
        pl.col(LABEL_VALID).mean().alias("label_valid_share"),
        ready.mean().alias("model_ready_share"),
    ]
    if C.SPREAD_BPS in frame.columns:
        aggregates.append(pl.col(C.SPREAD_BPS).median().alias("median_spread_bps"))
    if C.N_QUOTES in frame.columns:
        aggregates.append(pl.col(C.N_QUOTES).mean().alias("mean_quotes_per_bar"))
    if C.IS_STALE in frame.columns:
        aggregates.append(pl.col(C.IS_STALE).mean().alias("stale_share"))
    return frame.group_by(C.INSTRUMENT).agg(aggregates).sort(C.INSTRUMENT)


def block_ic(
    frame: pl.DataFrame,
    features: list[str],
    target: str = FWD_LOGRET,
    min_block_rows: int = DEFAULT_MIN_BLOCK_ROWS,
) -> pl.DataFrame:
    """Rank and linear correlation of every feature with ``target``, per stock-day.

    One correlation per instrument-session rather than one pooled over everything: pooling
    would let a single high-volatility name dominate, and would hide that a feature works on
    two days out of forty.
    """
    frames = []
    for name in features:
        mask = _present(name) & pl.col(target).is_not_null()
        column, label = pl.col(name).filter(mask), pl.col(target).filter(mask)
        frames.append(
            frame.group_by(BLOCK)
            .agg(
                mask.sum().alias("n"),
                pl.corr(column, label, method="spearman").alias(SPEARMAN),
                pl.corr(column, label).alias(PEARSON),
            )
            .filter(pl.col("n") >= min_block_rows)
            # a block where the feature never moves correlates to NaN, not null
            .filter(pl.col(SPEARMAN).is_not_null() & pl.col(SPEARMAN).is_not_nan())
            .with_columns(pl.lit(name).alias(FEATURE))
        )
    if not frames:
        return pl.DataFrame(schema=_ic_schema())
    return (
        pl.concat(frames)
        .select(FEATURE, C.INSTRUMENT, SESSION_ID, "n", SPEARMAN, PEARSON)
        .sort(FEATURE, C.INSTRUMENT, SESSION_ID)
    )


def ic_ranking(ic: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-block ICs into one row per feature, strongest first.

    The t-statistic is over the *blocks*, so it answers "is this feature's edge consistent
    across stock-days", not "is it significant in a pooled regression" — the second question
    has a sample size in the millions and says yes to almost anything. Blocks are not
    independent (names co-move, and a day is shared), so read it as a ranking device.
    """
    if ic.is_empty():
        return pl.DataFrame(schema=_ranking_schema())
    return (
        ic.group_by(FEATURE)
        .agg(
            pl.len().alias("blocks"),
            pl.col("n").sum().alias("rows"),
            pl.col(SPEARMAN).mean().alias("mean_ic"),
            pl.col(SPEARMAN).std().alias("std_ic"),
            pl.col(SPEARMAN).median().alias("median_ic"),
            (pl.col(SPEARMAN) > 0).mean().alias("share_positive"),
            pl.col(PEARSON).mean().alias("mean_pearson"),
        )
        .with_columns(
            (pl.col("mean_ic") * pl.col("blocks").sqrt() / (pl.col("std_ic") + EPS)).alias("t_stat")
        )
        .sort(pl.col("mean_ic").abs(), descending=True, nulls_last=True)
    )


def bucket_profile(
    frame: pl.DataFrame,
    features: list[str],
    target: str = FWD_LOGRET,
    buckets: int = DEFAULT_BUCKETS,
    min_block_rows: int = DEFAULT_MIN_BLOCK_ROWS,
) -> pl.DataFrame:
    """Mean forward return per rank bucket of each feature, in basis points.

    A correlation of 0.02 can come from a clean monotone tilt or from one extreme bucket, and
    those call for different models. Ranks are taken **inside each stock-day**, so a bucket
    means "high for this name today", not "high across the panel" — and the buckets are a
    diagnostic only: a whole-day rank is not knowable at prediction time and must never
    become a feature.
    """
    frames = []
    for name in features:
        usable = frame.filter(_present(name) & pl.col(target).is_not_null())
        usable = usable.filter(pl.len().over(BLOCK) >= min_block_rows)
        if usable.is_empty():
            continue
        index = ((pl.col(name).rank("ordinal").over(BLOCK) - 1) * buckets).cast(pl.Int64)
        bucket = (index // pl.len().over(BLOCK)).clip(0, buckets - 1)
        per_block = (
            usable.with_columns(bucket.alias(BUCKET))
            .group_by(*BLOCK, BUCKET)
            .agg((pl.col(target).mean() * BPS).alias("mean_bps"), pl.len().alias("n"))
        )
        frames.append(
            per_block.group_by(BUCKET)
            .agg(
                pl.col("mean_bps").median().alias("median_block_mean_bps"),
                pl.col("mean_bps").mean().alias("mean_block_mean_bps"),
                pl.len().alias("blocks"),
                pl.col("n").sum().alias("rows"),
            )
            .with_columns(pl.lit(name).alias(FEATURE))
        )
    if not frames:
        return pl.DataFrame(schema=_bucket_schema())
    return (
        pl.concat(frames)
        .select(FEATURE, BUCKET, "median_block_mean_bps", "mean_block_mean_bps", "blocks", "rows")
        .sort(FEATURE, BUCKET)
    )


def return_decay(
    frame: pl.DataFrame,
    features: list[str],
    steps: list[int],
    bar_seconds: int,
    min_block_rows: int = DEFAULT_MIN_BLOCK_ROWS,
) -> pl.DataFrame:
    """The same ICs re-measured against forward returns at several horizons.

    The one chart that tells a group whether it is working on the right horizon at all.
    Microstructure signals decay fast: a feature with a strong IC at one bar and nothing at
    twenty is not a weak feature, it is a feature for a different group.
    """
    group = list(BLOCK)
    log_mid = pl.col(C.MID).log()
    frames = []
    for k in steps:
        target = f"_fwd_{k}"
        shifted = frame.with_columns((log_mid.shift(-k).over(group) - log_mid).alias(target))
        ranked = ic_ranking(
            block_ic(shifted, features, target=target, min_block_rows=min_block_rows)
        )
        if ranked.is_empty():
            continue
        frames.append(
            ranked.select(FEATURE, "blocks", "mean_ic", "t_stat").with_columns(
                pl.lit(k * bar_seconds).alias(HORIZON_SECONDS), pl.lit(k).alias("steps")
            )
        )
    if not frames:
        return pl.DataFrame(schema=_decay_schema())
    return (
        pl.concat(frames)
        .select(FEATURE, HORIZON_SECONDS, "steps", "blocks", "mean_ic", "t_stat")
        .sort(FEATURE, HORIZON_SECONDS)
    )


def feature_correlation(frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Pairwise correlation between features, long form.

    Two features at 0.95 are one feature with two names; a linear model will split the
    coefficient between them and look unstable fold to fold for no good reason.

    Constant features are left out: their correlation is undefined, and a row of blanks in
    the heatmap says less than their absence from it plus a zero variance in the ranking.
    """
    varying = _varying(frame, features)
    if not varying:
        return pl.DataFrame(schema=_correlation_schema())
    wide = frame.select(varying).corr()
    return (
        wide.with_columns(pl.Series(FEATURE, varying))
        .unpivot(index=FEATURE, variable_name="other", value_name="correlation")
        .sort(FEATURE, "other")
    )


def missing_shares(frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Share of bars each feature is missing, per instrument.

    Missing is not imputed anywhere in the pipeline, so this is the share of rows a feature
    *costs*. A feature missing 40% of the time on one name is a reason to drop the feature or
    the name, and that decision belongs to the group, not to a silent fill.
    """
    if not features:
        return pl.DataFrame(schema=_missing_schema())
    shares = frame.group_by(C.INSTRUMENT).agg(
        [(1.0 - _present(f).mean()).alias(f) for f in features]
    )
    return shares.unpivot(
        index=C.INSTRUMENT, variable_name=FEATURE, value_name="missing_share"
    ).sort(C.INSTRUMENT, FEATURE)


def run_eda(frame: pl.DataFrame, horizon: HorizonSpec, params: EdaParams) -> EdaReport:
    """Compute the whole study on a built dataset.

    Args:
        frame: A dataset written by :func:`~hft_forecast.ml.datasets.build_dataset` —
            features, labels and the session columns.
        horizon: The horizon its labels were built for.
        params: Study knobs.

    Returns:
        The report, with one table per section.

    Raises:
        ConfigError: If the frame carries no ``f_*`` columns, or no row survives the label
            and session filters (which means the study would describe nothing).
    """
    require_columns(frame, NEEDS, "run_eda")
    features = feature_columns(frame)
    if not features:
        raise ConfigError("the dataset has no f_* columns; add a feature set to the config")
    sliced = train_slice(frame, params.train_sessions)
    valid = sliced.filter(pl.col(LABEL_VALID))
    if valid.is_empty():
        raise ConfigError(
            "no rows with a valid label in the studied sessions",
            details={"sessions": params.train_sessions, "rows": sliced.height},
        )
    bar_seconds = int(horizon.delta.astype("timedelta64[s]").astype(int))
    ic = block_ic(valid, features, min_block_rows=params.min_block_rows)
    ranking = ic_ranking(ic)
    report = EdaReport(
        horizon=horizon.name,
        horizon_seconds=horizon.seconds,
        bar_seconds=bar_seconds,
        summary=_summary(sliced, valid, ranking, params),
        coverage=coverage(sliced, features),
        ranking=ranking,
        buckets=bucket_profile(
            valid, features, buckets=params.buckets, min_block_rows=params.min_block_rows
        ),
        decay=return_decay(
            sliced,
            features,
            _decay_steps(horizon, params.decay_multiples),
            bar_seconds,
            params.min_block_rows,
        ),
        correlation=feature_correlation(valid, features),
        missing=missing_shares(sliced, features),
        blocks=ic,
    )
    return report


def _decay_steps(horizon: HorizonSpec, multiples: list[float]) -> list[int]:
    """Horizons to re-measure at, in bars: the configured one scaled and de-duplicated."""
    steps = {max(round(horizon.steps * m), 1) for m in multiples}
    steps.add(horizon.steps)
    return sorted(steps)


def _summary(
    sliced: pl.DataFrame, valid: pl.DataFrame, ranking: pl.DataFrame, params: EdaParams
) -> dict[str, Any]:
    """Header scalars: how much data, how balanced, and what came out on top."""
    direction = valid.get_column(DIRECTION).to_numpy() if DIRECTION in valid.columns else None
    top = ranking.head(1).to_dicts()
    return {
        "instruments": sliced.get_column(C.INSTRUMENT).n_unique(),
        "sessions": sliced.get_column(SESSION_ID).n_unique(),
        "bars": sliced.height,
        "labelled_bars": valid.height,
        "features": len(feature_columns(sliced)),
        # a feature that never moves has no correlation, so it is absent from the ranking
        # rather than sitting in it as a blank row; the gap is worth seeing
        "features_ranked": ranking.height,
        "train_sessions": params.train_sessions,
        "min_block_rows": params.min_block_rows,
        "class_down": _share(direction, -1),
        "class_flat": _share(direction, 0),
        "class_up": _share(direction, 1),
        "top_feature": top[0][FEATURE] if top else None,
        "top_mean_ic": top[0]["mean_ic"] if top else None,
    }


def _share(direction: Any, value: int) -> float | None:
    """Share of a direction class, or ``None`` when the column is absent."""
    if direction is None or not len(direction):
        return None
    return float(np.mean(direction == value))


def _varying(frame: pl.DataFrame, features: list[str]) -> list[str]:
    """The features that actually move, in the given order."""
    if not features or frame.is_empty():
        return []
    deviations = frame.select([pl.col(f).std().alias(f) for f in features]).row(0)
    return [
        name
        for name, std in zip(features, deviations, strict=True)
        if std is not None and np.isfinite(std) and std > 0
    ]


def _present(name: str) -> pl.Expr:
    """True where a feature has a usable value (not null, not inf)."""
    column = pl.col(name)
    return column.is_not_null() & column.is_finite()


def _ic_schema() -> dict[str, Any]:
    return {
        FEATURE: pl.String,
        C.INSTRUMENT: pl.String,
        SESSION_ID: pl.Int64,
        "n": pl.UInt32,
        SPEARMAN: pl.Float64,
        PEARSON: pl.Float64,
    }


def _ranking_schema() -> dict[str, Any]:
    return {
        FEATURE: pl.String,
        "blocks": pl.UInt32,
        "rows": pl.UInt32,
        "mean_ic": pl.Float64,
        "std_ic": pl.Float64,
        "median_ic": pl.Float64,
        "share_positive": pl.Float64,
        "mean_pearson": pl.Float64,
        "t_stat": pl.Float64,
    }


def _bucket_schema() -> dict[str, Any]:
    return {
        FEATURE: pl.String,
        BUCKET: pl.Int64,
        "median_block_mean_bps": pl.Float64,
        "mean_block_mean_bps": pl.Float64,
        "blocks": pl.UInt32,
        "rows": pl.UInt32,
    }


def _decay_schema() -> dict[str, Any]:
    return {
        FEATURE: pl.String,
        HORIZON_SECONDS: pl.Int64,
        "steps": pl.Int64,
        "blocks": pl.UInt32,
        "mean_ic": pl.Float64,
        "t_stat": pl.Float64,
    }


def _correlation_schema() -> dict[str, Any]:
    return {FEATURE: pl.String, "other": pl.String, "correlation": pl.Float64}


def _missing_schema() -> dict[str, Any]:
    return {C.INSTRUMENT: pl.String, FEATURE: pl.String, "missing_share": pl.Float64}
