"""The economic test: predictions through the existing backtest engine (§7.1).

Builds the configs; it does not run them. Running means a ``RunStore`` and the job
machinery, both of which live in ``services``, above ``ml`` in the layer order. ``services``
or the CLI calls :func:`per_name_configs` and executes what comes back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.run_config import (
    BacktestConfig,
    DataSpec,
    ExecutionSpec,
    PluginRef,
    RebalanceRule,
)
from hft_forecast.ml.config import ExperimentConfig

COST_VARIANTS: Final[dict[str, PluginRef | None]] = {
    "none": None,
    "quoted_spread": PluginRef(name="quoted_spread"),
    "quoted_spread_stressed": PluginRef(name="quoted_spread", params={"multiplier": 2.0}),
}
"""Slippage per variant.

The design asks for a third variant stacking square-root impact on the quoted spread, but a
run has one slippage model, so stressing the spread stands in for it. The breakeven cost
multiple (:func:`cost_scale_grid`) answers the same question more directly.
"""

PRIMARY_VARIANT: Final = "quoted_spread"
"""The variant the best model is chosen on."""


def cost_scale_grid() -> tuple[float, ...]:
    """Cost multiples for the breakeven search."""
    return (0.0, 0.5, 1.0, 2.0, 4.0)


def _execution() -> ExecutionSpec:
    """Decide at the bar end, trade at the next bar end, at the mid."""
    return ExecutionSpec(
        engine="vectorized",
        lag_bars=1,
        rebalance=RebalanceRule.ON_CHANGE,
        calendar="XNYS",
    )


def _strategy(
    config: ExperimentConfig, predictions: Path, model: str, **extra: object
) -> PluginRef:
    return PluginRef(
        name="prediction_signal",
        params={
            "predictions_path": str(predictions),
            "model": model,
            "hold_bars": config.horizon.steps,
            "overlap": "average",
            **extra,
        },
    )


def per_name_configs(
    config: ExperimentConfig,
    predictions: Path,
    model: str,
    variant: str = PRIMARY_VARIANT,
    instruments: list[str] | None = None,
) -> list[BacktestConfig]:
    """One single-instrument backtest per name: the primary economic test.

    Per name rather than pooled because a 50-name 5 s panel is 29.5 M rows, and because the
    distribution across names is more informative than one aggregate.

    Raises:
        KeyError: If ``variant`` is not one of :data:`COST_VARIANTS`.
    """
    slippage = _variant(variant)
    return [
        BacktestConfig(
            name=f"{config.experiment} {model} {instrument} {variant}",
            data=config.data.model_copy(update={"instruments": [instrument]}),
            strategy=_strategy(config, predictions, model),
            execution=_execution(),
            slippage=slippage,
            experiment=config.experiment,
            tags=["ml", "economic", config.horizon.name, model, variant],
            seed=config.seed,
        )
        for instrument in (instruments or config.data.instruments)
    ]


def cross_sectional_config(
    config: ExperimentConfig,
    predictions: Path,
    model: str,
    variant: str = PRIMARY_VARIANT,
    data: DataSpec | None = None,
) -> BacktestConfig:
    """One long/short book across the names, rebalanced every bar.

    Secondary on purpose: at 30 s with 50 names and marketable orders this is economically
    absurd. Run it, report it, and let the number say so.
    """
    slippage = _variant(variant)
    return BacktestConfig(
        name=f"{config.experiment} {model} cross-sectional {variant}",
        data=data or config.data,
        strategy=_strategy(config, predictions, model, output_kind="signals", mode="return"),
        constructor=PluginRef(name="quantile", params={"quantile": 0.1}),
        overlays=[PluginRef(name="gross_cap", params={"max_gross": 1.0})],
        execution=_execution(),
        slippage=slippage,
        experiment=config.experiment,
        tags=["ml", "economic", "cross_sectional", config.horizon.name, model, variant],
        seed=config.seed,
    )


def _variant(variant: str) -> PluginRef | None:
    if variant not in COST_VARIANTS:
        raise KeyError(f"Unknown cost variant '{variant}'; expected {sorted(COST_VARIANTS)}")
    return COST_VARIANTS[variant]


RETURN_COLUMN: Final = "net_return"


def pooled_returns(per_name: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Equal-weight average of the per-name net return series.

    A portfolio of the single-name books, which is what "pooled" means here. Each frame
    carries ``timestamp`` and :data:`RETURN_COLUMN`; names that are not trading at a given
    bar simply do not contribute to that bar's mean.
    """
    if not per_name:
        return pl.DataFrame({C.TIMESTAMP: [], "pooled_return": []})
    stacked = pl.concat([f.select(C.TIMESTAMP, RETURN_COLUMN) for f in per_name.values()])
    return (
        stacked.group_by(C.TIMESTAMP)
        .agg(pl.col(RETURN_COLUMN).mean().alias("pooled_return"))
        .sort(C.TIMESTAMP)
    )
