"""Persist a :class:`BacktestResult` as Parquet + JSON under ``runs/<run_id>/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.results import BacktestResult, OverlayReport
from hft_forecast.core.types import FloatArray, Frequency, TimeArray, np_times, pl_times

SERIES: Final = "series.parquet"
PANELS: Final = ("weights", "target_weights", "positions", "prices")
TABLES: Final = ("trades", "orders", "fills")
OVERLAYS: Final = "overlays.parquet"
META: Final = "meta.json"
COST_PREFIX: Final = "cost_"
BENCH_COL: Final = "benchmark_returns"
RF_COL: Final = "risk_free"


def _wide(ts: TimeArray, instruments: tuple[str, ...], values: FloatArray) -> pl.DataFrame:
    cols: dict[str, Any] = {C.TIMESTAMP: pl_times(ts)}
    for j, inst in enumerate(instruments):
        cols[inst] = values[:, j]
    return pl.DataFrame(cols)


def save_result(result: BacktestResult, directory: Path) -> None:
    """Write a result to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    ts = result.timestamps
    series: dict[str, Any] = {
        C.TIMESTAMP: pl_times(ts),
        "equity": result.equity,
        "returns": result.returns,
        "gross_returns": result.gross_returns,
        "turnover": result.turnover,
    }
    if result.benchmark_returns is not None:
        series[BENCH_COL] = result.benchmark_returns
    if result.risk_free is not None:
        series[RF_COL] = result.risk_free
    for cat, values in result.costs.items():
        series[f"{COST_PREFIX}{cat}"] = values
    pl.DataFrame(series).write_parquet(directory / SERIES)
    for name in PANELS:
        _wide(ts, result.instruments, getattr(result, name)).write_parquet(
            directory / f"{name}.parquet"
        )
    for name in TABLES:
        getattr(result, name).write_parquet(directory / f"{name}.parquet")
    for name, values in result.aux.items():
        _wide(ts, result.instruments, values).write_parquet(directory / f"aux_{name}.parquet")
    overlay_meta = []
    if result.overlay_reports:
        cols: dict[str, Any] = {C.TIMESTAMP: pl_times(ts)}
        for rep in result.overlay_reports:
            k = rep.position
            cols[f"gross_before_{k}"] = rep.gross_before
            cols[f"gross_after_{k}"] = rep.gross_after
            cols[f"returns_before_{k}"] = rep.returns_before
            cols[f"returns_after_{k}"] = rep.returns_after
            overlay_meta.append(
                {
                    "name": rep.name,
                    "position": k,
                    "mean_abs_change": rep.mean_abs_change,
                    "cells_changed": rep.cells_changed,
                    "notes": list(rep.notes),
                }
            )
        pl.DataFrame(cols).write_parquet(directory / OVERLAYS)
    meta = {
        "instruments": list(result.instruments),
        "periods_per_year": result.periods_per_year,
        "frequency": result.frequency.value,
        "benchmark_id": result.benchmark_id,
        "config": result.config,
        "metadata": result.metadata,
        "overlays": overlay_meta,
        "aux": sorted(result.aux),
    }
    (directory / META).write_text(json.dumps(meta, indent=2, default=str))


def load_result(directory: Path) -> BacktestResult:
    """Load a result saved by :func:`save_result`."""
    meta = json.loads((directory / META).read_text())
    instruments = tuple(meta["instruments"])
    series = pl.read_parquet(directory / SERIES)
    ts = np_times(series.get_column(C.TIMESTAMP))

    def panel(name: str) -> FloatArray:
        frame = pl.read_parquet(directory / f"{name}.parquet")
        if not instruments:
            return np.zeros((len(ts), 0))
        return frame.select(list(instruments)).to_numpy().astype(np.float64)

    costs = {
        c[len(COST_PREFIX) :]: series.get_column(c).to_numpy()
        for c in series.columns
        if c.startswith(COST_PREFIX)
    }
    reports: list[OverlayReport] = []
    if meta.get("overlays"):
        ov = pl.read_parquet(directory / OVERLAYS)
        for item in meta["overlays"]:
            k = item["position"]
            reports.append(
                OverlayReport(
                    name=item["name"],
                    position=k,
                    gross_before=ov.get_column(f"gross_before_{k}").to_numpy(),
                    gross_after=ov.get_column(f"gross_after_{k}").to_numpy(),
                    returns_before=ov.get_column(f"returns_before_{k}").to_numpy(),
                    returns_after=ov.get_column(f"returns_after_{k}").to_numpy(),
                    mean_abs_change=float(item["mean_abs_change"]),
                    cells_changed=int(item["cells_changed"]),
                    notes=tuple(item.get("notes", [])),
                )
            )
    return BacktestResult(
        timestamps=ts,
        instruments=instruments,
        equity=series.get_column("equity").to_numpy(),
        returns=series.get_column("returns").to_numpy(),
        gross_returns=series.get_column("gross_returns").to_numpy(),
        weights=panel("weights"),
        target_weights=panel("target_weights"),
        positions=panel("positions"),
        prices=panel("prices"),
        turnover=series.get_column("turnover").to_numpy(),
        costs=costs,
        periods_per_year=float(meta["periods_per_year"]),
        frequency=Frequency(meta["frequency"]),
        benchmark_returns=(
            series.get_column(BENCH_COL).to_numpy() if BENCH_COL in series.columns else None
        ),
        benchmark_id=meta.get("benchmark_id"),
        risk_free=series.get_column(RF_COL).to_numpy() if RF_COL in series.columns else None,
        trades=pl.read_parquet(directory / "trades.parquet"),
        orders=pl.read_parquet(directory / "orders.parquet"),
        fills=pl.read_parquet(directory / "fills.parquet"),
        overlay_reports=tuple(reports),
        aux={name: panel(f"aux_{name}") for name in meta.get("aux", [])},
        config=meta.get("config", {}),
        metadata=meta.get("metadata", {}),
    )
