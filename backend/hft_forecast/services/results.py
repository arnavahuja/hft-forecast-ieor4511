"""Read-side services over stored runs: metrics, charts, comparisons, exports."""

from __future__ import annotations

import io
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Final

import numpy as np
import polars as pl

from hft_forecast.analytics.service import AnalyticsService
from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.interfaces import ChartInput
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.specs import ChartSpec, MetricDescriptor, MetricValue
from hft_forecast.core.types import TimeArray, pl_times
from hft_forecast.services.run_store import RunRecord, RunStore

CACHE_SIZE: Final = 16
EXPORT_PANELS: Final = ("weights", "positions", "prices")


def _parse_bound(value: str | date | datetime | None) -> np.datetime64 | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime | date):
        return np.datetime64(value.isoformat()[:19])
    return np.datetime64(str(value).replace("Z", "")[:19])


def window_indices(timestamps: TimeArray, start: str | None, end: str | None) -> tuple[int, int]:
    """``[i0, i1)`` bar indices covering ``[start, end]``."""
    lo = _parse_bound(start)
    hi = _parse_bound(end)
    ts = timestamps.astype("datetime64[s]")
    i0 = int(np.searchsorted(ts, lo, "left")) if lo is not None else 0
    i1 = int(np.searchsorted(ts, hi, "right")) if hi is not None else len(ts)
    return i0, max(i1, i0)


class ResultService:
    """Loads results (LRU-cached) and serves metrics, charts and comparisons.

    Args:
        store: Run store.
        analytics: Analytics service.
    """

    def __init__(self, store: RunStore, analytics: AnalyticsService) -> None:
        self.store = store
        self.analytics = analytics
        self._cache: OrderedDict[str, BacktestResult] = OrderedDict()
        self._lock = threading.Lock()

    # ---- loading

    def result(self, run_id: str) -> BacktestResult:
        """Load a result (cached)."""
        with self._lock:
            if run_id in self._cache:
                self._cache.move_to_end(run_id)
                return self._cache[run_id]
        res = self.store.result(run_id)
        with self._lock:
            self._cache[run_id] = res
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return res

    def evict(self, run_id: str) -> None:
        """Drop a run from the cache (after deletion)."""
        with self._lock:
            self._cache.pop(run_id, None)

    def windowed(self, run_id: str, start: str | None, end: str | None) -> BacktestResult:
        """Result restricted to a date window (for brushing)."""
        res = self.result(run_id)
        if not start and not end:
            return res
        i0, i1 = window_indices(res.timestamps, start, end)
        if i1 - i0 < 2:
            raise ConfigError("Selected window has fewer than two bars")
        return res.window(i0, i1)

    # ---- metrics

    def descriptors(self) -> list[MetricDescriptor]:
        """All metric descriptors."""
        return self.analytics.descriptors()

    def metrics(
        self, run_id: str, start: str | None = None, end: str | None = None
    ) -> dict[str, MetricValue]:
        """Stored metrics, or recomputed for a window."""
        if not start and not end:
            return self.store.metrics(run_id)
        record = self.store.get(run_id)
        n, sharpes = self.store.trials(record.experiment)
        return self.analytics.compute(
            self.windowed(run_id, start, end),
            n_trials=n,
            trial_sharpes=sharpes,
            factors=self.store.factors(run_id),
        )

    # ---- charts

    def _inputs(
        self, run_ids: Sequence[str], start: str | None, end: str | None
    ) -> list[ChartInput]:
        inputs = []
        for rid in run_ids:
            record = self.store.get(rid)
            res = self.windowed(rid, start, end)
            metrics = self.metrics(rid, start, end)
            factors = self.store.factors(rid)
            inputs.append(
                ChartInput(
                    rid,
                    record.name or f"{record.strategy} {rid[:6]}",
                    res,
                    AnalyticsService.scalars(metrics),
                    {"factors": factors} if factors is not None else {},
                )
            )
        return inputs

    def available_charts(self, run_id: str | None, scope: str) -> list[dict[str, Any]]:
        """Charts for a scope, flagged as applicable to the run (single scope)."""
        specs = self.analytics.chart_specs(scope)
        if run_id is None:
            return specs
        from hft_forecast.core.registry import PluginKind, registry

        inputs = self._inputs([run_id], None, None)
        out = []
        for spec in specs:
            builder = registry(PluginKind.CHART).get(spec["name"]).create()
            out.append({**spec, "applicable": builder.applicable(inputs)})
        return out

    def chart(
        self,
        run_ids: Sequence[str],
        name: str,
        params: dict[str, Any] | None = None,
        start: str | None = None,
        end: str | None = None,
        max_points: int | None = None,
    ) -> ChartSpec:
        """Build a chart for one or more runs."""
        inputs = self._inputs(run_ids, start, end)
        kwargs: dict[str, Any] = {}
        if max_points is not None:
            kwargs["max_points"] = max_points
        return self.analytics.build_chart(name, inputs, params, **kwargs)

    # ---- tables

    def trades(self, run_id: str, offset: int = 0, limit: int = 500) -> dict[str, Any]:
        """A page of the trades table."""
        trades = self.result(run_id).trades
        page = trades.slice(offset, limit)
        return {"total": trades.height, "rows": _json_rows(page)}

    def positions(self, run_id: str, at: str | None = None) -> dict[str, Any]:
        """Positions and weights at a bar (default: last bar)."""
        res = self.result(run_id)
        idx = len(res.timestamps) - 1
        if at:
            idx = max(window_indices(res.timestamps, None, at)[1] - 1, 0)
        rows: list[dict[str, Any]] = [
            {
                "instrument_id": inst,
                "weight": float(res.weights[idx, j]),
                "units": float(res.positions[idx, j]),
                "price": _f(res.prices[idx, j]),
                "value": float(res.weights[idx, j] * res.equity[idx]),
            }
            for j, inst in enumerate(res.instruments)
            if abs(res.weights[idx, j]) > 0
        ]
        rows.sort(key=lambda r: -abs(float(r["weight"])))
        return {
            "timestamp": str(res.timestamps[idx]),
            "equity": float(res.equity[idx]),
            "rows": rows,
        }

    def overlays(self, run_id: str) -> list[dict[str, Any]]:
        """Before/after statistics per overlay."""
        from hft_forecast.analytics import stats as S

        res = self.result(run_id)
        ppy = res.periods_per_year
        out = []
        for rep in res.overlay_reports:
            before = {
                "cagr": S.cagr(rep.returns_before, ppy),
                "sharpe": S.sharpe(rep.returns_before, ppy),
                "max_drawdown": S.max_drawdown(rep.returns_before),
                "avg_gross": float(rep.gross_before.mean()),
            }
            after = {
                "cagr": S.cagr(rep.returns_after, ppy),
                "sharpe": S.sharpe(rep.returns_after, ppy),
                "max_drawdown": S.max_drawdown(rep.returns_after),
                "avg_gross": float(rep.gross_after.mean()),
            }
            out.append(
                {
                    "name": rep.name,
                    "position": rep.position,
                    "before": {k: S.nan_to_none(v) for k, v in before.items()},
                    "after": {k: S.nan_to_none(v) for k, v in after.items()},
                    "delta_cagr": S.nan_to_none(after["cagr"] - before["cagr"]),
                    "cells_changed": rep.cells_changed,
                    "mean_abs_change": rep.mean_abs_change,
                    "notes": list(rep.notes),
                }
            )
        return out

    # ---- comparison

    def compare_metrics(self, run_ids: Sequence[str], align: bool = False) -> dict[str, Any]:
        """Metrics table across runs with best/worst per row and config warnings."""
        records = [self.store.get(r) for r in run_ids]
        start = end = None
        if align:
            window = self.common_window(run_ids)
            if window is None:
                raise ConfigError("Runs do not overlap in time")
            start, end = window
        values = {r.id: self.metrics(r.id, start, end) for r in records}
        rows = []
        for d in self.descriptors():
            if d.kind.value != "scalar":
                continue
            vals = {
                rid: values[rid][d.key].value if d.key in values[rid] else None for rid in values
            }
            finite = {k: v for k, v in vals.items() if v is not None}
            best = worst = None
            if d.higher_is_better is not None and len(finite) >= 2:
                ordered = sorted(finite, key=lambda k: finite[k], reverse=bool(d.higher_is_better))
                best, worst = ordered[0], ordered[-1]
            rows.append(
                {
                    "key": d.key,
                    "label": d.label,
                    "group": d.group,
                    "format": d.format.value,
                    "higher_is_better": d.higher_is_better,
                    "values": vals,
                    "best": best,
                    "worst": worst,
                }
            )
        return {
            "runs": [{"id": r.id, "name": r.name, "strategy": r.strategy} for r in records],
            "rows": rows,
            "warnings": self.config_warnings(records),
            "window": {"start": start, "end": end},
        }

    def common_window(self, run_ids: Sequence[str]) -> tuple[str, str] | None:
        """Overlapping date range of runs (ISO strings)."""
        results = [self.result(r) for r in run_ids]
        lo = max(r.timestamps[0] for r in results)
        hi = min(r.timestamps[-1] for r in results)
        if lo > hi:
            return None
        return str(lo)[:19], str(hi)[:19]

    @staticmethod
    def config_warnings(records: Sequence[RunRecord]) -> list[str]:
        """Warn when runs differ in date range, costs or engine."""
        warnings = []

        def distinct(getter: Any) -> set[str]:
            return {str(getter(r.config)) for r in records}

        if len(distinct(lambda c: (c["data"]["start"], c["data"]["end"]))) > 1:
            warnings.append(
                "Runs cover different date ranges; use 'Align dates' to compare "
                "over the common period."
            )
        if len(distinct(lambda c: (c.get("costs"), c.get("slippage")))) > 1:
            warnings.append("Runs use different cost or slippage assumptions.")
        if len(distinct(lambda c: c["execution"]["engine"])) > 1:
            warnings.append("Runs use different engines.")
        if len(distinct(lambda c: c["data"]["frequency"])) > 1:
            warnings.append("Runs use different bar frequencies.")
        return warnings

    def combine(
        self, run_ids: Sequence[str], weights: Sequence[float] | None = None
    ) -> dict[str, Any]:
        """Portfolio of runs with fixed weights (rebalanced every bar) on common dates."""
        if not run_ids:
            raise ConfigError("Select at least one run")
        w = np.array(weights if weights else [1.0 / len(run_ids)] * len(run_ids))
        if len(w) != len(run_ids):
            raise ConfigError("One weight per run is required")
        results = [self.result(r) for r in run_ids]
        common = results[0].timestamps
        for res in results[1:]:
            common = np.intersect1d(common, res.timestamps)
        if len(common) < 2:
            raise ConfigError("Runs do not overlap in time")
        rets = np.column_stack(
            [res.returns[np.searchsorted(res.timestamps, common)] for res in results]
        )
        rets[0] = 0.0
        combined = rets @ w
        from hft_forecast.analytics import stats as S

        base = results[0]
        ppy = base.periods_per_year
        n = len(common)
        zero = np.zeros((n, 0))
        combo = BacktestResult(
            timestamps=common,
            instruments=(),
            equity=np.cumprod(1 + combined),
            returns=combined,
            gross_returns=combined,
            weights=zero,
            target_weights=zero,
            positions=zero,
            prices=zero,
            turnover=np.zeros(n),
            costs={},
            periods_per_year=ppy,
        )
        metrics = self.analytics.compute(combo)
        vols = np.array([S.annualized_vol(rets[:, k], ppy) for k in range(len(run_ids))])
        combo_vol = S.annualized_vol(combined, ppy)
        weighted_vol = float(np.abs(w) @ vols)
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(rets[1:].T) if len(run_ids) > 1 else np.ones((1, 1))
        from hft_forecast.analytics import chartkit as K

        return {
            "metrics": {k: v.model_dump() for k, v in metrics.items() if v.table is None},
            "weights": w.tolist(),
            "diversification_ratio": S.nan_to_none(weighted_vol / combo_vol) if combo_vol else None,
            "vol_reduction": S.nan_to_none(weighted_vol - combo_vol),
            "correlation": np.nan_to_num(np.atleast_2d(corr)).tolist(),
            "equity": [
                [t, v] for t, v in zip(K.iso(common), K.clean(K.wealth(combined)), strict=True)
            ],
        }

    # ---- export

    def export(self, run_id: str, fmt: str) -> tuple[bytes, str, str]:
        """Export a run as a zip of CSV or Parquet files, or an HTML tearsheet.

        Returns:
            ``(content, media_type, filename)``.
        """
        record = self.store.get(run_id)
        res = self.result(run_id)
        if fmt == "html":
            from hft_forecast.services.tearsheet import render_tearsheet

            html = render_tearsheet(record, res, self.store.metrics(run_id), self.descriptors())
            return html.encode(), "text/html", f"tearsheet-{run_id}.html"
        if fmt not in ("csv", "parquet"):
            raise ConfigError(f"Unknown export format '{fmt}'")
        frames = _export_frames(res)
        metrics = self.store.metrics(run_id)
        frames["metrics"] = pl.DataFrame(
            [
                {"key": k, "value": v.value, "benchmark": v.benchmark}
                for k, v in metrics.items()
                if v.table is None
            ],
            schema={"key": pl.String, "value": pl.Float64, "benchmark": pl.Float64},
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, frame in frames.items():
                out = io.BytesIO()
                if fmt == "csv":
                    frame.write_csv(out)
                else:
                    frame.write_parquet(out)
                zf.writestr(f"{name}.{fmt}", out.getvalue())
            zf.writestr("config.json", _json_dumps(record.config))
        return buf.getvalue(), "application/zip", f"run-{run_id}-{fmt}.zip"


def _export_frames(res: BacktestResult) -> dict[str, pl.DataFrame]:
    ts = pl_times(res.timestamps)
    series: dict[str, Any] = {
        C.TIMESTAMP: ts,
        "equity": res.equity,
        "returns": res.returns,
        "gross_returns": res.gross_returns,
        "turnover": res.turnover,
    }
    if res.benchmark_returns is not None:
        series["benchmark_returns"] = res.benchmark_returns
    for cat, v in res.costs.items():
        series[f"cost_{cat}"] = v
    frames = {
        "series": pl.DataFrame(series),
        "trades": res.trades,
        "orders": res.orders,
        "fills": res.fills,
    }
    for name in EXPORT_PANELS:
        panel = getattr(res, name)
        frames[name] = pl.DataFrame(
            {C.TIMESTAMP: ts, **{inst: panel[:, j] for j, inst in enumerate(res.instruments)}}
        )
    return frames


def _f(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = frame.to_dicts()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, datetime):
                row[k] = v.isoformat()
            elif isinstance(v, float) and not np.isfinite(v):
                row[k] = None
    return rows


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, default=str)
