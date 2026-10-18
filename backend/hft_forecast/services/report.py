"""Comparison report: CSV tables and PNG charts for any set of stored runs.

Nothing here knows about particular strategies. Periods, crisis windows, cost levels, the
volatility target and the rolling window all come from the request.
"""

from __future__ import annotations

import io
import math
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.analytics import stats as S
from hft_forecast.analytics.stats import align_factors, ols
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.registry import PluginKind, registry
from hft_forecast.core.results import COST_COMMISSION, COST_FEES, COST_SLIPPAGE, BacktestResult
from hft_forecast.core.run_config import BacktestConfig, PluginRef
from hft_forecast.core.types import FloatArray, np_times
from hft_forecast.services.report_charts import render_charts
from hft_forecast.services.results import ResultService
from hft_forecast.services.run_store import RunStore
from hft_forecast.services.runner import BacktestRunner

DECIMALS: Final = 8
MONTHLY_PPY: Final = 12.5
"""Runs with at most this many periods per year are aligned by calendar month."""
NEUTRAL_EXPOSURE: Final = 0.25
"""Average |net exposure| below this marks a run as market neutral (for vol scaling)."""
TRADING_COSTS: Final = frozenset({COST_COMMISSION, COST_SLIPPAGE, COST_FEES})
MODELS: Final = (("CAPM", ("mkt_rf",)), ("FF5+MOM", ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")))


class ReportWindow(BaseModel):
    """A labelled date window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    start: date
    end: date


class ReportRequest(BaseModel):
    """What to put in a comparison report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_ids: list[str] = Field(min_length=1)
    align: bool = Field(True, description="Restrict every run to the common date range")
    periods: list[ReportWindow] = Field(
        default_factory=list, description="Sub-periods for the sub-period table"
    )
    crisis_windows: list[ReportWindow] = Field(
        default_factory=list, description="Windows for cumulative returns"
    )
    cost_levels_bps: list[float] = Field(
        default_factory=list,
        description="One-way trading cost levels to re-run each strategy at (empty: skip)",
    )
    vol_target: float | None = Field(
        0.10, gt=0, le=1, description="Annual vol for scaled series (None: no scaling)"
    )
    vol_scale: str = Field(
        "neutral",
        pattern="^(neutral|all|none)$",
        description="Which runs get a vol-scaled series: market-neutral ones, all, or none",
    )
    rolling_window: int = Field(36, ge=3, le=1000, description="Bars in the rolling Sharpe")
    charts: bool = Field(True, description="Include PNG charts")


@dataclass
class RunSeries:
    """One run on the report's date grid."""

    run_id: str
    name: str
    result: BacktestResult
    config: dict[str, Any]
    net: pd.Series[float]
    gross: pd.Series[float]
    rf: pd.Series[float]
    turnover: pd.Series[float]
    longs: pd.Series[float]
    shorts: pd.Series[float]
    net_exposure: pd.Series[float]
    costs: pd.Series[float]
    factors: Any
    scaled: bool = False

    @property
    def ppy(self) -> float:
        """Periods per year of the run."""
        return float(self.result.periods_per_year)


def _keys(timestamps: Any, monthly: bool) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(np.asarray(timestamps).astype("datetime64[ns]"))
    if monthly:  # month-end labels, whatever day each source uses
        return ts.to_period("M").to_timestamp(how="end").normalize()
    return ts.normalize()


def _unique_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


class ReportBuilder:
    """Builds the report zip.

    Args:
        store: Run store (records, factors).
        results: Result service (cached results).
        runner: Backtest runner (only used for cost sensitivity re-runs).
    """

    def __init__(self, store: RunStore, results: ResultService, runner: BacktestRunner) -> None:
        self.store = store
        self.results = results
        self.runner = runner

    # ------------------------------------------------------------------ inputs

    def _series(self, req: ReportRequest) -> tuple[list[RunSeries], bool]:
        records = [self.store.get(r) for r in req.run_ids]
        results = [self.results.result(r) for r in req.run_ids]
        monthly = all(r.periods_per_year <= MONTHLY_PPY for r in results)
        names = _unique_names([rec.name or f"{rec.strategy} {rec.id[:6]}" for rec in records])
        out = []
        for rec, res, name in zip(records, results, names, strict=True):
            idx = _keys(res.timestamps, monthly)
            prev = np.concatenate(([res.initial_capital], res.equity[:-1]))
            with np.errstate(divide="ignore", invalid="ignore"):
                cost_frac = np.where(prev > 0, res.total_costs / prev, 0.0)
            rf = res.risk_free if res.risk_free is not None else np.zeros(res.n_bars)

            def s(values: Any, index: pd.DatetimeIndex = idx) -> pd.Series[float]:
                return pd.Series(np.asarray(values, dtype=np.float64), index=index)

            out.append(
                RunSeries(
                    run_id=rec.id,
                    name=name,
                    result=res,
                    config=rec.config,
                    net=s(res.returns),
                    gross=s(res.gross_returns),
                    rf=s(rf),
                    turnover=s(res.turnover),
                    longs=s((res.weights > 0).sum(axis=1)),
                    shorts=s((res.weights < 0).sum(axis=1)),
                    net_exposure=s(res.net_exposure),
                    costs=s(cost_frac),
                    factors=self.store.factors(rec.id),
                )
            )
        if req.align:
            common = out[0].net.index
            for rs in out[1:]:
                common = common.intersection(rs.net.index)
            if len(common) < 2:
                raise ConfigError("Runs do not overlap in time")
            for rs in out:
                for attr in (
                    "net",
                    "gross",
                    "rf",
                    "turnover",
                    "longs",
                    "shorts",
                    "net_exposure",
                    "costs",
                ):
                    setattr(rs, attr, getattr(rs, attr).loc[common])
                # the first common bar carries no holding-period return for runs that start there
        for rs in out:
            neutral = float(np.nanmean(np.abs(rs.net_exposure.to_numpy()))) < NEUTRAL_EXPOSURE
            rs.scaled = req.vol_target is not None and (
                req.vol_scale == "all" or (req.vol_scale == "neutral" and neutral)
            )
        return out, monthly

    # ------------------------------------------------------------------ tables

    @staticmethod
    def _stats(r: FloatArray, rf: FloatArray, ppy: float, index: pd.Index[Any]) -> dict[str, Any]:
        excess = r - rf
        mdd = S.max_drawdown(r)
        growth = S.cagr(r, ppy)
        periods = S.drawdown_periods(r)
        worst = periods[0] if periods else None
        clean = r[np.isfinite(r)]
        return {
            "ann_return_arithmetic": S.annualized_mean(r, ppy),
            "ann_return_geometric": growth,
            "ann_vol": S.annualized_vol(r, ppy),
            "sharpe": S.sharpe(excess, ppy),
            "sortino": S.sortino(excess, ppy),
            "max_drawdown": mdd,
            "max_dd_peak": str(index[worst.start].date()) if worst else "",
            "max_dd_trough": str(index[worst.trough].date()) if worst else "",
            "calmar": growth / abs(mdd) if mdd < 0 else math.nan,
            "skewness": S.skewness(r),
            "excess_kurtosis": S.excess_kurtosis(r),
            "hit_rate": float((clean > 0).mean()) if clean.size else math.nan,
            "best_period": float(clean.max()) if clean.size else math.nan,
            "worst_period": float(clean.min()) if clean.size else math.nan,
        }

    def returns_table(self, runs: list[RunSeries], req: ReportRequest) -> pd.DataFrame:
        """Gross and net returns per run (plus vol-scaled series and risk-free)."""
        cols: dict[str, pd.Series[float]] = {}
        for rs in runs:
            cols[f"{rs.name} | gross"] = rs.gross
            cols[f"{rs.name} | net"] = rs.net
            if rs.scaled and req.vol_target is not None:
                cols[f"{rs.name} | net vol-scaled {req.vol_target:.0%}"] = self._scaled(rs, req)
        if any(float(rs.rf.abs().sum()) > 0 for rs in runs):
            cols["risk_free"] = runs[0].rf
        return pd.DataFrame(cols)

    @staticmethod
    def _scaled(rs: RunSeries, req: ReportRequest) -> pd.Series[float]:
        vol = S.annualized_vol(rs.net.to_numpy(), rs.ppy)
        factor = (req.vol_target or 0.0) / vol if vol and np.isfinite(vol) else math.nan
        return rs.net * factor

    def metrics_table(self, runs: list[RunSeries]) -> pd.DataFrame:
        """Performance, risk and holdings statistics per run, gross and net."""
        rows = []
        for rs in runs:
            active = (rs.longs + rs.shorts) > 0
            for variant, series in (("gross", rs.gross), ("net", rs.net)):
                row = {
                    "strategy": rs.name,
                    "variant": variant,
                    "start": str(rs.net.index[0].date()),
                    "end": str(rs.net.index[-1].date()),
                    "periods": len(series),
                }
                row.update(self._stats(series.to_numpy(), rs.rf.to_numpy(), rs.ppy, series.index))
                row["avg_annual_turnover_one_way"] = float(rs.turnover.mean() * rs.ppy)
                row["avg_longs"] = float(rs.longs[active].mean()) if active.any() else 0.0
                row["avg_shorts"] = float(rs.shorts[active].mean()) if active.any() else 0.0
                rows.append(row)
        return pd.DataFrame(rows)

    def regressions_table(self, runs: list[RunSeries]) -> pd.DataFrame:
        """CAPM and FF5+MOM regressions of net excess returns."""
        rows = []
        for rs in runs:
            if rs.factors is None:
                continue
            ts = rs.net.index.to_numpy().astype("datetime64[ns]")
            for model, names in MODELS:
                aligned = align_factors(ts, rs.net.to_numpy(), rs.factors, names)
                if aligned is None:
                    continue
                y, x, _ = aligned
                coef, t, r2 = ols(y, x)
                row: dict[str, Any] = {
                    "strategy": rs.name,
                    "model": model,
                    "observations": len(y),
                    "alpha_annualized": coef[0] * rs.ppy,
                    "alpha_t": t[0],
                    "r_squared": r2,
                }
                for k, name in enumerate(names):
                    row[f"beta_{name}"] = coef[k + 1]
                    row[f"t_{name}"] = t[k + 1]
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def correlation_table(runs: list[RunSeries]) -> pd.DataFrame:
        """Pairwise correlation of net returns."""
        frame = pd.DataFrame({rs.name: rs.net for rs in runs}).iloc[1:]
        return frame.corr()

    @staticmethod
    def factor_correlation_table(runs: list[RunSeries]) -> pd.DataFrame:
        """Correlation of each run's net returns with each factor."""
        rows = []
        for rs in runs:
            if rs.factors is None:
                continue
            f_ts = np_times(rs.factors.get_column("timestamp"))
            row: dict[str, Any] = {"strategy": rs.name}
            ts = rs.net.index.to_numpy().astype("datetime64[ns]")
            for col in rs.factors.columns:
                if col == "timestamp":
                    continue
                vals = S.align_series(
                    ts, f_ts, rs.factors.get_column(col).to_numpy().astype(np.float64)
                )
                ok = np.isfinite(vals) & np.isfinite(rs.net.to_numpy())
                row[col] = (
                    float(np.corrcoef(rs.net.to_numpy()[ok], vals[ok])[0, 1])
                    if ok.sum() > 2
                    else math.nan
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def window_table(
        self, runs: list[RunSeries], windows: Sequence[ReportWindow], monthly: bool
    ) -> pd.DataFrame:
        """Return, risk and Sharpe of every run inside each window."""
        rows = []
        for w in windows:
            lo, hi = pd.Timestamp(w.start), pd.Timestamp(w.end)
            if monthly:
                lo = lo.to_period("M").to_timestamp(how="end").normalize()
                hi = hi.to_period("M").to_timestamp(how="end").normalize()
            for rs in runs:
                sel = (rs.net.index >= lo) & (rs.net.index <= hi)
                r = rs.net[sel].to_numpy()
                rf = rs.rf[sel].to_numpy()
                rows.append(
                    {
                        "window": w.label,
                        "start": str(w.start),
                        "end": str(w.end),
                        "strategy": rs.name,
                        "periods": int(sel.sum()),
                        "cumulative_return": S.total_return(r) if len(r) else math.nan,
                        "ann_return_geometric": S.cagr(r, rs.ppy) if len(r) else math.nan,
                        "ann_vol": S.annualized_vol(r, rs.ppy) if len(r) else math.nan,
                        "sharpe": S.sharpe(r - rf, rs.ppy) if len(r) else math.nan,
                        "max_drawdown": S.max_drawdown(r) if len(r) else math.nan,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def annual_table(runs: list[RunSeries]) -> pd.DataFrame:
        """Calendar-year compounded net return per run."""
        cols = {}
        for rs in runs:
            years = pd.DatetimeIndex(rs.net.index).year
            cols[rs.name] = rs.net.groupby(years).apply(lambda x: float(np.prod(1 + x) - 1))
        frame = pd.DataFrame(cols)
        frame.index.name = "year"
        return frame

    @staticmethod
    def holdings_table(runs: list[RunSeries]) -> pd.DataFrame:
        """Number of long and short positions per bar."""
        cols = {}
        for rs in runs:
            cols[f"{rs.name} | longs"] = rs.longs
            cols[f"{rs.name} | shorts"] = rs.shorts
        return pd.DataFrame(cols)

    @staticmethod
    def reconciliation_table(runs: list[RunSeries]) -> pd.DataFrame:
        """Gross minus net returns versus recorded costs."""
        rows = []
        for rs in runs:
            drag = (rs.gross - rs.net).to_numpy()
            costs = rs.costs.to_numpy()
            by_cat = {f"total_{cat}": float(np.nansum(v)) for cat, v in rs.result.costs.items()}
            rows.append(
                {
                    "strategy": rs.name,
                    "sum_gross_minus_net": float(np.nansum(drag)),
                    "sum_costs_over_equity": float(np.nansum(costs)),
                    "max_abs_difference": float(np.nanmax(np.abs(drag - costs)))
                    if len(drag)
                    else 0,
                    **by_cat,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ cost sensitivity

    @staticmethod
    def _cost_variant(config: BacktestConfig, bps: float) -> BacktestConfig:
        keep = []
        for ref in config.costs:
            category = getattr(registry(PluginKind.COST_MODEL).get(ref.name).cls, "category", "")
            if category not in TRADING_COSTS:
                keep.append(ref)
        keep.append(PluginRef(name="bps_notional", params={"bps": bps}))
        return config.model_copy(update={"costs": keep, "slippage": None})

    def cost_table(
        self, runs: list[RunSeries], req: ReportRequest, progress: Callable[[float, str], None]
    ) -> pd.DataFrame:
        """Net Sharpe and return with trading costs re-run at each level."""
        rows = []
        total = max(len(runs) * len(req.cost_levels_bps), 1)
        done = 0
        for rs in runs:
            base = BacktestConfig.model_validate(rs.config)
            prepared = self.runner.prepare(base)
            for bps in req.cost_levels_bps:
                cfg = self._cost_variant(base, bps)
                comps = self.runner.components(cfg)
                res = comps.engine.run(
                    cfg, prepared.data, comps.strategy, comps.pipeline, prepared.options
                )
                idx = _keys(res.timestamps, rs.ppy <= MONTHLY_PPY)
                net = pd.Series(res.returns, index=idx).reindex(rs.net.index)
                rf = rs.rf.reindex(net.index).fillna(0.0)
                r = net.to_numpy()
                rows.append(
                    {
                        "strategy": rs.name,
                        "one_way_bps": bps,
                        "net_sharpe": S.sharpe(r - rf.to_numpy(), rs.ppy),
                        "net_ann_return_geometric": S.cagr(r, rs.ppy),
                        "net_ann_return_arithmetic": S.annualized_mean(r, rs.ppy),
                    }
                )
                done += 1
                progress(done / total, f"cost level {done}/{total}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ build

    def build(
        self, req: ReportRequest, progress: Callable[[float, str], None] | None = None
    ) -> bytes:
        """Build the zip (CSV tables, PNG charts, README)."""
        report = progress or (lambda _f, _m: None)
        runs, monthly = self._series(req)
        tables: dict[str, pd.DataFrame] = {
            "returns": self.returns_table(runs, req),
            "metrics": self.metrics_table(runs),
            "regressions": self.regressions_table(runs),
            "correlations": self.correlation_table(runs),
            "factor_correlations": self.factor_correlation_table(runs),
            "annual_returns": self.annual_table(runs),
            "holdings_count": self.holdings_table(runs),
            "cost_reconciliation": self.reconciliation_table(runs),
        }
        if req.periods:
            tables["subperiods"] = self.window_table(runs, req.periods, monthly)
        if req.crisis_windows:
            tables["crisis_table"] = self.window_table(runs, req.crisis_windows, monthly)
        report(0.2, "tables")
        if req.cost_levels_bps:
            tables["cost_sensitivity"] = self.cost_table(
                runs, req, lambda f, m: report(0.2 + 0.6 * f, m)
            )
        pngs = (
            render_charts(runs, tables, req, lambda rs: self._scaled(rs, req)) if req.charts else {}
        )
        report(0.95, "writing")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, frame in tables.items():
                if frame is None or frame.empty:
                    continue
                index = name in ("returns", "correlations", "annual_returns", "holdings_count")
                text = frame.to_csv(
                    index=index, float_format=f"%.{DECIMALS}f", date_format="%Y-%m-%d"
                )
                zf.writestr(f"{name}.csv", text)
            for name, png in pngs.items():
                zf.writestr(f"charts/{name}", png)
            zf.writestr("README.txt", _readme(runs, req, monthly))
        return buf.getvalue()


_SCALED_NOTE: Final = "  [vol-scaled series included]"


def _readme(runs: list[RunSeries], req: ReportRequest, monthly: bool) -> str:
    lines = [
        "hft-forecast comparison report",
        "",
        "Runs:",
        *[f"  - {rs.name} (run {rs.run_id}){_SCALED_NOTE if rs.scaled else ''}" for rs in runs],
        "",
        f"Dates: {'common range of all runs' if req.align else 'each run over its own range'}"
        f"; bars matched by {'calendar month' if monthly else 'day'}.",
        "Returns: 'gross' is before trading costs and financing, 'net' after. With a",
        "risk-free series configured on a run, cash earns it and Sharpe/Sortino use excess",
        "returns over it; regressions always use excess returns over the factor file's rf.",
        "A long-short run whose cash earns the risk-free rate therefore has Sharpe and",
        "alpha computed on its long-minus-short spread.",
        f"Vol-scaled series: net returns x ({req.vol_target} / full-sample annualized vol); "
        "for display only.",
        "cost_sensitivity.csv: each run re-executed with its trading-cost models (commission,",
        "slippage, fees) replaced by one bps-of-notional model at each level; holding costs",
        "(borrow, financing) are kept.",
        "cost_reconciliation.csv: gross minus net per bar versus recorded costs / equity.",
        "factor_correlations.csv: correlation of net returns with each factor column.",
        "holdings_count.csv: number of long and short positions per bar.",
    ]
    return "\n".join(lines) + "\n"
