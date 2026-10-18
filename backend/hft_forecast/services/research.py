"""Research tools: parameter sweeps, walk-forward, Monte Carlo, sensitivity and regimes.

Every tool stores a generic result (``summary``, ``tables`` and library-neutral ``charts``) as
a research record in the run store, so the front end renders all of them the same way.
Sweeps and walk-forward prepare the data once and run the vectorized engine for each
parameter set (in threads).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from hft_forecast.analytics import chartkit as K
from hft_forecast.analytics import stats as S
from hft_forecast.analytics.robustness.core import (
    FAN_PERCENTILES,
    bootstrap_paths,
    breakeven,
    expand_grid,
    path_stats,
    pbo_cscv,
    random_grid,
    shuffle_trade_paths,
    walk_forward_windows,
)
from hft_forecast.core.errors import ConfigError, JobCancelledError
from hft_forecast.core.numeric import rolling_mean, rolling_std
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.run_config import BacktestConfig, PluginRef
from hft_forecast.core.specs import (
    Axis,
    AxisType,
    ChartSeries,
    ChartSpec,
    ChartType,
    MetricFormat,
    SeriesRole,
)
from hft_forecast.core.types import FloatArray
from hft_forecast.engine.vectorized import VectorizedEngine
from hft_forecast.services.container import Services
from hft_forecast.services.runner import PreparedRun

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

DEFAULT_METRIC: Final = "sharpe"
MAX_TRIALS: Final = 2000
TOP_EQUITY_CURVES: Final = 5
REGIME_SMA: Final = 200
REGIME_VOL_WINDOW: Final = 63
TERCILES: Final = 3
HIGHER_IS_WORSE: Final = {"ann_vol", "turnover_ann"}


# --------------------------------------------------------------------------- helpers


def with_params(config: BacktestConfig, target: str, params: dict[str, Any]) -> BacktestConfig:
    """Copy of ``config`` with ``params`` merged into a plugin reference.

    ``target`` is ``strategy``, ``constructor``, ``slippage``, ``fill_model``,
    ``overlays.N`` or ``costs.N``.
    """
    raw = config.model_dump(mode="json")
    head, _, index = target.partition(".")
    node: Any = raw.get(head)
    if index:
        if not isinstance(node, list) or not index.isdigit() or int(index) >= len(node):
            raise ConfigError(f"Invalid sweep target '{target}'")
        node = node[int(index)]
    if not isinstance(node, dict):
        raise ConfigError(f"Sweep target '{target}' is not set in the config")
    node["params"] = {**node.get("params", {}), **params}
    return BacktestConfig.model_validate(raw)


def quick_metrics(result: BacktestResult) -> dict[str, float | None]:
    """The metrics used to rank research trials."""
    r, ppy = result.returns, result.periods_per_year
    nn = S.nan_to_none
    return {
        "sharpe": nn(S.sharpe(r, ppy)),
        "cagr": nn(S.cagr(r, ppy)),
        "ann_vol": nn(S.annualized_vol(r, ppy)),
        "max_drawdown": nn(S.max_drawdown(r)),
        "sortino": nn(S.sortino(r, ppy)),
        "turnover_ann": nn(float(result.turnover.mean() * ppy)),
        "total_return": nn(S.total_return(r)),
    }


def _score(metrics: dict[str, float | None], metric: str) -> float:
    value = metrics.get(metric)
    if value is None:
        return -math.inf
    return -value if metric in HIGHER_IS_WORSE else value


@dataclass
class Trial:
    """One evaluated parameter set."""

    params: dict[str, Any]
    result: BacktestResult
    metrics: dict[str, float | None]


class Evaluator:
    """Runs configuration variants on data prepared once."""

    def __init__(
        self, services: Services, base: BacktestConfig, progress: Progress, cancelled: Cancelled
    ) -> None:
        self.services = services
        self.base = base.model_copy(
            update={"execution": base.execution.model_copy(update={"engine": "vectorized"})}
        )
        self.progress = progress
        self.cancelled = cancelled
        progress(0.02, "loading data")
        self.prepared: PreparedRun = services.runner.prepare(self.base)

    def run(self, config: BacktestConfig, cost_scale: float = 1.0) -> BacktestResult:
        """Run one variant (vectorized)."""
        comps = self.services.runner.components(config)
        engine = comps.engine
        if isinstance(engine, VectorizedEngine) and cost_scale != 1.0:
            engine = VectorizedEngine(engine.cost_models, engine.slippage, cost_scale)
        return engine.run(
            config, self.prepared.data, comps.strategy, comps.pipeline, self.prepared.options
        )

    def trials(
        self, target: str, points: Sequence[dict[str, Any]], start: float, end: float
    ) -> list[Trial]:
        """Evaluate parameter points in threads with progress reporting."""
        if len(points) > MAX_TRIALS:
            raise ConfigError(
                f"At most {MAX_TRIALS} trials per sweep", details={"requested": len(points)}
            )
        configs = [with_params(self.base, target, p) for p in points]
        out: list[Trial | None] = [None] * len(configs)
        workers = max(self.services.settings.max_workers, 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.run, cfg): k for k, cfg in enumerate(configs)}
            for done, fut in enumerate(futures):
                if self.cancelled():
                    raise JobCancelledError("Research job cancelled")
                k = futures[fut]
                res = fut.result()
                out[k] = Trial(points[k], res, quick_metrics(res))
                self.progress(
                    start + (end - start) * (done + 1) / len(configs),
                    f"trial {done + 1}/{len(configs)}",
                )
        return [t for t in out if t is not None]


def _table(
    title: str, rows: list[dict[str, Any]], formats: dict[str, str | None]
) -> dict[str, Any]:
    columns = [
        {"key": k, "label": k.replace("_", " "), "format": formats.get(k)}
        for k in (rows[0].keys() if rows else [])
    ]
    return {"title": title, "columns": columns, "rows": rows}


METRIC_FORMATS: Final[dict[str, str | None]] = {
    "sharpe": "ratio",
    "sortino": "ratio",
    "cagr": "percent",
    "ann_vol": "percent",
    "max_drawdown": "percent",
    "turnover_ann": "percent",
    "total_return": "percent",
}


def _record(
    services: Services,
    config: BacktestConfig,
    kind: str,
    payload: dict[str, Any],
    headline: dict[str, float | None],
) -> dict[str, Any]:
    record = services.store.create(config, kind=kind)
    services.store.complete_research(record.id, {"kind": kind, **payload}, headline)
    return {"run_id": record.id, "headline": headline}


def _equity_chart(
    title: str, series: list[tuple[str, FloatArray]], ts: Any, chart_id: str
) -> ChartSpec:
    return ChartSpec(
        id=chart_id,
        title=title,
        type=ChartType.LINE,
        x_axis=K.time_axis(),
        y_axes=[K.value_axis("Growth of 1", MetricFormat.NUMBER)],
        time_series=True,
        series=[K.time_series(name, ts, K.wealth(r)) for name, r in series],
    )


# --------------------------------------------------------------------------- sweep


def run_sweep(
    services: Services, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
) -> dict[str, Any]:
    """Grid or random search over one plugin's parameters.

    Payload: ``config``, ``target`` (default ``strategy``), ``grid`` (``{param: [values]}``) or
    ``random`` (``{"space": {...}, "n": int}``), ``metric``, ``seed``.
    """
    base = BacktestConfig.model_validate(payload["config"])
    target = str(payload.get("target", "strategy"))
    metric = str(payload.get("metric", DEFAULT_METRIC))
    if payload.get("grid"):
        points = expand_grid(payload["grid"])
    elif payload.get("random"):
        spec = payload["random"]
        points = random_grid(
            spec["space"],
            int(spec.get("n", 50)),
            np.random.default_rng(int(payload.get("seed", base.seed))),
        )
    else:
        raise ConfigError("Provide a 'grid' or a 'random' search space")
    ev = Evaluator(services, base, progress, cancelled)
    trials = ev.trials(target, points, 0.05, 0.85)
    ranked = sorted(trials, key=lambda t: _score(t.metrics, metric), reverse=True)
    best = ranked[0]
    ppy = best.result.periods_per_year
    sharpes_pp = np.array([(t.metrics["sharpe"] or np.nan) / math.sqrt(ppy) for t in trials])
    n_trials = len(trials)
    if base.experiment:
        _, prior = services.store.trials(base.experiment)
        n_trials += services.store.count_trials(base.experiment)
        sharpes_pp = np.concatenate([sharpes_pp, np.array(prior)])
    dsr = S.deflated_sharpe(best.result.returns, sharpes_pp, n_trials)
    matrix = np.column_stack([t.result.returns for t in trials])
    pbo = pbo_cscv(matrix[1:])
    progress(0.9, "building charts")
    charts = _sweep_charts(trials, ranked, metric, sorted(points[0]) if points else [])
    rows = [{**t.params, **t.metrics} for t in ranked]
    summary = {
        "trials": len(trials),
        "metric": metric,
        "best_value": best.metrics.get(metric),
        "best_params": best.params,
        "deflated_sharpe_best": S.nan_to_none(dsr),
        "pbo": pbo["pbo"],
        "experiment_trials": n_trials,
    }
    if base.experiment:
        services.store.add_trials(
            base.experiment,
            "sweep",
            [
                (t.params, t.metrics["sharpe"] / math.sqrt(ppy) if t.metrics["sharpe"] else None)
                for t in trials
            ],
        )
    return _record(
        services,
        base,
        "sweep",
        {
            "target": target,
            "summary": summary,
            "tables": [_table("Trials (best first)", rows, METRIC_FORMATS)],
            "charts": [c.model_dump(mode="json") for c in charts],
        },
        {
            "sharpe": best.metrics["sharpe"],
            "cagr": best.metrics["cagr"],
            "pbo": pbo["pbo"],
            "dsr": S.nan_to_none(dsr),
        },
    )


def _sweep_charts(
    trials: list[Trial], ranked: list[Trial], metric: str, params: list[str]
) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    varying = [p for p in params if len({repr(t.params[p]) for t in trials}) > 1]
    if len(varying) == 2:
        a, b = varying
        xs = sorted({t.params[a] for t in trials}, key=_sort_key)
        ys = sorted({t.params[b] for t in trials}, key=_sort_key)
        cells = [
            [xs.index(t.params[a]), ys.index(t.params[b]), t.metrics.get(metric)] for t in trials
        ]
        charts.append(
            ChartSpec(
                id="parameter_heatmap",
                title=f"{metric} by {a} and {b}",
                type=ChartType.HEATMAP,
                x_axis=Axis(type=AxisType.CATEGORY, name=a, categories=[str(x) for x in xs]),
                y_axes=[Axis(type=AxisType.CATEGORY, name=b, categories=[str(y) for y in ys])],
                series=[ChartSeries(name=metric, data=cells)],
                options={"value_format": METRIC_FORMATS.get(metric, "number"), "diverging": True},
            )
        )
    for p in varying:
        by_value: dict[str, list[float]] = {}
        for t in trials:
            v = t.metrics.get(metric)
            if v is not None:
                by_value.setdefault(repr(t.params[p]), []).append(v)
        values = sorted({t.params[p] for t in trials}, key=_sort_key)
        data = [[str(v), float(np.mean(by_value.get(repr(v), [np.nan])))] for v in values]
        charts.append(
            ChartSpec(
                id=f"sensitivity_{p}",
                title=f"Sensitivity of {metric} to {p} (mean over others)",
                type=ChartType.LINE,
                x_axis=Axis(type=AxisType.CATEGORY, name=p, categories=[d[0] for d in data]),
                y_axes=[Axis(name=metric)],
                series=[ChartSeries(name=metric, data=data)],
            )
        )
    top = ranked[:TOP_EQUITY_CURVES]
    ts = top[0].result.timestamps
    charts.append(
        _equity_chart(
            "Top parameter sets",
            [(", ".join(f"{k}={v}" for k, v in t.params.items()), t.result.returns) for t in top],
            ts,
            "sweep_top_equity",
        )
    )
    boot = _bootstrap_sharpes(ranked[0].result)
    if boot is not None:
        charts.append(boot)
    return charts


def _sort_key(value: Any) -> tuple[int, Any]:
    return (0, value) if isinstance(value, int | float) else (1, str(value))


def _bootstrap_sharpes(result: BacktestResult, n: int = 1000, seed: int = 7) -> ChartSpec | None:
    r = result.returns[1:]
    if len(r) < 20:
        return None
    paths = bootstrap_paths(r, n, len(r), 5.0, seed)
    sd = paths.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = paths.mean(axis=1) / sd * np.sqrt(result.periods_per_year)
    sharpes = sharpes[np.isfinite(sharpes)]
    counts, edges = np.histogram(sharpes, bins=40)
    mids = (edges[:-1] + edges[1:]) / 2
    return ChartSpec(
        id="bootstrap_sharpe",
        title="Bootstrap Sharpe distribution (best parameters)",
        type=ChartType.BAR,
        x_axis=Axis(type=AxisType.VALUE, name="Sharpe"),
        y_axes=[Axis(name="Samples", format=MetricFormat.INTEGER)],
        series=[
            ChartSeries(
                name="Samples", data=[[float(m), int(c)] for m, c in zip(mids, counts, strict=True)]
            )
        ],
    )


# --------------------------------------------------------------------------- walk-forward


def run_walk_forward(
    services: Services, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
) -> dict[str, Any]:
    """Walk-forward optimization with stitched out-of-sample returns.

    Payload: ``config``, ``target``, ``grid``, ``train_bars``, ``test_bars``, ``anchored``,
    ``metric``. Every parameter set is simulated once over the full history (strategies are
    causal, verified by the lookahead check), then each window picks the in-sample best and
    uses its returns out of sample.
    """
    base = BacktestConfig.model_validate(payload["config"])
    target = str(payload.get("target", "strategy"))
    metric = str(payload.get("metric", DEFAULT_METRIC))
    points = expand_grid(payload.get("grid") or {})
    if not points:
        raise ConfigError("Walk-forward needs a parameter 'grid'")
    ev = Evaluator(services, base, progress, cancelled)
    trials = ev.trials(target, points, 0.05, 0.8)
    ts = trials[0].result.timestamps
    ppy = trials[0].result.periods_per_year
    returns = np.column_stack([t.result.returns for t in trials])
    train = int(payload.get("train_bars", round(3 * ppy)))
    test = int(payload.get("test_bars", round(ppy / 2)))
    windows = walk_forward_windows(len(ts), train, test, bool(payload.get("anchored", False)))
    if not windows:
        raise ConfigError("Not enough bars for one train/test window")
    oos = np.zeros(len(ts))
    rows: list[dict[str, Any]] = []
    for w in windows:
        scores = [
            _window_score(returns[w.train_start : w.train_end, k], ppy, metric)
            for k in range(len(trials))
        ]
        k = int(np.argmax(scores))
        oos[w.train_end : w.test_end] = returns[w.train_end : w.test_end, k]
        rows.append(
            {
                "train_start": str(ts[w.train_start])[:10],
                "test_start": str(ts[w.train_end])[:10],
                "test_end": str(ts[w.test_end - 1])[:10],
                "params": trials[k].params,
                "in_sample": S.nan_to_none(scores[k]),
                "out_of_sample": S.nan_to_none(
                    _window_score(returns[w.train_end : w.test_end, k], ppy, metric)
                ),
            }
        )
    start = windows[0].train_end
    stitched = oos[start:]
    best_full = int(np.argmax([_score(t.metrics, metric) for t in trials]))
    is_mean = float(np.nanmean([r["in_sample"] for r in rows if r["in_sample"] is not None]))
    oos_metric = _window_score(stitched, ppy, metric)
    summary = {
        "windows": len(windows),
        "metric": metric,
        "oos_" + metric: S.nan_to_none(oos_metric),
        "oos_cagr": S.nan_to_none(S.cagr(stitched, ppy)),
        "oos_max_drawdown": S.nan_to_none(S.max_drawdown(stitched)),
        "mean_in_sample_" + metric: S.nan_to_none(is_mean),
        "efficiency": S.nan_to_none(oos_metric / is_mean) if is_mean else None,
        "full_sample_best_params": trials[best_full].params,
    }
    oos_rets = np.concatenate([np.zeros(start), stitched])
    charts = [
        _equity_chart(
            "Stitched out-of-sample vs full-sample best (hindsight)",
            [
                ("Walk-forward OOS", oos_rets),
                (
                    "Full-sample best (in-sample)",
                    np.concatenate([np.zeros(start), returns[start:, best_full]]),
                ),
            ],
            ts,
            "walk_forward_equity",
        ),
        ChartSpec(
            id="walk_forward_is_oos",
            title=f"In-sample vs out-of-sample {metric} per window",
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.CATEGORY, categories=[r["test_start"] for r in rows]),
            y_axes=[Axis(name=metric)],
            series=[
                ChartSeries(
                    name="In-sample",
                    data=[[r["test_start"], r["in_sample"]] for r in rows],
                    role=SeriesRole.REFERENCE,
                ),
                ChartSeries(
                    name="Out-of-sample", data=[[r["test_start"], r["out_of_sample"]] for r in rows]
                ),
            ],
        ),
    ]
    table_rows = [
        {
            **{k: v for k, v in r.items() if k != "params"},
            "params": ", ".join(f"{a}={b}" for a, b in r["params"].items()),
        }
        for r in rows
    ]
    return _record(
        services,
        base,
        "walk_forward",
        {
            "target": target,
            "summary": summary,
            "tables": [
                _table(
                    "Windows",
                    table_rows,
                    {
                        "in_sample": METRIC_FORMATS.get(metric),
                        "out_of_sample": METRIC_FORMATS.get(metric),
                    },
                )
            ],
            "charts": [c.model_dump(mode="json") for c in charts],
        },
        {
            "sharpe": S.nan_to_none(S.sharpe(stitched, ppy)),
            "cagr": S.nan_to_none(S.cagr(stitched, ppy)),
        },
    )


def _window_score(r: FloatArray, ppy: float, metric: str) -> float:
    fake = BacktestResult(
        timestamps=np.array([], dtype="datetime64[us]"),
        instruments=(),
        equity=np.array([]),
        returns=r,
        gross_returns=r,
        weights=np.zeros((len(r), 0)),
        target_weights=np.zeros((len(r), 0)),
        positions=np.zeros((len(r), 0)),
        prices=np.zeros((len(r), 0)),
        turnover=np.zeros(len(r)),
        costs={},
        periods_per_year=ppy,
    )
    return _score(quick_metrics(fake), metric)


# --------------------------------------------------------------------------- monte carlo


def run_monte_carlo(
    services: Services, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
) -> dict[str, Any]:
    """Bootstrap (stationary blocks) or trade-order shuffling of a stored run.

    Payload: ``run_id``, ``method`` (``bootstrap`` | ``trades``), ``n_paths``, ``mean_block``,
    ``horizon`` (bars), ``seed``.
    """
    run_id = str(payload["run_id"])
    record = services.store.get(run_id)
    result = services.results.result(run_id)
    method = str(payload.get("method", "bootstrap"))
    n_paths = int(payload.get("n_paths", 1000))
    seed = int(payload.get("seed", 7))
    progress(0.2, f"simulating {n_paths} paths")
    if method == "trades":
        trade_rets = result.trades.get_column("return").to_numpy()
        trade_rets = trade_rets[np.isfinite(trade_rets)]
        if len(trade_rets) < 2:
            raise ConfigError("Trade shuffling needs at least two trades")
        n_names = max(len(result.instruments), 1)
        paths = shuffle_trade_paths(trade_rets / n_names, n_paths, seed)
    else:
        r = result.returns[1:]
        horizon = int(payload.get("horizon", len(r)))
        paths = bootstrap_paths(r, n_paths, horizon, float(payload.get("mean_block", 5.0)), seed)
    if cancelled():
        raise JobCancelledError("Research job cancelled")
    stats_ = path_stats(paths)
    terminal, mdd, fan = stats_["terminal"], stats_["max_drawdown"], stats_["fan"]
    progress(0.8, "building charts")
    steps = np.arange(1, paths.shape[1] + 1)
    fan_chart = ChartSpec(
        id="monte_carlo_fan",
        title="Monte Carlo fan (growth of 1)",
        type=ChartType.LINE,
        x_axis=Axis(type=AxisType.VALUE, name="Step"),
        y_axes=[Axis(name="Growth of 1")],
        series=[
            ChartSeries(
                name=f"p{int(p)}",
                data=[[int(s), float(v)] for s, v in zip(steps, fan[k], strict=True)],
                role=SeriesRole.SERIES if p == 50 else SeriesRole.REFERENCE,
                dashed=p != 50,
            )
            for k, p in enumerate(FAN_PERCENTILES)
        ],
    )
    charts = [
        fan_chart,
        _hist("terminal_wealth", "Terminal wealth", terminal, "number"),
        _hist("mc_max_drawdown", "Max drawdown", mdd, "percent"),
    ]
    summary: dict[str, Any] = {
        "method": method,
        "paths": n_paths,
        "terminal_p5": float(np.percentile(terminal, 5)),
        "terminal_median": float(np.median(terminal)),
        "terminal_p95": float(np.percentile(terminal, 95)),
        "probability_of_loss": float((terminal < 1.0).mean()),
        "max_drawdown_median": float(np.median(mdd)),
        "max_drawdown_p5": float(np.percentile(mdd, 5)),
    }
    config = BacktestConfig.model_validate(record.config)
    return _record(
        services,
        config,
        "monte_carlo",
        {
            "source_run": run_id,
            "summary": summary,
            "tables": [],
            "charts": [c.model_dump(mode="json") for c in charts],
        },
        {
            "probability_of_loss": summary["probability_of_loss"],
            "max_drawdown": summary["max_drawdown_median"],
        },
    )


def _hist(chart_id: str, title: str, values: FloatArray, fmt: str) -> ChartSpec:
    counts, edges = np.histogram(values[np.isfinite(values)], bins=40)
    mids = (edges[:-1] + edges[1:]) / 2
    return ChartSpec(
        id=chart_id,
        title=f"{title} distribution",
        type=ChartType.BAR,
        x_axis=Axis(type=AxisType.VALUE, name=title, format=MetricFormat(fmt)),
        y_axes=[Axis(name="Paths", format=MetricFormat.INTEGER)],
        series=[
            ChartSeries(
                name="Paths", data=[[float(m), int(c)] for m, c in zip(mids, counts, strict=True)]
            )
        ],
    )


# --------------------------------------------------------------------------- sensitivity


def run_sensitivity(
    services: Services, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
) -> dict[str, Any]:
    """Cost multiples, extra execution delay or capital (capacity) sensitivity.

    Payload: ``config`` and ``kind``: ``cost`` (``multiples``), ``delay`` (``delays``) or
    ``capacity`` (``capitals``; adds square-root impact if no slippage model is set).
    """
    base = BacktestConfig.model_validate(payload["config"])
    kind = str(payload.get("kind", "cost"))
    ev = Evaluator(services, base, progress, cancelled)
    rows: list[dict[str, Any]] = []
    variants: list[tuple[Any, BacktestConfig, float]]
    if kind == "cost":
        multiples = [float(m) for m in payload.get("multiples", [0, 0.5, 1, 2, 3, 5, 10])]
        variants = [(m, ev.base, m) for m in multiples]
        label = "cost_multiple"
    elif kind == "delay":
        delays = [int(d) for d in payload.get("delays", [0, 1, 2, 3])]
        variants = [
            (
                d,
                ev.base.model_copy(
                    update={
                        "execution": ev.base.execution.model_copy(
                            update={"lag_bars": ev.base.execution.lag_bars + d}
                        )
                    }
                ),
                1.0,
            )
            for d in delays
        ]
        label = "extra_delay_bars"
    elif kind == "capacity":
        capitals = [float(c) for c in payload.get("capitals", [1e5, 1e6, 1e7, 1e8, 1e9])]
        cfg = ev.base
        if cfg.slippage is None:
            cfg = cfg.model_copy(update={"slippage": PluginRef(name="sqrt_impact")})
        variants = [(c, cfg.model_copy(update={"initial_capital": c}), 1.0) for c in capitals]
        label = "capital"
    else:
        raise ConfigError(f"Unknown sensitivity kind '{kind}'")
    curves = []
    for k, (value, cfg, scale) in enumerate(variants):
        if cancelled():
            raise JobCancelledError("Research job cancelled")
        res = ev.run(cfg, cost_scale=scale)
        metrics = quick_metrics(res)
        rows.append({label: value, **metrics})
        curves.append((f"{label}={value}", res.returns))
        progress(0.1 + 0.8 * (k + 1) / len(variants), f"variant {k + 1}/{len(variants)}")
    xs = [r[label] for r in rows]
    summary: dict[str, Any] = {"kind": kind, "variants": len(rows)}
    if kind == "cost":
        summary["breakeven_multiple_cagr"] = breakeven(xs, [r["cagr"] for r in rows])
        summary["breakeven_multiple_sharpe"] = breakeven(xs, [r["sharpe"] for r in rows])
    charts = [
        ChartSpec(
            id=f"{kind}_sensitivity",
            title=f"{kind.capitalize()} sensitivity",
            type=ChartType.LINE,
            x_axis=Axis(type=AxisType.CATEGORY, name=label, categories=[str(x) for x in xs]),
            y_axes=[
                Axis(name="Sharpe", format=MetricFormat.RATIO),
                Axis(name="CAGR", format=MetricFormat.PERCENT),
            ],
            series=[
                ChartSeries(
                    name="Sharpe",
                    data=[[str(x), r["sharpe"]] for x, r in zip(xs, rows, strict=True)],
                ),
                ChartSeries(
                    name="CAGR",
                    y_axis=1,
                    data=[[str(x), r["cagr"]] for x, r in zip(xs, rows, strict=True)],
                ),
            ],
        ),
        _equity_chart(f"Equity by {label}", curves, ev.prepared.data.timestamps, f"{kind}_equity"),
    ]
    return _record(
        services,
        base,
        f"sensitivity_{kind}",
        {
            "summary": summary,
            "tables": [_table(f"{kind.capitalize()} variants", rows, METRIC_FORMATS)],
            "charts": [c.model_dump(mode="json") for c in charts],
        },
        {"variants": float(len(rows))},
    )


# --------------------------------------------------------------------------- regimes


def regime_analysis(services: Services, run_id: str) -> dict[str, Any]:
    """Metrics by calendar year, benchmark trend (bull/bear) and volatility regime."""
    res = services.results.result(run_id)
    r, ppy = res.returns, res.periods_per_year
    years = res.timestamps.astype("datetime64[Y]").astype(int) + 1970
    groups: list[tuple[str, str, Any]] = [("year", str(y), years == y) for y in np.unique(years)]
    if res.benchmark_returns is not None:
        b = res.benchmark_returns
        level = np.cumprod(1 + b)
        sma = rolling_mean(level, REGIME_SMA)
        bull = level >= sma
        valid = np.isfinite(sma)
        groups += [("trend", "bull", valid & bull), ("trend", "bear", valid & ~bull)]
        vol = rolling_std(b, REGIME_VOL_WINDOW)
        ok = np.isfinite(vol)
        if ok.sum() > TERCILES:
            cuts = np.quantile(vol[ok], [1 / 3, 2 / 3])
            groups += [
                ("volatility", "low", ok & (vol <= cuts[0])),
                ("volatility", "mid", ok & (vol > cuts[0]) & (vol <= cuts[1])),
                ("volatility", "high", ok & (vol > cuts[1])),
            ]
    rows: list[dict[str, Any]] = []
    for dim, name, mask in groups:
        sub = r[mask]
        if len(sub) < 2:
            continue
        rows.append(
            {
                "dimension": dim,
                "regime": name,
                "bars": int(mask.sum()),
                "return": S.nan_to_none(S.total_return(sub)),
                "ann_return": S.nan_to_none(S.annualized_mean(sub, ppy)),
                "ann_vol": S.nan_to_none(S.annualized_vol(sub, ppy)),
                "sharpe": S.nan_to_none(S.sharpe(sub, ppy)),
                "max_drawdown": S.nan_to_none(S.max_drawdown(sub)),
                "benchmark_return": S.nan_to_none(S.total_return(res.benchmark_returns[mask]))
                if res.benchmark_returns is not None
                else None,
            }
        )
    return {
        "kind": "regimes",
        "source_run": run_id,
        "summary": {"regimes": len(rows)},
        "tables": [
            _table(
                "Metrics by regime",
                rows,
                {
                    **METRIC_FORMATS,
                    "return": "percent",
                    "ann_return": "percent",
                    "benchmark_return": "percent",
                    "bars": "integer",
                },
            )
        ],
        "charts": [],
    }


def unlock_test_period(services: Services, run_id: str) -> dict[str, Any]:
    """Clone a run with the test period unlocked (the unlock is recorded)."""
    record = services.store.get(run_id)
    cfg = BacktestConfig.model_validate(record.config)
    if cfg.split is None:
        raise ConfigError("This run has no train/validation/test split")
    unlocked = cfg.model_copy(
        update={"split": cfg.split.model_copy(update={"test_unlocked": True})}
    )
    services.store.record_test_unlock(cfg.experiment or "", run_id)
    return copy.deepcopy(unlocked.model_dump(mode="json"))
